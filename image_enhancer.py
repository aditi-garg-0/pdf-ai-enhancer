from __future__ import annotations

import gc
import io
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import time
import math
import sys

import numpy as np
from loguru import logger

# Optional heavy deps
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except Exception:
    REALESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN not available — using bicubic fallback")

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except Exception:
    GFPGAN_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

POTRACE_AVAILABLE = shutil.which("potrace") is not None

# Local config imports (guarded)
try:
    from config import Config, ImageEnhancementConfig, MODELS_DIR, REALESRGAN_MODEL_URLS, GFPGAN_MODEL_URLS
except Exception:
    @dataclass
    class ImageEnhancementConfig:
        enable_super_resolution: bool = False
        sr_outscale: float = 4.0
        output_format: str = "PNG"
        png_compress: int = 1
        jpeg_quality: int = 95
        enable_vectorization: bool = False
        vector_raster_scale: int = 4
        enable_face_restoration: bool = False
        enable_denoising: bool = True
        denoise_strength: float = 8.0
        enable_clahe: bool = True
        clahe_clip_limit: float = 3.0
        clahe_grid_size: int = 8
        remove_paper_texture: bool = True
        enable_sharpening: bool = True
        sharpen_sigma: float = 1.0
        sharpen_amount: float = 0.9
        final_unsharp: bool = True
        final_unsharp_sigma: float = 0.8
        final_unsharp_amount: float = 0.6
        enable_vectorization_cli: bool = False
        # Compression / sizing defaults tuned for high quality
        max_image_pixels: int = 6_000_000   # allow up to ~6MP before downscaling
        max_image_width: int = 3500
        max_image_height: int = 3500
        prefer_webp: bool = True
        webp_quality: int = 95
        jpeg_quality: int = 95

    class Config:
        enhancement = ImageEnhancementConfig()
        pipeline = type("P", (), {"output_dpi": 300})()

    MODELS_DIR = Path("/tmp/models")
    REALESRGAN_MODEL_URLS = {}
    GFPGAN_MODEL_URLS = {}

# Import extractor.ImageBlock if available; otherwise define a compatible stand-in
try:
    from extractor import ImageBlock
except Exception:
    @dataclass
    class ImageBlock:
        image_bytes: bytes
        bbox: Tuple[float, float, float, float]
        page_num: int
        xref: int
        width: int = 0
        height: int = 0
        dpi_x: float = 72.0
        dpi_y: float = 72.0
        colorspace: str = "RGB"
        ext: str = "png"
        sha256: str = ""
        is_diagram: bool = False
        is_photo: bool = False

Rect = Tuple[int, int, int, int]

# Debug helper directory (used to reproduce debug_enhanced_p1_xref5 behavior)
DEBUG_DIR = Path("/tmp/pdf_enhancer_debug")
try:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

def _dbg_save(arr: Optional[np.ndarray], tag: str, block: Optional[ImageBlock] = None):
    """
    Save a small JPEG preview and log basic stats for instrumentation.
    This helper was used to produce debug_enhanced_p1_xref5.
    """
    try:
        if arr is None:
            logger.info(f"[DBG] {tag}: arr is None")
            return
        a = np.clip(arr, 0, 255).astype(np.uint8)
        mean = float(a.mean())
        mn = float(a.min()); mx = float(a.max())
        logger.info(f"[DBG] {tag}: mean={mean:.2f} min={mn:.2f} max={mx:.2f}")
        if PIL_AVAILABLE:
            pil = Image.fromarray(a)
            fname = DEBUG_DIR / f"{int(time.time()*1000)}_{tag}_p{getattr(block,'page_num','NA')}_xref{getattr(block,'xref','NA')}.jpg"
            pil.thumbnail((1200, 1200))
            pil.save(str(fname), format="JPEG", quality=90)
    except Exception as e:
        logger.debug(f"[DBG] failed {tag}: {e}")

class ImageEnhancer:
    """
    High-fidelity image enhancer tuned for logos, diagrams, and small text.
    Defensive: never replaces a valid image with an invalid raster; logs and saves
    intermediate steps to /tmp/pdf_enhancer_debug for inspection.
    """

    def __init__(self, config: Config):
        self.cfg: ImageEnhancementConfig = config.enhancement if hasattr(config, "enhancement") else getattr(config, "enhancement", ImageEnhancementConfig())
        self._sr_model = None
        self._face_model = None

    # -------------------------
    # Public API
    # -------------------------

    def enhance_block(self, block: ImageBlock) -> ImageBlock:
        """
        Enhance an ImageBlock and return an ImageBlock-compatible object.
        Instrumentation (_dbg_save) was used to create debug_enhanced_p1_xref5.
        """
        # Defensive: obtain bytes
        candidate_bytes = None
        try:
            candidate_bytes = getattr(block, "image_bytes", None)
        except Exception:
            candidate_bytes = None

        arr = None
        if candidate_bytes:
            arr = self._bytes_to_numpy(candidate_bytes)
        _dbg_save(arr, "decoded", block)

        # If decode failed, try block helpers
        if arr is None:
            try:
                if hasattr(block, "to_pil"):
                    pil = block.to_pil()
                    arr = np.array(pil.convert("RGB"))
                elif hasattr(block, "to_numpy"):
                    arr = block.to_numpy()
            except Exception:
                arr = None
        _dbg_save(arr, "after_helpers", block)

        # If still None, create a white placeholder
        if arr is None:
            w = int(getattr(block, "width", 512) or 512)
            h = int(getattr(block, "height", 512) or 512)
            arr = np.ones((h, w, 3), dtype=np.uint8) * 255
            logger.debug(f"Enhancer: using placeholder image for xref={getattr(block,'xref',None)}")

        # Basic classification heuristics
        is_diagram = bool(getattr(block, "is_diagram", False))
        is_photo = bool(getattr(block, "is_photo", False))

        try:
            # 1. Remove marks conservatively
            if CV2_AVAILABLE:
                arr = self._remove_marks_preserve_color(arr)
                _dbg_save(arr, "after_remove_marks", block)

            # 2. Preprocess
            if CV2_AVAILABLE:
                arr = self._preprocess_preserve_color(arr, is_diagram=is_diagram, is_logo=False, is_photo=is_photo)
                _dbg_save(arr, "after_preprocess", block)

            # 3. Pre-upscale small text
            if CV2_AVAILABLE:
                arr = self._pre_upscale_small_text_regions(arr)
                _dbg_save(arr, "after_pre_upscale", block)

            # 4. Vectorize logos/diagrams if configured (DEFENSIVE)
            if getattr(self.cfg, "enable_vectorization", False) and (is_diagram or is_photo is False):
                svg_bytes = self._try_vectorize(arr)
                if svg_bytes is not None and CAIROSVG_AVAILABLE:
                    try:
                        scale = int(getattr(self.cfg, "vector_raster_scale", 4))
                        png = cairosvg.svg2png(bytestring=svg_bytes, output_width=arr.shape[1] * scale, output_height=arr.shape[0] * scale)
                        arr2 = self._bytes_to_numpy(png)
                        # Defensive checks: arr2 must be valid and not near-empty
                        if arr2 is not None and arr2.size and arr2.ndim == 3 and arr2.shape[2] >= 3:
                            mean_val = float(arr2.mean())
                            max_val = float(arr2.max())
                            if max_val > 1 and mean_val > 1.0:
                                # Good raster: resize back to original size
                                if CV2_AVAILABLE:
                                    arr = cv2.resize(arr2, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                                else:
                                    if PIL_AVAILABLE:
                                        pil = Image.fromarray(arr2.astype(np.uint8))
                                        pil = pil.resize((arr.shape[1], arr.shape[0]), resample=Image.LANCZOS)
                                        arr = np.array(pil)
                                is_diagram = True
                                _dbg_save(arr, "after_vectorize", block)
                            else:
                                logger.debug("Vectorization produced near-empty raster; skipping replacement")
                        else:
                            logger.debug("Vectorization raster decode failed or invalid; skipping replacement")
                    except Exception as e:
                        logger.debug(f"Vectorization rasterization error (skipping): {e}")

            # 5. Super-resolution
            if getattr(self.cfg, "enable_super_resolution", False):
                arr = self._super_resolve(arr)
                _dbg_save(arr, "after_sr", block)

            # 6. Face restoration for photos
            if getattr(self.cfg, "enable_face_restoration", False) and is_photo and not is_diagram:
                arr = self._restore_faces(arr)
                _dbg_save(arr, "after_face_restore", block)

            # 7. Postprocess
            arr = self._postprocess_preserve_color(arr, is_diagram=is_diagram, is_logo=False)
            _dbg_save(arr, "after_postprocess", block)

        except Exception as e:
            logger.debug(f"Enhancement pipeline error: {e}")

        # Convert to bytes with robust handling and ext sniffing
        try:
            out_bytes = self._to_bytes(arr, block=block)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            if not out_bytes:
                placeholder = np.ones((h, w, 3), dtype=np.uint8) * 255
                out_bytes = self._to_bytes(placeholder, block=block)
        except Exception as e:
            logger.warning(f"Failed to serialize enhanced image: {e}")
            h = int(arr.shape[0]) if arr is not None else 256
            w = int(arr.shape[1]) if arr is not None else 256
            placeholder = np.ones((h, w, 3), dtype=np.uint8) * 255
            out_bytes = self._to_bytes(placeholder, block=block)

        # Sniff actual bytes to determine extension (prevents embedding mismatches)
        try:
            b = out_bytes or b""
            if b.startswith(b"\x89PNG\r\n\x1a\n"):
                actual_ext = "png"
            elif b.startswith(b"\xff\xd8"):
                actual_ext = "jpg"
            elif b[:4] == b"RIFF" and b[8:12] == b"WEBP":
                actual_ext = "webp"
            else:
                actual_ext = getattr(self.cfg, "output_format", "PNG").lower()
        except Exception:
            actual_ext = getattr(self.cfg, "output_format", "PNG").lower()

        # Build ImageBlock-compatible return
        try:
            ib_cls = ImageBlock
            enhanced_block = ib_cls(
                image_bytes = out_bytes,
                bbox = getattr(block, "bbox", (0, 0, w, h)),
                page_num = getattr(block, "page_num", 0),
                xref = getattr(block, "xref", -1),
                width = w,
                height = h,
                dpi_x = getattr(block, "dpi_x", 72.0) * (w / (getattr(block, "width", w) or w)),
                dpi_y = getattr(block, "dpi_y", 72.0) * (h / (getattr(block, "height", h) or h)),
                colorspace = "RGB",
                ext = actual_ext,
                sha256 = "",
                is_diagram = bool(is_diagram),
                is_photo = bool(is_photo),
            )
        except Exception:
            class _Simple:
                pass
            enhanced_block = _Simple()
            enhanced_block.image_bytes = out_bytes
            enhanced_block.bbox = getattr(block, "bbox", (0, 0, w, h))
            enhanced_block.page_num = getattr(block, "page_num", 0)
            enhanced_block.xref = getattr(block, "xref", -1)
            enhanced_block.width = w
            enhanced_block.height = h
            enhanced_block.dpi_x = getattr(block, "dpi_x", 72.0)
            enhanced_block.dpi_y = getattr(block, "dpi_y", 72.0)
            enhanced_block.colorspace = "RGB"
            enhanced_block.ext = actual_ext
            enhanced_block.sha256 = ""
            enhanced_block.is_diagram = bool(is_diagram)
            enhanced_block.is_photo = bool(is_photo)

        return enhanced_block

    def enhance_page_render(self, img: np.ndarray) -> np.ndarray:
        img = self._remove_marks_preserve_color(img) if CV2_AVAILABLE else img
        img = self._preprocess_preserve_color(img, is_diagram=False, is_logo=False, is_photo=False) if CV2_AVAILABLE else img
        if getattr(self.cfg, "enable_super_resolution", False):
            img = self._super_resolve(img)
        img = self._postprocess_preserve_color(img, is_diagram=False, is_logo=False)
        return img

    # -------------------------
    # Utilities: decoding / encoding
    # -------------------------

    def _bytes_to_numpy(self, bts: bytes) -> Optional[np.ndarray]:
        if not bts:
            return None
        if PIL_AVAILABLE:
            try:
                pil = Image.open(io.BytesIO(bts)).convert("RGB")
                return np.array(pil)
            except Exception:
                pass
        if CV2_AVAILABLE:
            try:
                arr = np.frombuffer(bts, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            except Exception:
                return None
        return None

    # -------------------------
    # Smart downscale + compression helpers
    # -------------------------

    def _get_cfg_val(self, name, default):
        try:
            return getattr(self.cfg, name)
        except Exception:
            return default

    def _maybe_downscale_and_compress(self, img: np.ndarray, block=None) -> (bytes, str):
        """
        Decide format and compress. Returns (bytes, ext) where ext in {'png','jpg','webp'}.
        Behavior:
          - If image is diagram-like (block.is_diagram True) -> prefer PNG (quantize if available).
          - Else prefer WebP (if enabled) or JPEG for photos.
          - Downscale only if image exceeds configured max pixels or dimensions.
        """
        if img is None:
            return b"", "png"

        # Configurable thresholds
        max_pixels = int(self._get_cfg_val("max_image_pixels", getattr(self.cfg, "max_image_pixels", 6_000_000)))
        max_width = int(self._get_cfg_val("max_image_width", getattr(self.cfg, "max_image_width", 3500)))
        max_height = int(self._get_cfg_val("max_image_height", getattr(self.cfg, "max_image_height", 3500)))
        prefer_webp = bool(self._get_cfg_val("prefer_webp", getattr(self.cfg, "prefer_webp", True)))
        jpeg_quality = int(self._get_cfg_val("jpeg_quality", getattr(self.cfg, "jpeg_quality", 95)))
        webp_quality = int(self._get_cfg_val("webp_quality", getattr(self.cfg, "webp_quality", 95)))
        png_compress = int(self._get_cfg_val("png_compress", getattr(self.cfg, "png_compress", 1)))
        pngquant_path = shutil.which("pngquant")

        # Ensure uint8 RGB
        try:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[2] == 4:
                img = img[..., :3]
        except Exception:
            pass

        h, w = img.shape[:2]
        pixels = int(h) * int(w)

        # Downscale if too large (conservative)
        scale = 1.0
        if pixels > max_pixels or w > max_width or h > max_height:
            scale_w = max_width / float(w) if w > max_width else 1.0
            scale_h = max_height / float(h) if h > max_height else 1.0
            scale_px = (max_pixels / float(pixels)) ** 0.5 if pixels > max_pixels else 1.0
            scale = min(scale_w, scale_h, scale_px, 1.0)
            if scale < 1.0:
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                if CV2_AVAILABLE:
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                elif PIL_AVAILABLE:
                    img = np.array(Image.fromarray(img).resize((new_w, new_h), resample=Image.LANCZOS))
                h, w = img.shape[:2]

        # Decide format
        is_diagram = bool(getattr(block, "is_diagram", False))
        is_photo = bool(getattr(block, "is_photo", False))
        # Heuristic: if many unique colors and photographic, treat as photo
        try:
            if not is_diagram and not is_photo:
                uniq = len(np.unique(img.reshape(-1, 3), axis=0))
                if uniq > 2000:
                    is_photo = True
        except Exception:
            pass

        # Encode
        if is_diagram:
            # Prefer PNG for diagrams; optionally quantize with pngquant
            if PIL_AVAILABLE:
                pil = Image.fromarray(img)
                buf = io.BytesIO()
                pil.save(buf, format="PNG", compress_level=png_compress)
                data = buf.getvalue()
                # Try pngquant for extra compression (best for flat colors)
                if pngquant_path:
                    try:
                        p = subprocess.Popen([pngquant_path, "--quality=60-95", "--speed=1", "--output", "-", "--force", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                        out, _ = p.communicate(data)
                        if out and len(out) > 0:
                            return out, "png"
                    except Exception:
                        pass
                return data, "png"
            else:
                try:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_compress])
                    if ok:
                        return buf.tobytes(), "png"
                except Exception:
                    pass
                return b"", "png"

        # For photos: prefer WebP if allowed, else JPEG
        if prefer_webp and PIL_AVAILABLE:
            try:
                pil = Image.fromarray(img)
                buf = io.BytesIO()
                pil.save(buf, format="WEBP", quality=webp_quality, method=6)
                data = buf.getvalue()
                if data and len(data) > 0:
                    return data, "webp"
            except Exception:
                pass

        # JPEG fallback
        if PIL_AVAILABLE:
            try:
                pil = Image.fromarray(img)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
                data = buf.getvalue()
                return data, "jpg"
            except Exception:
                pass

        if CV2_AVAILABLE:
            try:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ok:
                    return buf.tobytes(), "jpg"
            except Exception:
                pass

        # Last resort: lossless PNG via PIL or empty bytes
        if PIL_AVAILABLE:
            try:
                pil = Image.fromarray(img)
                buf = io.BytesIO()
                pil.save(buf, format="PNG", compress_level=png_compress)
                return buf.getvalue(), "png"
            except Exception:
                pass

        return b"", "png"

    def _to_bytes(self, img: np.ndarray, prefer_ext: Optional[str] = None, block=None) -> bytes:
        """
        Convert numpy RGB array to bytes using smart compression. Returns bytes.
        prefer_ext can be 'png','jpg','webp' to force a format.
        """
        if img is None:
            return b""
        try:
            # Ensure uint8 and handle channels
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
        except Exception:
            return b""

        # If image has alpha channel, composite over white to avoid gray matte when embedding
        try:
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                # img is RGBA uint8
                if PIL_AVAILABLE:
                    pil_tmp = Image.fromarray(img, mode="RGBA")
                    bg = Image.new("RGB", pil_tmp.size, (255, 255, 255))
                    bg.paste(pil_tmp, mask=pil_tmp.split()[3])  # paste using alpha as mask
                    img = np.array(bg)
                else:
                    # CV2 path: alpha blending onto white
                    alpha = img[..., 3].astype(np.float32) / 255.0
                    rgb = img[..., :3].astype(np.float32)
                    white = np.ones_like(rgb, dtype=np.float32) * 255.0
                    comp = (rgb * alpha[..., None]) + (white * (1.0 - alpha[..., None]))
                    img = np.clip(comp, 0, 255).astype(np.uint8)
        except Exception:
            pass

        # Ensure final array is 3-channel RGB uint8
        try:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            if img.ndim == 3 and img.shape[2] == 3:
                img = np.clip(img, 0, 255).astype(np.uint8)
        except Exception:
            return b""

        # Honor forced extension if provided
        if prefer_ext in ("png", "jpg", "jpeg", "webp"):
            try:
                if prefer_ext in ("jpg", "jpeg"):
                    q = int(self._get_cfg_val("jpeg_quality", getattr(self.cfg, "jpeg_quality", 95)))
                    if PIL_AVAILABLE:
                        pil = Image.fromarray(img)
                        buf = io.BytesIO()
                        pil.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                        return buf.getvalue()
                if prefer_ext == "webp" and PIL_AVAILABLE:
                    q = int(self._get_cfg_val("webp_quality", getattr(self.cfg, "webp_quality", 95)))
                    pil = Image.fromarray(img)
                    buf = io.BytesIO()
                    pil.save(buf, format="WEBP", quality=q, method=6)
                    return buf.getvalue()
                if prefer_ext == "png" and PIL_AVAILABLE:
                    pil = Image.fromarray(img)
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG", compress_level=int(self._get_cfg_val("png_compress", getattr(self.cfg, "png_compress", 1))))
                    return buf.getvalue()
            except Exception:
                pass

        # Use smart compressor
        try:
            data, ext = self._maybe_downscale_and_compress(img, block=block)
            if data:
                return data
        except Exception as e:
            logger.debug(f"_to_bytes compress fallback error: {e}")

        # Fallback: always return a safe PNG (guarantees embeddable bytes)
        try:
            if PIL_AVAILABLE:
                pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                buf = io.BytesIO()
                pil.save(buf, format="PNG", compress_level=int(self._get_cfg_val("png_compress", getattr(self.cfg, "png_compress", 1))))
                return buf.getvalue()
        except Exception as e:
            logger.debug(f"_to_bytes final PNG fallback failed: {e}")

        return b""

    # -------------------------
    # Mark removal (conservative)
    # -------------------------

    def _remove_marks_preserve_color(self, img: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return img
        try:
            h, w = img.shape[:2]
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]
            l_blur = cv2.GaussianBlur(l, (9, 9), 0)
            local_mean = cv2.blur(l_blur, (31, 31))
            diff = cv2.absdiff(l_blur, local_mean)
            diff_thresh = getattr(self.cfg, "mark_diff_thresh", 18)
            _, mask1 = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            sat_thresh = getattr(self.cfg, "mark_sat_thresh", 30)
            _, mask2 = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.bitwise_and(mask1, mask2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_dil = cv2.dilate(edges, kernel, iterations=2)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges_dil))
            if mask.sum() < (getattr(self.cfg, "mark_min_pixels_frac", 0.0005) * h * w):
                return img
            inpaint_flag = cv2.INPAINT_TELEA if getattr(self.cfg, "inpaint_method", "telea").lower() != "ns" else cv2.INPAINT_NS
            inpainted = cv2.inpaint(img, mask.astype(np.uint8), 3, inpaint_flag)
            try:
                lab_orig = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab_inp = cv2.cvtColor(inpainted, cv2.COLOR_RGB2LAB)
                a = np.where(mask[..., None] == 255, lab_orig[..., 1:2], lab_inp[..., 1:2])
                b = np.where(mask[..., None] == 255, lab_orig[..., 2:3], lab_inp[..., 2:3])
                l_channel = lab_inp[..., 0:1]
                lab_merged = np.concatenate([l_channel, a, b], axis=2)
                merged = cv2.cvtColor(lab_merged.astype(np.uint8), cv2.COLOR_LAB2RGB)
                return merged
            except Exception:
                return inpainted
        except Exception as e:
            logger.debug(f"Mark removal error: {e}")
            return img

    # -------------------------
    # Preprocessing (color-preserving)
    # -------------------------

    def _preprocess_preserve_color(self, img: np.ndarray, is_diagram: bool, is_logo: bool, is_photo: bool) -> np.ndarray:
        if not CV2_AVAILABLE:
            return img
        try:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if is_logo or is_diagram:
                try:
                    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
                except Exception:
                    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            if getattr(self.cfg, "enable_denoising", True):
                h = getattr(self.cfg, "denoise_strength", 8.0)
                if is_diagram:
                    h = min(h * 1.2, 20.0)
                try:
                    img = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
                except Exception:
                    pass
            if getattr(self.cfg, "enable_clahe", True):
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=getattr(self.cfg, "clahe_clip_limit", 3.0),
                                            tileGridSize=(getattr(self.cfg, "clahe_grid_size", 8),
                                                          getattr(self.cfg, "clahe_grid_size", 8)))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                except Exception:
                    pass
            if getattr(self.cfg, "remove_paper_texture", True) and not is_photo:
                try:
                    blurred = cv2.GaussianBlur(img, (51, 51), 0)
                    diff = cv2.subtract(img, blurred)
                    img = cv2.addWeighted(img, 0.85, diff, 0.15, 0)
                except Exception:
                    pass
            if getattr(self.cfg, "enable_sharpening", True):
                try:
                    sigma = getattr(self.cfg, "sharpen_sigma", 1.0)
                    amount = getattr(self.cfg, "sharpen_amount", 0.9)
                    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
                    img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
                except Exception:
                    pass
            if is_diagram:
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    l = lab[:, :, 0]
                    th = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
                    mask = th == 255
                    l2 = np.where(mask, 255, l)
                    lab[:, :, 0] = l2
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                except Exception:
                    pass
            return img
        except Exception as e:
            logger.debug(f"Preprocess error: {e}")
            return img

    # -------------------------
    # Pre-upscale small text regions (before SR)
    # -------------------------

    def _pre_upscale_small_text_regions(self, img: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return img
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            try:
                mser = cv2.MSER_create(_delta=5, _min_area=20, _max_area=2000)
            except Exception:
                mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            mask = np.zeros_like(gray)
            for r in regions:
                hull = cv2.convexHull(r.reshape(-1, 1, 2))
                cv2.drawContours(mask, [hull], -1, 255, -1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = img.copy()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 6 or h < 6 or w * h > 20000:
                    continue
                ar = w / max(h, 1)
                if ar > 12 or ar < 0.08:
                    continue
                region = img[y:y + h, x:x + w]
                try:
                    up = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                    lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(max(4, w // 4), max(4, h // 4)))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    up = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    blurred = cv2.GaussianBlur(up, (0, 0), 0.6)
                    up = cv2.addWeighted(up, 1.6, blurred, -0.6, 0)
                    up_down = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC)
                    out[y:y + h, x:x + w] = up_down
                except Exception:
                    pass
            return out
        except Exception as e:
            logger.debug(f"Pre-upscale small text error: {e}")
            return img

    # -------------------------
    # Super-resolution helpers
    # -------------------------

    def _super_resolve(self, img: np.ndarray) -> np.ndarray:
        if not REALESRGAN_AVAILABLE:
            return self._bicubic_upscale(img)
        try:
            upsampler = self._get_sr_model()
            out, _ = upsampler.enhance(img, outscale=getattr(self.cfg, "sr_outscale", 4.0))
            if isinstance(out, np.ndarray):
                out = np.clip(out, 0, 255).astype(np.uint8)
            return out
        except RuntimeError as e:
            logger.warning(f"SR OOM: {e} — falling back to bicubic")
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            return self._bicubic_upscale(img)
        except Exception as e:
            logger.debug(f"SR error: {e}")
            return self._bicubic_upscale(img)

    def _bicubic_upscale(self, img: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return img
        h, w = img.shape[:2]
        scale = int(getattr(self.cfg, "sr_outscale", 4.0))
        if scale <= 1:
            return img
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    def _get_sr_model(self) -> "RealESRGANer":
        if self._sr_model is not None:
            return self._sr_model
        model_name = getattr(self.cfg, "sr_model", "RealESRGAN_x4plus")
        model_path = MODELS_DIR / f"{model_name}.pth"
        if not model_path.exists():
            url = REALESRGAN_MODEL_URLS.get(model_name)
            if url:
                urllib.request.urlretrieve(url, str(model_path))
            else:
                raise FileNotFoundError(f"SR model not found: {model_name}")
        if "anime" in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        elif "x2" in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        gpu_id = getattr(self.cfg, "sr_gpu_id", None)
        device_str = f"cuda:{gpu_id}" if (REALESRGAN_AVAILABLE and torch.cuda.is_available() and gpu_id is not None) else "cpu"
        self._sr_model = RealESRGANer(
            scale=4 if "x2" not in model_name else 2,
            model_path=str(model_path),
            model=model,
            tile=getattr(self.cfg, "sr_tile", 512),
            tile_pad=getattr(self.cfg, "sr_tile_pad", 10),
            pre_pad=0,
            half=getattr(self.cfg, "sr_half", False) and device_str.startswith("cuda"),
            device=device_str,
        )
        logger.info(f"Real-ESRGAN loaded: {model_name} on {device_str}")
        return self._sr_model

    # -------------------------
    # Face restoration
    # -------------------------

    def _restore_faces(self, img: np.ndarray) -> np.ndarray:
        if not GFPGAN_AVAILABLE:
            return img
        try:
            restorer = self._get_face_model()
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, _, output = restorer.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.debug(f"Face restore error: {e}")
            return img

    def _get_face_model(self) -> "GFPGANer":
        if self._face_model is not None:
            return self._face_model
        model_name = getattr(self.cfg, "face_model", "GFPGANv1.4")
        model_path = MODELS_DIR / f"{model_name}.pth"
        if not model_path.exists():
            url = GFPGAN_MODEL_URLS.get(model_name)
            if url:
                urllib.request.urlretrieve(url, str(model_path))
        self._face_model = GFPGANer(model_path=str(model_path), upscale=1, arch="clean", channel_multiplier=2)
        return self._face_model

    # -------------------------
    # Vectorization helper
    # -------------------------

    def _try_vectorize(self, img: np.ndarray) -> Optional[bytes]:
        if not POTRACE_AVAILABLE or not getattr(self.cfg, "enable_vectorization", False):
            return None
        if not CV2_AVAILABLE:
            return None
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tmp_pbm = Path("/tmp") / f"veclocal_{np.random.randint(1e9)}.pbm"
            tmp_svg = tmp_pbm.with_suffix(".svg")
            cv2.imwrite(str(tmp_pbm), th)
            subprocess.run(["potrace", "-s", "-o", str(tmp_svg), str(tmp_pbm)], check=True, timeout=20)
            svg_bytes = tmp_svg.read_bytes()
            tmp_pbm.unlink(missing_ok=True)
            tmp_svg.unlink(missing_ok=True)
            return svg_bytes
        except Exception as e:
            logger.debug(f"Vectorization failed: {e}")
            return None

    # -------------------------
    # Postprocessing (color-preserving)
    # -------------------------

    def _postprocess_preserve_color(self, img: np.ndarray, is_diagram: bool, is_logo: bool) -> np.ndarray:
        try:
            if img is None:
                return img
            # Keep a copy for fallback if postprocess collapses values
            try:
                pre_img = img.copy()
            except Exception:
                pre_img = None

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # Heuristic: convert BGR->RGB if channel ordering looks swapped
            if CV2_AVAILABLE and img.ndim == 3 and img.shape[2] == 3:
                try:
                    ch0_mean = float(img[..., 0].mean())
                    ch2_mean = float(img[..., 2].mean())
                    if ch0_mean > ch2_mean * 1.5:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass

            # Gentle PIL adjustments
            if PIL_AVAILABLE:
                try:
                    pil = Image.fromarray(img)
                    sat = max(0.95, float(getattr(self.cfg, "saturation_boost", 1.05)))
                    bri = max(0.98, float(getattr(self.cfg, "brightness_boost", 1.02)))
                    pil = ImageEnhance.Color(pil).enhance(sat)
                    pil = ImageEnhance.Brightness(pil).enhance(bri)
                    img = np.array(pil)
                except Exception:
                    pass

            # Safe unsharp
            try:
                if CV2_AVAILABLE and getattr(self.cfg, "final_unsharp", True):
                    sigma = float(getattr(self.cfg, "final_unsharp_sigma", 0.8))
                    amount = float(getattr(self.cfg, "final_unsharp_amount", 0.6))
                    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
                    img_f = cv2.addWeighted(img.astype(np.float32), 1.0 + amount, blurred.astype(np.float32), -amount, 0.0)
                    img = np.clip(img_f, 0, 255).astype(np.uint8)
            except Exception:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # Safety: if processed image is near-black, return pre_img (last good)
            try:
                mean_val = float(img.mean())
                if mean_val < 6.0:
                    logger.warning("Postprocess produced near-black image; returning preprocessed fallback")
                    if pre_img is not None:
                        return np.clip(pre_img, 0, 255).astype(np.uint8)
                    else:
                        return np.clip(img, 0, 255).astype(np.uint8)
            except Exception:
                pass

            return np.clip(img, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.debug(f"Postprocess error safe fallback: {e}")
            try:
                return np.clip(img, 0, 255).astype(np.uint8)
            except Exception:
                return np.ones((16, 16, 3), dtype=np.uint8) * 255

    # -------------------------
    # Quality metrics
    # -------------------------

    def _compute_quality(self, original: np.ndarray, enhanced: np.ndarray) -> dict:
        metrics = {}
        if not SKIMAGE_AVAILABLE or not CV2_AVAILABLE:
            return metrics
        try:
            h, w = enhanced.shape[:2]
            orig_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)
            orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            metrics["ssim"] = float(ssim_fn(orig_gray, enh_gray, data_range=255))
            metrics["psnr"] = float(psnr_fn(orig_gray, enh_gray, data_range=255))
            orig_edges = cv2.Canny(orig_gray, 50, 150)
            enh_edges = cv2.Canny(enh_gray, 50, 150)
            try:
                metrics["edge_ssim"] = float(ssim_fn(orig_edges, enh_edges, data_range=255))
            except Exception:
                metrics["edge_ssim"] = 0.0
            metrics["lap_var"] = float(cv2.Laplacian(enh_gray, cv2.CV_64F).var())
            metrics["mean_diff"] = float(np.mean(np.abs(orig_resized.astype(np.float32) - enhanced.astype(np.float32))))
        except Exception as e:
            logger.debug(f"Quality compute error: {e}")
        return metrics

    # -------------------------
    # Cleanup
    # -------------------------

    def cleanup(self):
        self._sr_model = None
        self._face_model = None
        gc.collect()
        if REALESRGAN_AVAILABLE:
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

# -------------------------
# LineRegularizer (utility)
# -------------------------

class LineRegularizer:
    @staticmethod
    def angle_of_segment(x1, y1, x2, y2) -> float:
        return math.atan2((y2 - y1), (x2 - x1))

    @staticmethod
    def segment_length(x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def merge_collinear_segments(segments, angle_tol=0.05, dist_tol=10):
        if not segments:
            return []
        used = [False] * len(segments)
        merged = []
        for i, s in enumerate(segments):
            if used[i]:
                continue
            x1, y1, x2, y2 = s
            ai = LineRegularizer.angle_of_segment(x1, y1, x2, y2)
            group = [s]
            used[i] = True
            for j in range(i + 1, len(segments)):
                if used[j]:
                    continue
                x3, y3, x4, y4 = segments[j]
                aj = LineRegularizer.angle_of_segment(x3, y3, x4, y4)
                if abs((ai - aj + math.pi) % math.pi - math.pi/2) < angle_tol or abs(ai - aj) < angle_tol:
                    mid_i = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    mid_j = ((x3 + x4) / 2.0, (y3 + y4) / 2.0)
                    if math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1]) < dist_tol:
                        group.append(segments[j])
                        used[j] = True
            xs = []
            ys = []
            for (a, b, c, d) in group:
                xs.extend([a, c])
                ys.extend([b, d])
            merged.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return merged

    @staticmethod
    def extend_segment_to_edges(seg, edges_mask):
        x1, y1, x2, y2 = seg
        h, w = edges_mask.shape[:2]
        angle = LineRegularizer.angle_of_segment(x1, y1, x2, y2)
        dx = math.cos(angle)
        dy = math.sin(angle)
        fx, fy = x2, y2
        while 0 <= int(fx) < w and 0 <= int(fy) < h and edges_mask[int(fy), int(fx)] != 0:
            fx += dx
            fy += dy
        bx, by = x1, y1
        while 0 <= int(bx) < w and 0 <= int(by) < h and edges_mask[int(by), int(bx)] != 0:
            bx -= dx
            by -= dy
        return (int(max(0, bx)), int(max(0, by)), int(min(w - 1, fx)), int(min(h - 1, fy)))

# -------------------------
# CLI for local testing
# -------------------------

def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="ImageEnhancer quick test")
    parser.add_argument("input", help="Input image file (PNG/JPEG)")
    parser.add_argument("output", help="Output image file (PNG/JPEG)")
    parser.add_argument("--sr", action="store_true", help="Enable SR (if available)")
    args = parser.parse_args()

    cfg = Config() if 'Config' in globals() else Config()
    cfg.enhancement.enable_super_resolution = args.sr
    enh = ImageEnhancer(cfg)
    if not PIL_AVAILABLE:
        print("PIL not available; cannot run CLI.")
        return
    img = Image.open(args.input).convert("RGB")
    arr = np.array(img)
    res = enh.enhance_page_render(arr)
    out_pil = Image.fromarray(res.astype(np.uint8))
    out_pil.save(args.output)
    print("Saved", args.output)

if __name__ == "__main__":
    try:
        _cli_main()
    except Exception as e:
        logger.exception(f"CLI error: {e}")
