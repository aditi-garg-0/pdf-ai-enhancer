from __future__ import annotations

import gc
import io
import shutil
import subprocess
import urllib.request
import zipfile
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

# basicsr==1.4.2 imports torchvision.transforms.functional_tensor, which was
# removed in torchvision>=0.17 (requirements.txt pins torchvision==0.18.1).
# Without this shim the basicsr import raises ImportError, REALESRGAN_AVAILABLE
# silently becomes False, and every image gets a plain bicubic upscale instead
# of the actual GAN super-resolution -- this is the direct cause of the
# "Real-ESRGAN not available -- using bicubic fallback" log line and the
# blurry/soft output quality. The shim re-exposes the one function basicsr
# needs under its old (pre-0.17) location, so basicsr imports cleanly without
# downgrading torch/torchvision.
try:
    import sys as _sys
    import types as _types
    if "torchvision.transforms.functional_tensor" not in _sys.modules:
        import torchvision.transforms.functional as _tv_functional
        _shim = _types.ModuleType("torchvision.transforms.functional_tensor")
        _shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
        _sys.modules["torchvision.transforms.functional_tensor"] = _shim
except Exception:
    pass

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except Exception as _e:
    REALESRGAN_AVAILABLE = False
    logger.warning(f"Real-ESRGAN not available ({_e}) — using bicubic fallback")

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
            # NOTE: a successful potrace trace is already pixel-pure (binary
            # black/white by construction — that's what tracing means). The
            # old code then shrank the 4x cairosvg raster back down to the
            # original block size with INTER_LANCZOS4, and *still* ran the
            # GAN super-resolution step on top of that afterward. Lanczos has
            # negative interpolation lobes (the same ringing _unsharp_no_halo
            # exists to prevent) and a photographic GAN model has no concept
            # of "this edge must stay a hard binary transition" — either one
            # alone is enough to repaint a soft grey band across every traced
            # edge, and running both back to back all but guarantees it.
            # Fix: keep the clean vector raster at its own higher resolution
            # (that resolution increase *is* the "super-res" for this block)
            # and skip the GAN SR pass entirely for anything that vectorized
            # successfully this call.
            # Guard against running diagram-only steps (vectorization, and
            # below, GAN SR at full outscale) on whole-page-sized blocks. On
            # a scanned PDF the "image block" can BE the entire page (e.g.
            # 3300x4400 at 400 DPI) -- tracing that as if it were a small
            # logo and then rasterizing the trace at 4x produces a
            # multi-hundred-megapixel intermediate (the DecompressionBomb
            # warning) for no quality benefit, and burns most of the
            # pipeline's runtime doing it.
            max_vectorize_dim = int(getattr(self.cfg, "max_vectorize_dim_px", 2500))
            block_too_large_to_vectorize = max(arr.shape[0], arr.shape[1]) > max_vectorize_dim

            vectorized_this_pass = False
            if (getattr(self.cfg, "enable_vectorization", False) and (is_diagram or is_photo is False)
                    and not block_too_large_to_vectorize):
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
                                # Keep the vector raster at its native (already
                                # upscaled) resolution instead of shrinking it
                                # back down with a ringing-prone filter. This
                                # *is* the high-res output for this block.
                                arr = arr2.astype(np.uint8)
                                is_diagram = True
                                vectorized_this_pass = True
                                _dbg_save(arr, "after_vectorize", block)
                            else:
                                logger.debug("Vectorization produced near-empty raster; skipping replacement")
                        else:
                            logger.debug("Vectorization raster decode failed or invalid; skipping replacement")
                    except Exception as e:
                        logger.debug(f"Vectorization rasterization error (skipping): {e}")

            # 5. Super-resolution
            # Skipped when this block was just cleanly vectorized above —
            # running a photographic GAN upscaler on an already pixel-pure
            # binary trace is what was reintroducing the grey boundary band.
            if getattr(self.cfg, "enable_super_resolution", False) and not vectorized_this_pass:
                arr = self._selective_super_resolve(arr)
                _dbg_save(arr, "after_sr", block)
            elif vectorized_this_pass:
                logger.debug("Skipping GAN super-resolution: block already vectorized to a clean high-res raster")

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
            img = self._selective_super_resolve(img)
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

        # Configurable thresholds.
        # NOTE: these previously defaulted to 6MP / 3500px, which is *below*
        # what a single 4x Real-ESRGAN pass produces from a normal 900-1200px
        # extracted diagram (900x1200 * 4 = 3600x4800 = ~17MP). Every SR'd
        # diagram was silently being shrunk back down right before writing,
        # throwing away most of the resolution the SR pass just added. Raised
        # to a ceiling that comfortably fits 4x SR output at archival (600
        # DPI) extraction sizes; set explicitly in config if you need a hard
        # cap for memory reasons.
        max_pixels = int(self._get_cfg_val("max_image_pixels", getattr(self.cfg, "max_image_pixels", 60_000_000)))
        max_width = int(self._get_cfg_val("max_image_width", getattr(self.cfg, "max_image_width", 10000)))
        max_height = int(self._get_cfg_val("max_image_height", getattr(self.cfg, "max_image_height", 10000)))
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
        # NOTE: a raw unique-RGB-triple count is not a safe photo signal in
        # this pipeline. Any real scanned/enhanced document page -- even a
        # clean bitonal scan -- picks up thousands of unique colors just
        # from antialiasing, denoising, CLAHE, and 4x SR interpolation; the
        # previous `uniq > 2000` check here was true for essentially every
        # real page, silently forcing is_photo=True and routing ordinary
        # text/diagram content through lossy WebP/JPEG (quality 97) instead
        # of lossless PNG. High-quality JPEG/WebP still chroma-subsamples
        # and DCT-quantizes, which measurably softens the fine,
        # high-frequency strokes a 4x-SR'd page is now full of -- a second,
        # independent contributor to "washed out" output that survives even
        # after the contrast/black-point fixes elsewhere in this file.
        # Require the same paper-white-background test relied on everywhere
        # else here before ever trusting a color-count signal: a real photo
        # of a scene essentially never has this profile, so this only
        # fires for genuinely photographic content.
        try:
            if not is_diagram and not is_photo and not self._has_light_background(img):
                uniq = len(np.unique(img.reshape(-1, 3), axis=0))
                if uniq > 20000:
                    is_photo = True
        except Exception:
            pass

        # Encode
        # This condition used to be `if is_diagram:` alone -- meaning
        # is_photo, despite being computed in detail just above, was never
        # actually consulted for routing. Any block NOT classified as a
        # diagram (which for plain scanned text -- low edge_density, no
        # dense line-art -- is most of them; see classify_image) fell
        # straight through to the WebP/JPEG branch below regardless of
        # is_photo's value, even when is_photo was correctly False. That
        # meant ordinary text/document pages were being lossy-compressed
        # (JPEG quality 97 / WebP quality 97) by default -- softening
        # exactly the fine, high-frequency strokes a 4x-SR pass produces,
        # which is a second, independent way the final output ends up
        # looking soft/washed out even after every upstream contrast fix.
        # Flip the default: treat "not positively identified as a photo"
        # as the safe case and use lossless PNG, matching this project's
        # own stated intent (README: diagrams/text are always full-color,
        # non-quantized PNG) -- only genuinely photographic content should
        # ever take the lossy path.
        if is_diagram or not is_photo:
            # Diagrams/logos: always lossless PNG, full color depth, no palette
            # quantization. pngquant used to run here "for extra compression",
            # but it reduces to a <=256-color dithered palette -- which
            # re-scatters grey/off-white noise pixels across exactly the
            # backgrounds _whiten_diagram_background() just forced to pure
            # white, and visibly bands any smooth SR gradient. File size is
            # not worth trading away cleanliness for on diagram content, so
            # quantization is disabled here unconditionally.
            if PIL_AVAILABLE:
                pil = Image.fromarray(img)
                buf = io.BytesIO()
                pil.save(buf, format="PNG", compress_level=png_compress)
                return buf.getvalue(), "png"
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

    # -------------------------
    # Halo-safe unsharp masking
    # -------------------------

    def _auto_contrast_stretch(self, img: np.ndarray, apply_gamma: bool = True) -> np.ndarray:
        """
        Global black/white-point stretch on the L channel ("auto levels").

        This is the fix for "washed out" / faded-ink output. Every other
        contrast-affecting step in this pipeline only ever pushes pixels
        LIGHTER: _whiten_diagram_background below snaps background-
        classified pixels to pure white; CLAHE redistributes contrast
        *within* small local tiles but has no page-wide notion of "true
        black"; sharpening amplifies existing edges but can't create
        contrast that wasn't there. Nothing anywhere moves a page's actual
        ink darker. On a faded/typewriter-era scan (or any source where
        "black" text actually sits at L~140-180, not near 0), the result
        sails through the whole pipeline still looking grey -- background
        goes pure white (whitening working correctly), ink stays exactly
        as faint as it started, and the net visual effect is "washed out",
        even though every individual step behaved correctly in isolation.

        Fix: one global linear remap of the page's actual dark/light
        extremes onto the full 0-255 range, run early (before CLAHE/
        sharpen/whiten) so those later steps have real contrast to work
        with instead of compensating for a compressed range. Robust
        percentiles (not literal min/max) so a single scanner-noise pixel
        or hairline artifact can't skew the black/white point.
        """
        if not CV2_AVAILABLE:
            return img
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0].astype(np.float32)
            lo_pct = float(getattr(self.cfg, "auto_contrast_black_percentile", 1.0))
            hi_pct = float(getattr(self.cfg, "auto_contrast_white_percentile", 99.0))
            lo = float(np.percentile(l, lo_pct))
            hi = float(np.percentile(l, hi_pct))
            min_range = float(getattr(self.cfg, "auto_contrast_min_range", 20.0))
            if (hi - lo) < min_range:
                # Near-flat tile (blank margin, solid photo region) -- no
                # real ink contrast to recover; stretching a small noise
                # floor to 0-255 would just amplify grain/JPEG blocking.
                return img
            stretched = (l - lo) * (255.0 / (hi - lo))
            stretched = np.clip(stretched, 0, 255)

            # Linear stretch alone only pins the extreme (1st-percentile)
            # darkest pixels to true black; it preserves every other
            # pixel's *relative* position in the range, so partial-
            # coverage/anti-aliased ink (thin strokes, worn typewriter-era
            # scans) that isn't part of that extreme 1% stays exactly as
            # grey, proportionally, as it started -- this is the "black
            # point improved but the page still reads as washed out"
            # symptom. gamma>1 on the already-stretched [0,1] range pushes
            # midtones darker while fixing 0 and 1 in place, so it doesn't
            # touch true white background or true black cores, only the
            # grey-in-between that IS the visible ink on a faded scan.
            # gamma is tuned (see config.py) for exactly ONE application --
            # e.g. median L 106 -> stretch-only 60 -> stretch+gamma(1.8) 19.
            # This function gets called a second time later in the pipeline
            # (postprocess) purely to redo the LINEAR black-point stretch,
            # because SR/upscaling softens true-black back up toward ~19.
            # That second call must NOT reapply gamma: recomputing percentiles
            # on an already-gamma-corrected image and raising the result to
            # 1.8 again compounds to an effective exponent of ~3.24, which
            # crushes midtones far past the tuned target and is what was
            # making output read as "too dark" / heavier than the original
            # scan. apply_gamma=False is passed by that second call site.
            gamma = float(getattr(self.cfg, "auto_contrast_gamma", 1.0))
            if apply_gamma and gamma and gamma != 1.0:
                stretched = 255.0 * np.power(stretched / 255.0, gamma)

            lab[:, :, 0] = np.clip(stretched, 0, 255).astype(np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logger.debug(f"Auto-contrast stretch error: {e}")
            return img

    def _has_light_background(self, img: np.ndarray) -> bool:
        """
        Whitening (_whiten_diagram_background below) is safe and desirable
        whenever a block actually HAS a light background to clean up --
        which is true for the vast majority of real content in this
        pipeline: text, line art, stamps, logos on white paper. It was
        gated on is_diagram alone, which requires color_std < 80. That
        threshold is backwards for exactly the content it's meant to
        protect: a real, tightly-cropped, high-contrast black-on-white
        crop (a bold title, a stamp, a paragraph of text) is maximally
        BIMODAL -- pixels cluster near 0 and near 255 -- which is what
        HIGH std looks like, not low. Verified empirically against three
        synthetic tightly-cropped text samples (a bold title, body text, a
        stamp): color_std came back 97-122 in every case, so is_diagram
        was False for all of them and none ever got whitened, regardless
        of how the whiten_c/block_size constants were tuned -- the gate
        was closed before those constants ever mattered.

        Use a direct, purpose-built test instead of repurposing a
        classifier tuned for a different decision (which SR/vectorization
        path to use): look at whether the lighter portion of the image is
        actually near-white. The 75th percentile (not mean/median) is
        robust to a large dark foreground -- a bold, dense title can be
        >50% black ink and still correctly read as "yes, light background"
        as long as the paper it's printed on is white.
        """
        if not CV2_AVAILABLE:
            return False
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]
            p75 = float(np.percentile(l, 75))
            return p75 > 225
        except Exception:
            return False

    def _whiten_diagram_background(self, img: np.ndarray) -> np.ndarray:
        """
        Force background-classified pixels in a diagram/logo to pure white.

        Must run as the LAST pixel-value-changing step for a diagram, after any
        and all sharpening. Sharpening always runs the risk of nudging a
        background pixel a few levels away from pure white/whatever background
        tone it had; if whitening ran only once earlier in the pipeline and a
        sharpen pass runs afterward (as final_unsharp does in postprocess),
        that later pass silently reintroduces a grey tint at every boundary
        with nothing downstream to clean it back up. Calling this again after
        the final sharpen guarantees a clean result regardless of how many
        sharpen passes ran before it.

        The 15px / C=9 tuning below was validated at ~300 DPI native
        resolution. On an upscaled (SR'd) image the anti-aliased transition
        band around a line is proportionally wider in pixels, so the same
        settings under-classify edge pixels as "background" and grey
        survives. Note: growing the block size to compensate does NOT help
        — a larger window pulls the local mean toward the dark line itself,
        which tightens the effective threshold rather than loosening it
        (verified empirically, not just assumed). What actually widens the
        catch net is increasing C. Both are exposed as config knobs
        (enhancement.whiten_block_size, enhancement.whiten_c,
        enhancement.whiten_c_resolution_scale) rather than hardcoded, since
        the right value depends on your actual SR strength/model and is
        best tuned against real pages using debug_whitening=True rather
        than a one-size-fits-all formula.
        """
        if not CV2_AVAILABLE:
            return img
        try:
            h, w = img.shape[:2]
            block_size = int(getattr(self.cfg, "whiten_block_size", 15))
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, block_size)

            base_c = float(getattr(self.cfg, "whiten_c", 9.0))
            c_resolution_scale = float(getattr(self.cfg, "whiten_c_resolution_scale", 1.0))
            ref_dim = 900.0
            scale_factor = max(1.0, min(w, h) / ref_dim)
            # Linear scaling here (c_val = base_c * scale_factor, uncapped)
            # was too aggressive once the whitening gate broadened to fire
            # on entire diagram blocks rather than just small halos: on a
            # real 4x-SR'd full-page schematic, scale_factor reaches ~4,
            # pushing C past 30. That's fine in isolation (a single clean
            # line on white survives even C=36 -- verified), but a dense
            # diagram has many neighborhoods where several close lines/
            # labels pull the local mean down, and a large C there starts
            # classifying legitimate thin/antialiased ink as "background"
            # too -- the reported "too white, lines washed out" look.
            # sqrt-dampen the growth and hard-cap it so resolution scaling
            # still helps with genuinely wide SR halos without bleeding
            # into real content on dense pages.
            c_val = base_c * (1.0 + (math.sqrt(scale_factor) - 1.0) * c_resolution_scale)
            whiten_c_max = float(getattr(self.cfg, "whiten_c_max", 14.0))
            c_val = min(c_val, whiten_c_max)

            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]
            th = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val)

            # `th` alone is a RELATIVE test (pixel > local_mean - C) and has
            # no idea whether the neighborhood it's in is actually light.
            # A large uniformly-dark region -- the interior of a bold
            # letter, a solid stamp fill, a logo -- has local_mean roughly
            # equal to its own dark pixel value, so it satisfies
            # `pixel > local_mean - C` too (e.g. 0 > 0 - 9) and gets flagged
            # "background" just like real paper. Only the thin anti-aliased
            # boundary ring around such a shape (where the block straddles
            # ink and paper and local_mean gets pulled up) fails the test
            # and stays dark -- which is exactly the "hollowed-out outline"
            # look. Gate the mask on ABSOLUTE local brightness as well, so
            # a pixel only qualifies as background if its neighborhood is
            # actually near paper-white, not merely locally flat.
            local_mean = cv2.boxFilter(l, ddepth=-1, ksize=(block_size, block_size))
            min_brightness = float(getattr(self.cfg, "whiten_min_local_brightness", 140.0))
            mask = (th == 255) & (local_mean > min_brightness)

            l2 = np.where(mask, 255, l)
            lab[:, :, 0] = l2
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logger.debug(f"Diagram background whitening error: {e}")
            return img

    def _unsharp_no_halo(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        """
        Unsharp mask that cannot ring.

        Plain unsharp masking (img*(1+amount) - blurred*amount) overshoots past
        white and undershoots past black on any hard edge -- a diagram line, a
        text stroke -- by construction. cv2.addWeighted's saturate_cast only
        clips the *final* uint8 range (0-255); it does nothing to stop a
        near-white background pixel next to a black line from being driven down
        into a visible grey ring, because 250 undershooting to 210 never hits
        the 0-255 clip boundary at all. Running two unsharp passes back-to-back
        (once in preprocess, again in postprocess -- see enhance_block steps 2
        and 7) compounds this every time, and the second pass runs *after* the
        diagram background-whitening cleanup below, so it re-introduces the
        grey ring right at the end of the pipeline with nothing left to clean
        it up.

        Fix: clamp each sharpened pixel to the local min/max of the
        *un-sharpened* neighborhood. Real contrast enhancement (pushing a
        pixel further toward whichever real nearby value it's already closer
        to) is preserved; overshoot/undershoot past what's actually nearby is
        physically not possible, so no ring can form no matter how large
        `amount` is.
        """
        if not CV2_AVAILABLE:
            return img
        try:
            img_u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
            img_f = img_u8.astype(np.float32)
            blurred_f = cv2.GaussianBlur(img_f, (0, 0), sigma)
            sharpened = img_f + amount * (img_f - blurred_f)

            # Local min/max envelope from the pre-sharpen image, sized to the
            # blur radius so it covers the same neighborhood the halo would
            # otherwise spread across.
            k = max(3, int(round(sigma * 2)) | 1)  # odd kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            local_max = cv2.dilate(img_u8, kernel).astype(np.float32)
            local_min = cv2.erode(img_u8, kernel).astype(np.float32)

            sharpened = np.minimum(sharpened, local_max)
            sharpened = np.maximum(sharpened, local_min)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.debug(f"Halo-safe unsharp error: {e}")
            return img

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
            # Gated the same way whitening is below (is_diagram OR a
            # paper-white background), NOT on is_photo. classify_image's
            # is_photo heuristic mis-fires on ordinary bimodal text/title
            # blocks (see diagram_analyzer.classify_image's docstring) --
            # gating this correction on that flag meant it silently never
            # ran on exactly the content it was built to fix.
            if getattr(self.cfg, "enable_auto_contrast", True) and (is_diagram or self._has_light_background(img)):
                try:
                    img = self._auto_contrast_stretch(img)
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
            if getattr(self.cfg, "remove_paper_texture", True) and not is_photo and not is_diagram:
                # NOTE: this must NOT run on is_diagram content. It measures
                # local darkness via a wide Gaussian blur and darkens any
                # pixel that's brighter than its blurred neighborhood -- on a
                # diagram, every white pixel near a black line qualifies,
                # since the blur averages the nearby black stroke in. A 51px
                # kernel reaches ~25px in every direction, which is exactly
                # the soft grey halo you see around every line/character:
                # this pass, not whitening, was putting it there.
                # _whiten_diagram_background already does the right thing
                # for diagram backgrounds (adaptive per-pixel, not a fixed
                # wide blur), so diagrams skip this entirely. For genuine
                # photo-adjacent/text content where it still runs, the
                # kernel is shrunk to a size that actually matches paper
                # grain (a few px) instead of large enough to blur out
                # diagram-scale linework.
                try:
                    img_f = img.astype(np.float32)
                    blurred_f = cv2.GaussianBlur(img_f, (7, 7), 0)
                    detail = img_f - blurred_f  # signed high-pass, no clipping
                    denoised_f = img_f - 0.15 * detail
                    img = np.clip(denoised_f, 0, 255).astype(np.uint8)
                except Exception:
                    pass
            if getattr(self.cfg, "enable_sharpening", True):
                try:
                    sigma = getattr(self.cfg, "sharpen_sigma", 1.0)
                    amount = getattr(self.cfg, "sharpen_amount", 0.9)
                    img = self._unsharp_no_halo(img, sigma, amount)
                except Exception:
                    pass
            if is_diagram or self._has_light_background(img):
                img = self._whiten_diagram_background(img)
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
                    up = self._unsharp_no_halo(up, 0.6, 0.6)
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

    def _effective_sr_outscale(self, img: np.ndarray) -> float:
        """
        Reduce the configured outscale if the requested scale would blow past
        max_image_pixels -- e.g. a full scanned page at 3300x4400 does not
        need (and shouldn't get) the same 4x treatment as a 400x300 logo
        crop; running full 4x SR on it just to shrink the result back down
        afterward wastes minutes per page for no visible gain.
        """
        outscale = float(getattr(self.cfg, "sr_outscale", 4.0))
        max_pixels = float(self._get_cfg_val("max_image_pixels", getattr(self.cfg, "max_image_pixels", 60_000_000)))
        h, w = img.shape[:2]
        projected = (h * outscale) * (w * outscale)
        if projected > max_pixels and h * w > 0:
            capped = math.sqrt(max_pixels / (h * w))
            outscale = max(1.0, min(outscale, capped))
        return outscale

    def _selective_super_resolve(self, img: np.ndarray) -> np.ndarray:
        """
        selective_sr: only run the expensive GAN model on tiles that
        actually contain ink; blank/margin tiles get a cheap bicubic
        upscale instead.

        NOTE on why this is tile-grid based rather than contour-bounding-box
        based (an earlier version of this method used bounding boxes and it
        did not help at all -- observed: identical tile count/runtime
        before and after enabling it): on a schematic, wires commonly run
        edge-to-edge across the full page. A single such line's bounding
        box spans the full width even though the line itself is only a few
        px thick, so contour-bbox "content regions" balloon to cover
        basically the whole page and the selective path always fell back to
        full-image SR. A fixed grid sized to match the SR model's own tile
        size (self.cfg.sr_tile) doesn't have that failure mode: each grid
        cell is judged only by the ink actually inside it, so a thin line
        passing through causes exactly the cells it touches to be
        GAN-processed and nothing more -- margins and blank interior areas
        (there are real ones on a typical schematic page, e.g. between
        symbol clusters) get skipped regardless of how far a line elsewhere
        on the page happens to stretch.
        """
        outscale = self._effective_sr_outscale(img)

        if not getattr(self.cfg, "selective_sr", False) or not CV2_AVAILABLE:
            return self._super_resolve(img, outscale=outscale)

        h, w = img.shape[:2]
        pad = 16  # crop context so the model doesn't see a hard-cropped edge

        # The crop handed to _super_resolve() below is (grid_tile + 2*pad)
        # wide/tall. The cached RealESRGANer instance has its own internal
        # tile_size fixed to sr_tile at construction time (see
        # _get_sr_model), and re-splits ANY image wider/taller than that
        # threshold via its own tile_process(), regardless of who called
        # it. If grid_tile defaults to the same value as sr_tile, the
        # padded crop (grid_tile + 32) exceeds sr_tile and gets needlessly
        # re-tiled a second time inside the model (observed as repeated
        # "Tile 1/4, Tile 2/4..." — double-tiling, pure overhead, no
        # quality benefit since the crop was already small enough to
        # process as one unit). Keep the padded crop at or under sr_tile so
        # RealESRGANer's own ceil(w / tile_size) resolves to 1 and it skips
        # its internal split.
        sr_tile = int(getattr(self.cfg, "sr_tile", 512))
        configured_tile = int(getattr(self.cfg, "selective_sr_tile_px", 0))
        if configured_tile > 0:
            tile = configured_tile
            if tile + 2 * pad > sr_tile:
                logger.debug(
                    f"selective_sr_tile_px={tile} + padding exceeds sr_tile={sr_tile}; "
                    f"crops will be internally re-tiled by RealESRGANer (double-tiling)."
                )
        else:
            tile = max(64, sr_tile - 2 * pad - 8)  # -8 safety margin against rounding
        if tile <= 0 or h <= tile or w <= tile:
            return self._super_resolve(img, outscale=outscale)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)  # 255 = ink
            density_threshold = float(getattr(self.cfg, "selective_sr_tile_density", 0.004))

            out_w, out_h = max(1, int(round(w * outscale))), max(1, int(round(h * outscale)))
            out = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            processed, total = 0, 0

            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    total += 1
                    y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                    cell = ink[y0:y1, x0:x1]
                    density = float(np.count_nonzero(cell)) / max(1, cell.size)
                    if density < density_threshold:
                        continue  # bicubic base already covers this cell

                    ey0, ex0 = max(0, y0 - pad), max(0, x0 - pad)
                    ey1, ex1 = min(h, y1 + pad), min(w, x1 + pad)
                    crop = img[ey0:ey1, ex0:ex1]
                    try:
                        sr_crop = self._super_resolve(crop, outscale=outscale)
                    except Exception as e:
                        logger.debug(f"Selective SR tile failed, leaving bicubic base: {e}")
                        continue
                    processed += 1

                    # Trim the padded SR'd crop back down to just this cell's
                    # region (at output scale) before pasting.
                    iy0 = int(round((y0 - ey0) * outscale))
                    ix0 = int(round((x0 - ex0) * outscale))
                    iy1 = iy0 + int(round((y1 - y0) * outscale))
                    ix1 = ix0 + int(round((x1 - x0) * outscale))
                    sub = sr_crop[iy0:iy1, ix0:ix1]

                    ty0, tx0 = int(round(y0 * outscale)), int(round(x0 * outscale))
                    ty1 = min(out.shape[0], ty0 + sub.shape[0])
                    tx1 = min(out.shape[1], tx0 + sub.shape[1])
                    out[ty0:ty1, tx0:tx1] = sub[: ty1 - ty0, : tx1 - tx0]

            logger.debug(f"Selective SR (tiled): {processed}/{total} tiles GAN-processed")
            return out
        except Exception as e:
            logger.debug(f"Selective SR error, falling back to full-image SR: {e}")
            return self._super_resolve(img, outscale=outscale)

    def _super_resolve(self, img: np.ndarray, outscale: Optional[float] = None) -> np.ndarray:
        if outscale is None:
            outscale = self._effective_sr_outscale(img)
        if not REALESRGAN_AVAILABLE:
            return self._bicubic_upscale(img, outscale=outscale)
        try:
            upsampler = self._get_sr_model()
            out, _ = upsampler.enhance(img, outscale=outscale)
            if isinstance(out, np.ndarray):
                out = np.clip(out, 0, 255).astype(np.uint8)
            return out
        except RuntimeError as e:
            logger.warning(f"SR OOM: {e} — falling back to bicubic")
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                    _torch.mps.empty_cache()
            except Exception:
                pass
            return self._bicubic_upscale(img, outscale=outscale)
        except Exception as e:
            logger.debug(f"SR error: {e}")
            return self._bicubic_upscale(img, outscale=outscale)

    def _bicubic_upscale(self, img: np.ndarray, outscale: Optional[float] = None) -> np.ndarray:
        if not CV2_AVAILABLE:
            return img
        h, w = img.shape[:2]
        scale = float(outscale) if outscale is not None else self._effective_sr_outscale(img)
        if scale <= 1:
            return img
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _get_sr_model(self) -> "RealESRGANer":
        if self._sr_model is not None:
            return self._sr_model
        model_name = getattr(self.cfg, "sr_model", "RealESRGAN_x4plus")
        model_path = MODELS_DIR / f"{model_name}.pth"

        # A .pth file is a zip archive (PyTorch's default checkpoint format).
        # A partial/interrupted download (dropped connection, disk full,
        # Ctrl+C mid-transfer, etc.) still leaves a file on disk -- just not
        # a valid one. The old check here was only "does the file exist",
        # which meant a single interrupted download would poison every
        # future run forever: the file exists, so it's never re-downloaded,
        # and torch fails with a cryptic "failed finding central directory"
        # error on every single call, silently falling back to bicubic every
        # time (this is exactly what your last log showed). Validate before
        # trusting a cached file, and self-heal by re-downloading if not.
        if model_path.exists() and not zipfile.is_zipfile(str(model_path)):
            logger.warning(f"Cached SR model {model_path} is corrupt/incomplete — deleting and re-downloading")
            model_path.unlink(missing_ok=True)

        if not model_path.exists():
            url = REALESRGAN_MODEL_URLS.get(model_name)
            if not url:
                raise FileNotFoundError(f"SR model not found: {model_name}")
            logger.info(f"Downloading SR model {model_name} (first use)...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
            except Exception as e:
                model_path.unlink(missing_ok=True)
                raise IOError(f"Download of SR model {model_name} failed ({e}); nothing cached, try again") from e
            if not zipfile.is_zipfile(str(model_path)):
                model_path.unlink(missing_ok=True)
                raise IOError(
                    f"Downloaded SR model {model_name} is corrupt/incomplete "
                    f"(interrupted connection?) — deleted, please retry"
                )
        # NOTE: each Real-ESRGAN release uses a DIFFERENT network
        # architecture, not just different weights. The previous version of
        # this function only branched on "anime" / "x2" and used RRDBNet for
        # everything else -- which is correct for RealESRGAN_x4plus, but
        # realesr-general-x4v3 is actually a SRVGGNetCompact model.
        # Loading its weights into an RRDBNet would either fail outright or
        # silently produce garbage output, so it gets its own branch here.
        if "anime" in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == "realesr-general-x4v3":
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
            netscale = 4
        elif "x2" in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        gpu_id = getattr(self.cfg, "sr_gpu_id", None)
        # NOTE: torch.cuda.is_available() is always False on a Mac (no
        # NVIDIA hardware) -- this device selection previously only checked
        # CUDA, so every Apple Silicon machine silently fell back to plain
        # CPU inference regardless of sr_gpu_id, with no indication that a
        # real GPU (Metal/MPS) was sitting right there unused. This is very
        # likely the dominant remaining cause of multi-minute-per-tile
        # runtimes even after the double-tiling fix. fp16 (sr_half) is left
        # CUDA-only below -- MPS fp16 support is inconsistent across ops in
        # current torch and can silently produce NaNs for some models, so
        # MPS always runs fp32.
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if REALESRGAN_AVAILABLE and torch.cuda.is_available() and gpu_id is not None:
            device_str = f"cuda:{gpu_id}"
        elif REALESRGAN_AVAILABLE and has_mps and gpu_id is not None:
            device_str = "mps"
        else:
            device_str = "cpu"
        self._sr_model = RealESRGANer(
            scale=netscale,
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
                    img = self._unsharp_no_halo(img, sigma, amount)
            except Exception:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # Re-stretch the black point one more time: the GAN SR pass (or
            # its bicubic fallback) interpolates thin strokes, which softens
            # exactly the black point _auto_contrast_stretch fixed back in
            # preprocess -- measured on this codebase's own test page, a
            # clean 4x bicubic upscale alone drags the true-black point from
            # 0 back up to ~19, and real Real-ESRGAN output can do the same
            # via its own smoothing on out-of-domain (non-photo) input. This
            # is the same "must run again after SR/sharpen, not just once
            # early" situation _whiten_diagram_background already documents
            # for the light end; nothing was doing the equivalent for the
            # dark end, which is why ink kept coming back grey even after
            # the preprocess-stage fix.
            if CV2_AVAILABLE and getattr(self.cfg, "enable_auto_contrast", True) and (is_diagram or self._has_light_background(img)):
                try:
                    # apply_gamma=False: gamma was already applied once in
                    # preprocess. This call's only job is to re-pin the
                    # black/white point after SR softened it (see comment
                    # above) -- reapplying gamma here as well double-darkens
                    # every midtone pixel (effective exponent ~gamma^2).
                    img = self._auto_contrast_stretch(img, apply_gamma=False)
                except Exception:
                    pass

            # Re-whiten diagram backgrounds: this MUST be the last pixel-value
            # step. The unsharp pass just above can nudge background pixels
            # away from pure white at every boundary; re-running the whitening
            # here (rather than relying solely on the earlier preprocess pass)
            # guarantees no grey survives at edges in the final output.
            if CV2_AVAILABLE and (is_diagram or self._has_light_background(img)):
                try:
                    if getattr(self.cfg, "debug_whitening", False):
                        pre_whiten = img.copy()
                        img = self._whiten_diagram_background(img)
                        diff = cv2.absdiff(pre_whiten, img)
                        # Amplify so a subtle few-level grey correction is
                        # actually visible in the saved preview, not just a
                        # near-black image.
                        diff_vis = np.clip(diff.astype(np.int32) * 8, 0, 255).astype(np.uint8)
                        _dbg_save(diff_vis, "whitening_diff_x8", None)
                    else:
                        img = self._whiten_diagram_background(img)
                except Exception:
                    pass

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
                elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                    _torch.mps.empty_cache()
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