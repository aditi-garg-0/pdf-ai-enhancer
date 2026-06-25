# image_enhancer.py
# High-fidelity image enhancer focused on preserving color, improving logos/diagrams/small text,
# removing paper marks conservatively, and producing higher-resolution outputs.
#
# Key improvements over prior versions:
#  - Never globally convert images to grayscale; operate in LAB/HSV for masks but preserve AB channels.
#  - Multi-pass SR support (chain x4 models to reach higher effective upscales when requested).
#  - Aggressive diagram vectorization path: try potrace -> rasterize SVG at high DPI (if available).
#  - OCR/MSER-assisted small-text upscaling before SR to preserve legibility.
#  - Conservative mark/stain detection with edge protection and color-preserving inpaint blending.
#  - Logo-specific handling: detect simple vector-like logos and vectorize/rasterize at high resolution.
#  - Defensive config access via getattr with sensible defaults.
#
# Drop this file into your project replacing the previous image_enhancer.py.

from __future__ import annotations

import gc
import io
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from loguru import logger

# Optional heavy deps
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

# Optional: cairosvg for SVG->PNG rasterization (for vectorized diagrams)
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

# Optional: potrace binary for vectorization
POTRACE_AVAILABLE = shutil.which("potrace") is not None

from config import Config, ImageEnhancementConfig, MODELS_DIR, REALESRGAN_MODEL_URLS, GFPGAN_MODEL_URLS
from extractor import ImageBlock

Rect = Tuple[int, int, int, int]


class ImageEnhancer:
    """
    High-fidelity image enhancer tuned for logos, diagrams, and small text.
    """

    def __init__(self, config: Config):
        self.cfg: ImageEnhancementConfig = config.enhancement
        self._sr_model = None
        self._face_model = None

    # -------------------------
    # Public API
    # -------------------------

    def enhance_block(self, block: ImageBlock) -> ImageBlock:
        img = block.to_numpy()  # HxWx3 RGB uint8
        original = img.copy()

        logger.debug(f"Enhancing block xref={getattr(block,'xref',None)} p{getattr(block,'page_num',None)} size={img.shape[:2]}")

        is_logo = self._detect_logo(block, img)
        is_diagram = bool(getattr(block, "is_diagram", False)) or is_logo
        is_photo = bool(getattr(block, "is_photo", False))

        # 1. Remove marks conservatively (preserve color)
        img = self._remove_marks_preserve_color(img)

        # 2. Preprocess (color-preserving denoise + local contrast)
        img = self._preprocess_preserve_color(img, is_diagram=is_diagram, is_logo=is_logo, is_photo=is_photo)

        # 3. Enhance small text regions before SR (upsample small text areas to preserve detail)
        img = self._pre_upscale_small_text_regions(img)

        # 4. Vectorize logos/diagrams if configured and beneficial
        if getattr(self.cfg, "enable_vectorization", False) and (is_diagram or is_logo):
            svg_bytes = self._try_vectorize(img)
            if svg_bytes is not None and CAIROSVG_AVAILABLE:
                try:
                    # Rasterize at a higher DPI/scale to get crisp lines
                    scale = int(getattr(self.cfg, "vector_raster_scale", 4))
                    png = cairosvg.svg2png(bytestring=svg_bytes, output_width=img.shape[1] * scale, output_height=img.shape[0] * scale)
                    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
                    if arr is not None:
                        if arr.ndim == 2:
                            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                        elif arr.shape[2] == 4:
                            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                        else:
                            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        # Downscale back to original size with Lanczos to preserve crispness
                        img = cv2.resize(arr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                except Exception:
                    pass

        # 5. Super-resolution: multi-pass if requested (gives higher effective upscale)
        if getattr(self.cfg, "enable_super_resolution", False):
            target_scale = float(getattr(self.cfg, "sr_outscale", 4.0))
            max_single = 4.0  # Real-ESRGAN typical single-model scale
            if REALESRGAN_AVAILABLE and target_scale > max_single:
                img = self._multi_pass_sr(img, target_scale, max_single)
            else:
                img = self._super_resolve(img)

        # 6. Face restoration for photos (optional)
        if getattr(self.cfg, "enable_face_restoration", False) and is_photo and not is_diagram:
            img = self._restore_faces(img)

        # 7. Postprocess: color-preserving tone and final sharpening
        img = self._postprocess_preserve_color(img, is_diagram=is_diagram, is_logo=is_logo)

        # 8. Quality metrics (optional)
        if SKIMAGE_AVAILABLE:
            try:
                metrics = self._compute_quality(original, img)
                logger.debug(f"Quality metrics: SSIM={metrics.get('ssim'):.4f} PSNR={metrics.get('psnr'):.2f}dB")
            except Exception:
                pass

        enhanced_bytes = self._to_bytes(img)
        pil = Image.open(io.BytesIO(enhanced_bytes))

        return ImageBlock(
            image_bytes=enhanced_bytes,
            bbox=block.bbox,
            page_num=block.page_num,
            xref=block.xref,
            width=pil.width,
            height=pil.height,
            dpi_x=block.dpi_x * (pil.width / block.width if block.width else 1),
            dpi_y=block.dpi_y * (pil.height / block.height if block.height else 1),
            colorspace="RGB",
            ext=getattr(self.cfg, "output_format", "PNG").lower(),
            sha256="",
            is_diagram=is_diagram,
            is_photo=is_photo,
        )

    def enhance_page_render(self, img: np.ndarray) -> np.ndarray:
        img = self._remove_marks_preserve_color(img)
        img = self._preprocess_preserve_color(img, is_diagram=False, is_logo=False, is_photo=False)
        if getattr(self.cfg, "enable_super_resolution", False):
            img = self._super_resolve(img)
        img = self._postprocess_preserve_color(img, is_diagram=False, is_logo=False)
        return img

    # -------------------------
    # Mark removal (conservative, color-preserving)
    # -------------------------

    def _remove_marks_preserve_color(self, img: np.ndarray) -> np.ndarray:
        try:
            h, w = img.shape[:2]
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]

            # Local deviation detection
            l_blur = cv2.GaussianBlur(l, (9, 9), 0)
            local_mean = cv2.blur(l_blur, (31, 31))
            diff = cv2.absdiff(l_blur, local_mean)
            diff_thresh = getattr(self.cfg, "mark_diff_thresh", 18)
            _, mask1 = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

            # Low saturation detection
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            sat_thresh = getattr(self.cfg, "mark_sat_thresh", 30)
            _, mask2 = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY_INV)

            mask = cv2.bitwise_and(mask1, mask2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

            # Protect edges (text/lines)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_dil = cv2.dilate(edges, kernel, iterations=2)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges_dil))

            if mask.sum() < (getattr(self.cfg, "mark_min_pixels_frac", 0.0005) * h * w):
                return img

            inpaint_flag = cv2.INPAINT_TELEA if getattr(self.cfg, "inpaint_method", "telea").lower() != "ns" else cv2.INPAINT_NS
            inpainted = cv2.inpaint(img, mask.astype(np.uint8), 3, inpaint_flag)

            # Blend AB channels from original to preserve color
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
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mser = cv2.MSER_create(_delta=5, _min_area=20, _max_area=2000)
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
                # Upscale region modestly (x2) before global SR to preserve strokes
                try:
                    up = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                    # local CLAHE + unsharp
                    lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(max(4, w // 4), max(4, h // 4)))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    up = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    blurred = cv2.GaussianBlur(up, (0, 0), 0.6)
                    up = cv2.addWeighted(up, 1.6, blurred, -0.6, 0)
                    # Paste back scaled down to original bbox but keep detail for SR to amplify
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
            return out
        except RuntimeError as e:
            logger.warning(f"SR OOM: {e} — falling back to bicubic")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return self._bicubic_upscale(img)
        except Exception as e:
            logger.debug(f"SR error: {e}")
            return self._bicubic_upscale(img)

    def _multi_pass_sr(self, img: np.ndarray, target_scale: float, single_scale: float = 4.0) -> np.ndarray:
        """
        Chain multiple SR passes to reach a higher effective upscale while using a single-model scale.
        Example: target_scale=8, single_scale=4 -> two passes of x4 (first pass then second pass).
        """
        try:
            # Compute number of passes (ceil of log)
            passes = max(1, int(np.ceil(np.log(target_scale) / np.log(single_scale))))
            remaining = target_scale ** (1.0 / passes)
            out = img
            for i in range(passes):
                # Use nearest integer scale supported by model (2 or 4)
                scale = int(round(remaining))
                if scale < 2:
                    scale = 2
                # If model supports only 4, call with outscale=scale but Real-ESRGANer will handle
                upsampler = self._get_sr_model()
                out, _ = upsampler.enhance(out, outscale=scale)
            return out
        except Exception as e:
            logger.debug(f"Multi-pass SR failed: {e}")
            return self._super_resolve(img)

    def _detect_regions_mask(self, img: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dil = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            min_area = getattr(self.cfg, "selective_sr_min_area", 100)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            return mask
        except Exception:
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    def _bicubic_upscale(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = int(getattr(self.cfg, "sr_outscale", 4.0))
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
            pil = Image.fromarray(img.astype(np.uint8))
            sat = getattr(self.cfg, "saturation_boost", 1.05)
            bri = getattr(self.cfg, "brightness_boost", 1.02)
            pil = ImageEnhance.Color(pil).enhance(sat)
            pil = ImageEnhance.Brightness(pil).enhance(bri)
            img = np.array(pil)

            if is_logo:
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    l = lab[:, :, 0]
                    l = cv2.equalizeHist(l)
                    lab[:, :, 0] = l
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                except Exception:
                    pass

            if getattr(self.cfg, "final_unsharp", True):
                sigma = getattr(self.cfg, "final_unsharp_sigma", 0.8)
                amount = getattr(self.cfg, "final_unsharp_amount", 0.6)
                try:
                    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
                    img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
                except Exception:
                    pass

            img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        except Exception as e:
            logger.debug(f"Postprocess error: {e}")
            return img

    # -------------------------
    # Quality metrics
    # -------------------------

    def _compute_quality(self, original: np.ndarray, enhanced: np.ndarray) -> dict:
        metrics = {}
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
    # Utilities
    # -------------------------

    def _to_bytes(self, img: np.ndarray) -> bytes:
        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        fmt = getattr(self.cfg, "output_format", "PNG").upper()
        if fmt == "JPEG":
            pil.save(buf, format="JPEG", quality=getattr(self.cfg, "jpeg_quality", 95), optimize=True)
        elif fmt == "WEBP":
            pil.save(buf, format="WEBP", quality=getattr(self.cfg, "jpeg_quality", 95), method=6)
        else:
            pil.save(buf, format="PNG", compress_level=getattr(self.cfg, "png_compress", 1))
        return buf.getvalue()

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

# ---------------------------------------------------------------------
# Additional helper classes and CLI utilities to append to the bottom
# of your existing image_enhancer.py file. These provide modular
# diagram/line processing, vectorization helpers, and a small CLI
# for quick local testing of the diagram pipeline.
# ---------------------------------------------------------------------

from dataclasses import dataclass, field
import json
import argparse
import math
import tempfile

@dataclass
class DiagramEnhancerConfig:
    """Small config object for diagram-specific tuning (defaults safe)."""
    hough_threshold: int = field(default=80)
    hough_min_line_len_frac: float = field(default=0.02)  # fraction of min(h,w)
    hough_max_gap: int = field(default=8)
    diagram_line_width_frac: float = field(default=0.0015)  # fraction of min(h,w)
    diagram_stroke_strength: float = field(default=0.95)
    diagram_thin_lines: bool = field(default=True)
    skeleton_min_area: int = field(default=20)
    vector_raster_scale: int = field(default=4)
    enable_vectorization: bool = field(default=True)
    vector_temp_dir: Optional[Path] = field(default=None)

    def to_dict(self):
        return {
            "hough_threshold": self.hough_threshold,
            "hough_min_line_len_frac": self.hough_min_line_len_frac,
            "hough_max_gap": self.hough_max_gap,
            "diagram_line_width_frac": self.diagram_line_width_frac,
            "diagram_stroke_strength": self.diagram_stroke_strength,
            "diagram_thin_lines": self.diagram_thin_lines,
            "skeleton_min_area": self.skeleton_min_area,
            "vector_raster_scale": self.vector_raster_scale,
            "enable_vectorization": self.enable_vectorization,
        }

class LineRegularizer:
    """
    Utilities to cluster, extend, and regularize Hough line segments.
    This class is intentionally lightweight and pure-numpy/OpenCV so it
    can be used independently of the main enhancer class.
    """

    @staticmethod
    def angle_of_segment(x1, y1, x2, y2) -> float:
        return math.atan2((y2 - y1), (x2 - x1))

    @staticmethod
    def segment_length(x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def merge_collinear_segments(segments, angle_tol=0.05, dist_tol=10):
        """
        Merge segments that are nearly collinear and close to each other.
        segments: list of (x1,y1,x2,y2)
        Returns merged list of segments.
        """
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
                # angle difference
                if abs((ai - aj + math.pi) % math.pi - math.pi/2) < angle_tol or abs(ai - aj) < angle_tol:
                    # check distance between segment midpoints
                    mid_i = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    mid_j = ((x3 + x4) / 2.0, (y3 + y4) / 2.0)
                    if math.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1]) < dist_tol:
                        group.append(segments[j])
                        used[j] = True
            # merge group by taking min/max extents along the line direction
            xs = []
            ys = []
            for (a, b, c, d) in group:
                xs.extend([a, c])
                ys.extend([b, d])
            merged.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return merged

    @staticmethod
    def extend_segment_to_edges(seg, edges_mask):
        """
        Extend a segment along its angle until it hits image edges or non-edge area.
        seg: (x1,y1,x2,y2)
        edges_mask: binary mask of edges (255/0)
        Returns extended (x1,y1,x2,y2)
        """
        x1, y1, x2, y2 = seg
        h, w = edges_mask.shape[:2]
        angle = LineRegularizer.angle_of_segment(x1, y1, x2, y2)
        dx = math.cos(angle)
        dy = math.sin(angle)

        # extend forward
        def walk(x, y, step=1, limit=1000):
            for _ in range(limit):
                nx = int(round(x + dx * step))
                ny = int(round(y + dy * step))
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    break
                if edges_mask[ny, nx] == 0:
                    break
                x, y = nx, ny
            return x, y

        ex1, ey1 = walk(x1, y1, step=-1)
        ex2, ey2 = walk(x2, y2, step=1)
        return (ex1, ey1, ex2, ey2)

class DiagramProcessor:
    """
    High-level diagram processing helpers that use the functions already
    present in the file (skeletonize, vectorize, etc.). This class is
    intended to be instantiated with the same config used by ImageEnhancer.
    """

    def __init__(self, cfg: Optional[DiagramEnhancerConfig] = None):
        self.cfg = cfg or DiagramEnhancerConfig()
        # temp dir for vectorization artifacts
        self.temp_dir = Path(self.cfg.vector_temp_dir) if self.cfg.vector_temp_dir else Path(tempfile.gettempdir())

    def regularize_and_redraw(self, img: np.ndarray) -> np.ndarray:
        """
        Detect Hough segments, cluster/merge them, optionally extend them,
        and redraw with anti-aliased strokes on a transparent overlay.
        Returns an RGB image with redrawn lines composited over the original.
        """
        try:
            h, w = img.shape[:2]
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]
            # stronger edge detection for diagrams
            edges = cv2.Canny(l, 40, 140)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Hough lines
            min_len = max(10, int(min(h, w) * self.cfg.hough_min_line_len_frac))
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=self.cfg.hough_threshold,
                                    minLineLength=min_len, maxLineGap=self.cfg.hough_max_gap)
            segments = []
            if lines is not None:
                for lseg in lines:
                    x1, y1, x2, y2 = lseg[0]
                    segments.append((x1, y1, x2, y2))

            # Merge collinear segments
            merged = LineRegularizer.merge_collinear_segments(segments, angle_tol=0.06, dist_tol=12)

            # Create overlay and draw merged segments
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            base_width = max(1, int(min(h, w) * self.cfg.diagram_line_width_frac))
            for seg in merged:
                # optionally extend to edges using the edges mask
                try:
                    seg_ext = LineRegularizer.extend_segment_to_edges(seg, edges)
                except Exception:
                    seg_ext = seg
                x1, y1, x2, y2 = map(int, seg_ext)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 0, 255), thickness=base_width, lineType=cv2.LINE_AA)

            # If no lines were found, fallback to skeleton of edges
            if overlay[:, :, 3].sum() == 0:
                sk = self._skeletonize(edges)
                overlay[sk > 0] = (0, 0, 0, 255)

            # Composite overlay onto original while preserving color
            alpha = overlay[:, :, 3].astype(np.float32) / 255.0
            alpha3 = np.repeat(alpha[..., None], 3, axis=2)
            img_f = img.astype(np.float32) / 255.0
            stroke_rgb = np.zeros_like(img_f)  # black strokes
            comp = (1 - alpha3) * img_f + alpha3 * stroke_rgb
            comp = np.clip((comp * 255.0).astype(np.uint8), 0, 255)
            # Optionally thin lines
            if self.cfg.diagram_thin_lines:
                gray_comp = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
                _, bw = cv2.threshold(gray_comp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                sk = self._skeletonize(bw)
                colorized = img.copy()
                colorized[sk > 0] = (0, 0, 0)
                alpha_sk = (sk > 0).astype(np.uint8) * 255
                alpha_sk3 = np.repeat((alpha_sk[..., None] / 255.0), 3, axis=2)
                final = (1 - alpha_sk3) * comp + alpha_sk3 * colorized
                return np.clip(final, 0, 255).astype(np.uint8)
            return comp
        except Exception:
            return img

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Reuse the skeletonization approach from the main file if present,
        otherwise provide a local implementation.
        """
        try:
            # If the main file defines a function _skeletonize at module level, use it
            if "ImageEnhancer" in globals() and hasattr(ImageEnhancer, "_skeletonize"):
                # instantiate a dummy enhancer to call method (not ideal but safe)
                dummy_cfg = type("C", (), {"enhancement": None})
                # call static-like method via class if available
                return ImageEnhancer._skeletonize(self=None, binary=binary)  # type: ignore
        except Exception:
            pass
        # fallback local thinning
        try:
            if binary.dtype != np.uint8:
                binary = binary.astype(np.uint8)
            _, bw = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            skel = np.zeros_like(bw)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            eroded = bw.copy()
            while True:
                eroded_next = cv2.erode(eroded, element)
                opened = cv2.morphologyEx(eroded_next, cv2.MORPH_OPEN, element)
                temp = cv2.subtract(eroded_next, opened)
                skel = cv2.bitwise_or(skel, temp)
                eroded = eroded_next
                if cv2.countNonZero(eroded) == 0:
                    break
            return skel
        except Exception:
            return (binary > 0).astype(np.uint8) * 255

# -------------------------
# Small CLI for local testing
# -------------------------

def _cli_test_diagram_pipeline(input_path: str, output_path: str, cfg_json: Optional[str] = None):
    """
    Quick CLI helper to run the diagram pipeline on a single image file.
    Usage:
        python image_enhancer.py --test-diagram in.png out.png
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")
    # convert BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load config if provided
    cfg = DiagramEnhancerConfig()
    if cfg_json:
        try:
            data = json.loads(cfg_json)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception:
            pass

    proc = DiagramProcessor(cfg)
    out = proc.regularize_and_redraw(img)

    # convert RGB->BGR for imwrite
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_bgr)
    print(f"Wrote diagram-enhanced image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for diagram enhancement")
    parser.add_argument("--test-diagram", nargs=2, metavar=("IN", "OUT"), help="Run diagram pipeline on IN and save to OUT")
    parser.add_argument("--cfg", type=str, help="Optional JSON string to override DiagramEnhancerConfig")
    args = parser.parse_args()
    if args.test_diagram:
        _cli_test_diagram_pipeline(args.test_diagram[0], args.test_diagram[1], cfg_json=args.cfg)
