"""
image_enhancer.py
=================
Stage 2: AI-powered image and diagram enhancement.

Processes every ImageBlock extracted from the PDF:
  1. Preprocessing  — denoise, CLAHE, unsharp mask, morphology
  2. Super-Resolution — Real-ESRGAN 4x (general or anime/line-art model)
  3. Face Restoration — GFPGAN for portrait regions in scanned docs
  4. Post-processing  — final tone correction, colour grading
  5. Quality scoring  — SSIM + PSNR before/after comparison
"""

from __future__ import annotations

import gc
import io
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from loguru import logger
from tqdm import tqdm

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN not available — using bicubic fallback")

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from config import Config, ImageEnhancementConfig, MODELS_DIR, REALESRGAN_MODEL_URLS, GFPGAN_MODEL_URLS
from extractor import ImageBlock


# ──────────────────────────────────────────────────────────────
#  Image Enhancer
# ──────────────────────────────────────────────────────────────

class ImageEnhancer:
    """
    Full AI image enhancement pipeline for PDF image/diagram blocks.
    """

    def __init__(self, config: Config):
        self.cfg: ImageEnhancementConfig = config.enhancement
        self._sr_model  = None   # Lazy-loaded Real-ESRGAN
        self._face_model = None  # Lazy-loaded GFPGAN

    # ── Public API ────────────────────────────────────────────

    def enhance_block(self, block: ImageBlock) -> ImageBlock:
        """
        Full enhancement pipeline for a single ImageBlock.
        Returns a new ImageBlock with enhanced image_bytes.
        """
        img = block.to_numpy()          # H×W×3 uint8
        original = img.copy()

        logger.debug(
            f"  Enhancing image xref={block.xref} "
            f"({block.width}×{block.height}) p{block.page_num}"
        )

        # 1. Preprocess
        img = self._preprocess(img, is_diagram=block.is_diagram)

        # 2. Super-resolution
        if self.cfg.enable_super_resolution:
            img = self._super_resolve(img)

        # 3. Face restoration (only if not a diagram)
        if self.cfg.enable_face_restoration and not block.is_diagram and block.is_photo:
            img = self._restore_faces(img)

        # 4. Post-process
        img = self._postprocess(img)

        # 5. Quality metrics
        metrics = {}
        if SKIMAGE_AVAILABLE:
            metrics = self._compute_quality(original, img)
            logger.debug(f"    SSIM: {metrics.get('ssim', 0):.4f}  PSNR: {metrics.get('psnr', 0):.2f}dB")

        # Convert back to bytes
        enhanced_bytes = self._to_bytes(img)
        pil = Image.open(io.BytesIO(enhanced_bytes))

        return ImageBlock(
            image_bytes = enhanced_bytes,
            bbox        = block.bbox,
            page_num    = block.page_num,
            xref        = block.xref,
            width       = pil.width,
            height      = pil.height,
            dpi_x       = block.dpi_x * (pil.width  / block.width  if block.width  else 1),
            dpi_y       = block.dpi_y * (pil.height / block.height if block.height else 1),
            colorspace  = "RGB",
            ext         = self.cfg.output_format.lower(),
            sha256      = "",  # Will be recomputed in __post_init__
            is_diagram  = block.is_diagram,
            is_photo    = block.is_photo,
        )

    def enhance_page_render(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance a full-page rendered numpy array.
        Used when the whole page is treated as an image (scanned docs).
        """
        img = self._preprocess(img, is_diagram=False)
        if self.cfg.enable_super_resolution:
            img = self._super_resolve(img)
        img = self._postprocess(img)
        return img

    # ── Stage 1: Preprocessing ────────────────────────────────

    def _preprocess(self, img: np.ndarray, is_diagram: bool = False) -> np.ndarray:
        """
        Full preprocessing chain:
          - Denoise (Non-Local Means — best quality)
          - CLAHE (adaptive contrast, preserves local detail)
          - Unsharp Mask (controlled sharpening)
          - Morphological cleanup (for diagrams: remove noise dots)
        """
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 255)).astype(np.uint8)

        # ── Denoise ──────────────────────────────────────────
        if self.cfg.enable_denoising:
            h = self.cfg.denoise_strength
            if is_diagram:
                # Stronger denoising for clean line art
                h = min(h * 1.5, 20.0)
            img = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h        = h,
                hColor   = h,
                templateWindowSize = 7,
                searchWindowSize   = 21,
            )

        # ── CLAHE (Contrast Limited Adaptive Histogram Equalization) ──
        if self.cfg.enable_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(
                clipLimit  = self.cfg.clahe_clip_limit,
                tileGridSize = (self.cfg.clahe_grid_size, self.cfg.clahe_grid_size),
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # ── Unsharp Mask ─────────────────────────────────────
        if self.cfg.enable_sharpening:
            sigma  = self.cfg.sharpen_sigma
            amount = self.cfg.sharpen_amount
            blurred = cv2.GaussianBlur(img, (0, 0), sigma)
            img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

        # ── Morphological noise removal for diagrams ─────────
        if is_diagram:
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)   # Remove speckles
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Close tiny gaps

        return img

    # ── Stage 2: Super-Resolution ─────────────────────────────

    def _super_resolve(self, img: np.ndarray) -> np.ndarray:
        if not REALESRGAN_AVAILABLE:
            return self._bicubic_upscale(img)

        try:
            upsampler = self._get_sr_model()
            output, _ = upsampler.enhance(img, outscale=self.cfg.sr_outscale)
            return output
        except RuntimeError as e:
            # OOM — retry with CPU
            logger.warning(f"  SR GPU OOM: {e} — retrying on CPU")
            if REALESRGAN_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._bicubic_upscale(img)
        except Exception as e:
            logger.error(f"  SR failed: {e} — bicubic fallback")
            return self._bicubic_upscale(img)

    def _bicubic_upscale(self, img: np.ndarray) -> np.ndarray:
        """High-quality bicubic upscale as fallback."""
        h, w = img.shape[:2]
        scale = int(self.cfg.sr_outscale)
        return cv2.resize(
            img, (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    def _get_sr_model(self) -> "RealESRGANer":
        """Lazy-load and cache Real-ESRGAN model."""
        if self._sr_model is not None:
            return self._sr_model

        model_name = self.cfg.sr_model
        model_path = MODELS_DIR / f"{model_name}.pth"

        # Download if missing
        if not model_path.exists():
            url = REALESRGAN_MODEL_URLS.get(model_name)
            if url:
                logger.info(f"  Downloading {model_name} …")
                urllib.request.urlretrieve(url, str(model_path))
            else:
                raise FileNotFoundError(f"Model not found: {model_name}")

        # Select architecture
        if "anime" in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
        elif "x2" in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)

        gpu_id = self.cfg.sr_gpu_id
        device_str = f"cuda:{gpu_id}" if (
            REALESRGAN_AVAILABLE and torch.cuda.is_available() and gpu_id is not None
        ) else "cpu"

        self._sr_model = RealESRGANer(
            scale       = 4 if "x2" not in model_name else 2,
            model_path  = str(model_path),
            model       = model,
            tile        = self.cfg.sr_tile,
            tile_pad    = self.cfg.sr_tile_pad,
            pre_pad     = 0,
            half        = self.cfg.sr_half and device_str.startswith("cuda"),
            device      = device_str,
        )
        logger.info(f"  Real-ESRGAN loaded: {model_name} on {device_str}")
        return self._sr_model

    # ── Stage 3: Face Restoration ─────────────────────────────

    def _restore_faces(self, img: np.ndarray) -> np.ndarray:
        if not GFPGAN_AVAILABLE:
            return img
        try:
            restorer = self._get_face_model()
            # GFPGAN expects BGR
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, _, output = restorer.enhance(
                bgr,
                has_aligned   = False,
                only_center_face = False,
                paste_back    = True,
            )
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.debug(f"  Face restoration skipped: {e}")
            return img

    def _get_face_model(self) -> "GFPGANer":
        if self._face_model is not None:
            return self._face_model

        model_name = self.cfg.face_model
        model_path = MODELS_DIR / f"{model_name}.pth"

        if not model_path.exists():
            url = GFPGAN_MODEL_URLS.get(model_name)
            if url:
                logger.info(f"  Downloading {model_name} …")
                urllib.request.urlretrieve(url, str(model_path))

        self._face_model = GFPGANer(
            model_path    = str(model_path),
            upscale       = 1,
            arch          = "clean",
            channel_multiplier = 2,
        )
        return self._face_model

    # ── Stage 4: Post-processing ──────────────────────────────

    def _postprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Final tone + saturation correction.
        Subtle improvements — never over-process.
        """
        pil = Image.fromarray(img.astype(np.uint8))

        # Subtle saturation boost (1.0 = no change, 1.1 = 10% boost)
        pil = ImageEnhance.Color(pil).enhance(1.05)

        # Very slight brightness normalisation
        pil = ImageEnhance.Brightness(pil).enhance(1.02)

        return np.array(pil)

    # ── Quality Metrics ───────────────────────────────────────

    def _compute_quality(
        self, original: np.ndarray, enhanced: np.ndarray
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            # Resize original to same size as enhanced for comparison
            h, w = enhanced.shape[:2]
            orig_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)

            orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2GRAY)
            enh_gray  = cv2.cvtColor(enhanced,     cv2.COLOR_RGB2GRAY)

            metrics["ssim"] = float(ssim_fn(orig_gray, enh_gray, data_range=255))
            metrics["psnr"] = float(psnr_fn(orig_gray, enh_gray, data_range=255))
            metrics["mean_diff"] = float(np.mean(np.abs(
                orig_resized.astype(np.float32) - enhanced.astype(np.float32)
            )))
        except Exception as e:
            logger.debug(f"  Quality metric error: {e}")
        return metrics

    # ── Utilities ─────────────────────────────────────────────

    def _to_bytes(self, img: np.ndarray) -> bytes:
        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        fmt = self.cfg.output_format.upper()
        if fmt == "JPEG":
            pil.save(buf, format="JPEG", quality=self.cfg.jpeg_quality, optimize=True)
        elif fmt == "WEBP":
            pil.save(buf, format="WEBP", quality=self.cfg.jpeg_quality, method=6)
        else:
            pil.save(buf, format="PNG", compress_level=self.cfg.png_compress)
        return buf.getvalue()

    def cleanup(self):
        """Free GPU memory when done."""
        self._sr_model   = None
        self._face_model = None
        gc.collect()
        if REALESRGAN_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass