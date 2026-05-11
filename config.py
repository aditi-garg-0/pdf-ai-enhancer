"""
config.py
=========
Central configuration hub for the PDF AI Enhancer pipeline.
All tunable parameters, model paths, API settings, and feature flags live here.
Override any value via environment variables or a config.yaml file.
"""

from __future__ import annotations

import os
import multiprocessing
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

# ── Project Root ─────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.resolve()
MODELS_DIR    = ROOT_DIR / "models"
CACHE_DIR     = ROOT_DIR / ".cache"
OUTPUT_DIR    = ROOT_DIR / "output"
TEMP_DIR      = ROOT_DIR / ".tmp"

for _d in (MODELS_DIR, CACHE_DIR, OUTPUT_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  Pydantic config models (validated, typed, documented)
# ──────────────────────────────────────────────────────────────

class APIConfig(BaseModel):
    """API keys and endpoints."""
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key:    str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Claude model selection
    claude_model:   str = "claude-opus-4-5"          # Used for vision + text correction
    claude_fast:    str = "claude-sonnet-4-20250514"  # Used for cheaper/faster calls
    openai_model:   str = "gpt-4o"                   # Fallback if Claude unavailable

    # Rate limiting
    max_concurrent_api_calls: int   = 3
    api_retry_attempts:       int   = 5
    api_retry_min_wait:       float = 1.0   # seconds
    api_retry_max_wait:       float = 60.0  # seconds

    # Token budgets
    max_tokens_correction: int  = 4096
    max_tokens_analysis:   int  = 2048
    max_tokens_summary:    int  = 1024


class ExtractionConfig(BaseModel):
    """PDF extraction parameters."""
    # Rendering DPI for rasterized pages
    render_dpi:        int  = 300    # 300 = print quality; 600 = archival
    render_dpi_ocr:    int  = 400    # Higher DPI for better OCR accuracy
    render_colorspace: str  = "RGB"  # RGB | CMYK | GRAY

    # Page content classification thresholds
    text_density_threshold:  float = 0.05   # Min text chars/pixel² to be "text page"
    image_area_threshold:    float = 0.10   # Min image area fraction to extract
    diagram_aspect_ratio_min: float = 0.2   # Exclude very thin slivers

    # Text extraction mode
    text_extract_mode: Literal["fast", "layout", "full"] = "full"
    # fast   = pypdf (fastest, least accurate)
    # layout = pdfplumber with coords
    # full   = PyMuPDF dict blocks (most detail)

    # Embedded font extraction
    extract_fonts: bool = True

    # Table detection
    table_extraction_method: Literal["camelot", "pdfplumber", "paddleocr", "all"] = "all"
    camelot_flavor: Literal["lattice", "stream"] = "lattice"


class ImageEnhancementConfig(BaseModel):
    """Image / diagram enhancement parameters."""

    # ── Preprocessing ────────────────────────────────────────
    enable_denoising:  bool  = True
    denoise_strength:  float = 10.0   # fastNlMeans h param (higher = more denoising)
    enable_clahe:      bool  = True
    clahe_clip_limit:  float = 3.0
    clahe_grid_size:   int   = 8      # tile grid size (NxN)
    enable_sharpening: bool  = True
    sharpen_sigma:     float = 1.5
    sharpen_amount:    float = 1.5    # unsharp mask weight

    # ── Super-Resolution ─────────────────────────────────────
    enable_super_resolution: bool = True
    sr_model: Literal[
        "RealESRGAN_x4plus",           # General images — best overall
        "RealESRGAN_x4plus_anime",     # Line art / diagrams — sharpest edges
        "RealESRGAN_x2plus",           # 2x upscale (faster, less aggressive)
        "realesr-general-x4v3",        # Latest general model
    ] = "RealESRGAN_x4plus_anime"

    sr_outscale:   float = 4.0         # Output upscale factor
    sr_tile:       int   = 512         # Tile size (reduce if OOM)
    sr_tile_pad:   int   = 10
    sr_half:       bool  = True        # FP16 (faster on GPU, slight quality loss)
    sr_gpu_id:     Optional[int] = 0   # None = CPU; 0 = first GPU

    # ── Diagram-specific ─────────────────────────────────────
    enable_vectorization: bool = True   # potrace: raster → SVG
    vectorize_threshold:  int  = 128    # Binarization threshold before tracing
    vectorize_min_area:   int  = 50     # Ignore tiny noise specks

    # ── Face restoration (for portrait/photo regions) ─────────
    enable_face_restoration: bool = True
    face_model: Literal["GFPGANv1.4", "GFPGANv1.3", "RestoreFormer"] = "GFPGANv1.4"

    # ── Output format ─────────────────────────────────────────
    output_format: Literal["PNG", "JPEG", "WEBP"] = "PNG"
    jpeg_quality:  int  = 97
    png_compress:  int  = 1    # 0-9 (lower = faster, larger file)


class OCRConfig(BaseModel):
    """OCR engine configuration."""

    # Engine priority order (tried in order, results fused)
    engine_priority: list[Literal["tesseract", "easyocr", "paddleocr", "surya"]] = [
        "paddleocr", "tesseract", "easyocr"
    ]

    # Tesseract
    tesseract_lang:    str  = "eng"     # e.g. "eng+hin+fra" for multilingual
    tesseract_oem:     int  = 3         # 3 = LSTM (best accuracy)
    tesseract_psm:     int  = 6         # 6 = uniform block; 11 = sparse text
    tesseract_dpi:     int  = 300

    # EasyOCR
    easyocr_langs:     list[str] = ["en"]
    easyocr_gpu:       bool = True
    easyocr_detail:    int  = 1         # 1 = bounding boxes; 0 = text only

    # PaddleOCR
    paddle_lang:       str  = "en"
    paddle_use_gpu:    bool = True
    paddle_use_angle_classifier: bool = True  # Handle rotated text

    # Confidence thresholds
    min_confidence:    float = 0.6      # Discard low-confidence OCR tokens
    ensemble_strategy: Literal["voting", "confidence", "longest"] = "confidence"

    # Post-processing
    enable_ai_correction:    bool = True   # Send OCR text to Claude for cleanup
    enable_spell_correction: bool = True   # SymSpell fast spell check
    enable_grammar_fix:      bool = True   # LanguageTool grammar correction
    enable_unicode_fix:      bool = True   # ftfy encoding artifact cleanup


class DiagramConfig(BaseModel):
    """AI diagram analysis & reconstruction."""
    enable_ai_analysis:      bool = True   # Claude Vision: understand diagram content
    enable_reconstruction:   bool = False  # Re-draw from semantic description (slow)
    enable_vectorization:    bool = True   # potrace raster → SVG
    embed_as_vector:         bool = True   # Embed SVG in output PDF (not raster)
    min_diagram_size_px:     int  = 100    # Min width/height to attempt enhancement
    diagram_types: list[str] = [           # Classes Claude identifies
        "flowchart", "schematic", "graph", "table",
        "formula", "equation", "map", "photo", "logo"
    ]


class PipelineConfig(BaseModel):
    """Master pipeline settings."""

    # Processing scope
    pages: Optional[str]    = None     # None = all; "1-5,8,10-15" = specific
    skip_text_pages: bool   = False    # Skip pages with mostly text
    skip_image_pages: bool  = False    # Skip pages with mostly images

    # Concurrency
    workers: int = min(4, multiprocessing.cpu_count())
    batch_size: int = 5                # Pages per processing batch

    # Output
    output_dpi: int = 300
    output_suffix: str = "_enhanced"
    embed_searchable_text: bool = True  # Invisible text layer for PDF search
    compress_output: bool = True
    optimize_output: bool = True        # Remove duplicate objects, linearize

    # Resume support
    enable_checkpoint: bool = True      # Resume interrupted pipeline from DB
    checkpoint_db: str = str(CACHE_DIR / "pipeline_state.db")

    # Quality metrics
    compute_ssim:  bool = True          # Structural Similarity before/after
    compute_psnr:  bool = True          # Peak Signal-to-Noise Ratio
    save_report:   bool = True          # JSON quality report per page


class Config(BaseModel):
    """Root config — composes all sub-configs."""
    api:         APIConfig         = Field(default_factory=APIConfig)
    extraction:  ExtractionConfig  = Field(default_factory=ExtractionConfig)
    enhancement: ImageEnhancementConfig = Field(default_factory=ImageEnhancementConfig)
    ocr:         OCRConfig         = Field(default_factory=OCRConfig)
    diagram:     DiagramConfig     = Field(default_factory=DiagramConfig)
    pipeline:    PipelineConfig    = Field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved → {path}")

    @field_validator("api", mode="before")
    @classmethod
    def warn_missing_keys(cls, v):
        if isinstance(v, dict):
            if not v.get("anthropic_api_key") and not os.getenv("ANTHROPIC_API_KEY"):
                logger.warning("ANTHROPIC_API_KEY not set — AI features will be limited")
        return v


# ── Singleton loader ──────────────────────────────────────────
_CONFIG_PATH = ROOT_DIR / "config.yaml"

def get_config(path: Optional[Path] = None) -> Config:
    """Load config from YAML if it exists, else return defaults."""
    p = path or _CONFIG_PATH
    if p.exists():
        logger.info(f"Loading config from {p}")
        return Config.from_yaml(p)
    logger.info("No config.yaml found — using defaults")
    return Config()


# ── Real-ESRGAN model download URLs ──────────────────────────
REALESRGAN_MODEL_URLS = {
    "RealESRGAN_x4plus":        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime":  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus":        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesr-general-x4v3":     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}

GFPGAN_MODEL_URLS = {
    "GFPGANv1.4":     "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    "GFPGANv1.3":     "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "RestoreFormer":  "https://github.com/wzhouxiff/RestoreFormer/releases/download/v1.0.0/RestoreFormer.ckpt",
}