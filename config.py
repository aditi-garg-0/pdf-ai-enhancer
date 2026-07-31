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
    gemini_api_key:    str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Claude model selection
    claude_model:   str = "claude-opus-4-5"          # Used for vision + text correction
    claude_fast:    str = "claude-sonnet-4-20250514"  # Used for cheaper/faster calls
    openai_model:   str = "gpt-4o"                   # Fallback if Claude unavailable

    # Gemini model selection (free-tier fallback when no Anthropic key is set —
    # both are multimodal, so they can do diagram Vision analysis as well as
    # text correction. Flash-Lite has the most generous free-tier quota.)
    gemini_model:      str = "gemini-2.5-flash"       # Vision + text, free tier
    gemini_model_fast: str = "gemini-2.5-flash-lite"  # Text-only correction, free tier

    # Provider priority when multiple keys are set: "anthropic" tries Claude
    # first and falls back to Gemini if the Claude call fails or no Claude
    # key is set; "gemini" does the reverse.
    ai_provider_priority: Literal["anthropic", "gemini"] = "anthropic"

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
    ] = "RealESRGAN_x4plus"
    # NOTE: this used to default to RealESRGAN_x4plus_anime on the theory
    # that a "line art" model would suit diagrams best. In practice, that
    # model is trained specifically on hand-drawn ANIME illustration —
    # which conventionally includes flat cel-shading and drop-shadow-style
    # highlights as part of the art style itself. Run on a scanned
    # technical schematic (out-of-domain input, but still "line art" in
    # the literal sense the model matches on), it hallucinates that same
    # stylized shading: a soft, consistently one-directional shadow
    # duplicated next to every line, digit, and even non-diagram content
    # like a scanned rubber-stamp mark. A directionally-consistent offset
    # ghost across the whole page (rather than symmetric ringing) is the
    # signature of this, not of any blur/sharpen/CLAHE step, which are all
    # radially symmetric and can't produce a one-sided duplicate.
    # RealESRGAN_x4plus is trained on general/natural images with no such
    # stylization bias and is the safer default for scanned documents.

    sr_outscale:   float = 4.0         # Output upscale factor
    sr_tile:       int   = 512         # Tile size (reduce if OOM)
    sr_tile_pad:   int   = 10
    sr_half:       bool  = True        # FP16 (faster on GPU, slight quality loss)
    sr_gpu_id:     Optional[int] = 0   # None = CPU; 0 = first GPU

    # Selective SR: only run the GAN model on grid tiles that actually
    # contain ink; blank/margin tiles get a cheap bicubic upscale instead.
    # Tile-grid based (matches sr_tile by default) rather than
    # contour-bounding-box based -- bounding boxes balloon to cover most of
    # a page when lines run edge-to-edge, which made an earlier version of
    # this feature a no-op on schematics. See _selective_super_resolve's
    # docstring for the full explanation.
    selective_sr: bool = True
    selective_sr_tile_px: int = 0        # 0 = use sr_tile's value
    selective_sr_tile_density: float = 0.004   # min fraction of ink pixels
                                                 # in a tile to warrant GAN SR

    # ── Diagram-specific ─────────────────────────────────────
    enable_vectorization: bool = True   # potrace: raster → SVG
    vectorize_threshold:  int  = 128    # Binarization threshold before tracing
    vectorize_min_area:   int  = 50     # Ignore tiny noise specks
    vector_raster_scale:  int  = 4      # Rasterize traced SVG at this multiple
    max_vectorize_dim_px: int  = 2500   # Skip vectorization above this size --
                                         # it's for logo/diagram crops, not
                                         # whole scanned pages

    # ── Face restoration (for portrait/photo regions) ─────────
    enable_face_restoration: bool = True
    face_model: Literal["GFPGANv1.4", "GFPGANv1.3", "RestoreFormer"] = "GFPGANv1.4"

    # ── Output format ─────────────────────────────────────────
    output_format: Literal["PNG", "JPEG", "WEBP"] = "PNG"
    jpeg_quality:  int  = 97
    png_compress:  int  = 1    # 0-9 (lower = faster, larger file; both lossless)
    prefer_webp:   bool = False  # WebP for photo blocks instead of JPEG
    webp_quality:  int  = 97

    # ── Output resolution ceiling ─────────────────────────────
    # These previously existed only as hardcoded fallbacks inside
    # image_enhancer.py (6MP / 3500px) and were NOT real fields here, so any
    # value set for them in config.yaml was silently ignored by Pydantic and
    # the hardcoded low ceiling always won -- shrinking 4x-super-resolved
    # diagrams back down right before they were written out. Now real,
    # tunable fields, defaulted high enough to comfortably hold 4x SR output.
    max_image_pixels: int = 60_000_000
    max_image_width:  int = 10000
    max_image_height: int = 10000

    # ── Extra polish (previously present in config.yaml but not defined as
    #    fields here, so they had zero effect regardless of the value set) ──
    remove_paper_texture: bool = True
    saturation_boost: float = 1.05
    brightness_boost: float = 1.02
    final_unsharp: bool = True
    final_unsharp_sigma: float = 0.8
    final_unsharp_amount: float = 0.6

    # ── Global auto-contrast / black-point stretch (fixes washed-out,
    #    faded-ink scans) ─────────────────────────────────────
    # Everything else in this file either lightens backgrounds
    # (whiten_diagram_background below) or sharpens edges that are
    # already there -- nothing ever moves a dark pixel darker. A
    # faded/typewriter-era scan (or any scan where "black" ink actually
    # sits at L~140-180, not near 0) sails straight through denoise ->
    # CLAHE -> sharpen -> whiten still looking grey, because CLAHE
    # redistributes contrast *within* small tiles and has no page-wide
    # concept of "the true black point here is 150, not 0". This applies
    # a single global linear remap of the page's actual dark/light
    # extremes (robust percentiles, not literal min/max, so a single
    # scanner-noise pixel can't skew it) onto the full 0-255 range,
    # before CLAHE/sharpening run. This is what fixes the "washed out"
    # look -- whitening alone cannot, since it only ever acts on pixels
    # already classified as background.
    enable_auto_contrast: bool = True
    auto_contrast_black_percentile: float = 1.0   # robust "true black" point
    auto_contrast_white_percentile: float = 99.0  # robust "true white" point
    auto_contrast_min_range: float = 20.0
    # Applied AFTER the linear black/white stretch above, on the already-
    # stretched 0-255 range: out = 255*(in/255)**gamma. The linear stretch
    # alone only guarantees the extreme (1st-percentile) darkest pixels hit
    # true black -- it does nothing extra for the much larger population of
    # partial-coverage/anti-aliased ink pixels (thin serif strokes, aged/
    # typewriter-era scans) that sit in the middle of the range even after
    # stretching, since a linear remap preserves each pixel's relative
    # position. gamma>1 pushes midtones darker while leaving 0 and 255
    # fixed, so true whites/true blacks are untouched. 1.0 = off (identical
    # to pre-existing behavior). Verified against a real scanned page: raw
    # ink-pixel median L=106 -> linear-stretch-only median=60 (still
    # visibly grey) -> stretch+gamma(1.8) median=19 (reads as black),
    # background patch mean moved <2 L in the same test. Tune 1.4-2.2;
    # higher darkens more but starts eating thin anti-aliased edges.
    auto_contrast_gamma: float = 1.8
    # Skip the stretch if (white_point - black_point) is already below
    # this -- on a near-flat/blank tile (empty margin, solid-color photo
    # region) there's no real ink contrast to recover, and stretching a
    # ~10-level noise floor up to 0-255 just amplifies scanner grain/JPEG
    # blocking instead of revealing text.

    # ── Diagram background whitening (grey-boundary cleanup) ───

    whiten_block_size: int = 15         # adaptiveThreshold block size (px). Leave
                                         # fixed — bigger windows pull the local
                                         # mean toward dark lines and tighten the
                                         # background threshold instead of relaxing it.
    whiten_c: float = 9.0               # base adaptiveThreshold C offset
    whiten_c_resolution_scale: float = 1.0
    whiten_c_max: float = 14.0
    # adaptiveThreshold(THRESH_BINARY) only tests pixel > local_mean - C --
    # a purely RELATIVE test. It has no notion of absolute brightness, so
    # any large *uniformly dark* region (the interior of a bold letter, a
    # solid ink stamp fill, a logo) also satisfies pixel > local_mean - C,
    # because inside such a region local_mean already equals the pixel's
    # own (dark) value. Left ungated, this hollows every solid glyph into
    # an outline: true interior pixels get flagged "background" and forced
    # to white, while only the thin anti-aliased boundary ring (where the
    # neighborhood straddles ink and paper, pulling local_mean up) stays
    # dark -- because that ring is the one place the relative test still
    # fails. whiten_min_local_brightness adds the missing ABSOLUTE check:
    # a pixel only qualifies as background if its local neighborhood mean
    # is itself already near-paper-white, not merely locally uniform.
    # 0-255 scale; 140 is a safe floor for scanned pages (true paper is
    # usually 200+). Raise if faint grey paper texture still survives;
    # lower only if legitimate light-grey backgrounds stop whitening.
    whiten_min_local_brightness: float = 140.0
    # How much to grow whiten_c as image resolution grows beyond a ~900px
    # reference (e.g. after 4x SR). 0 = no scaling. Growth is sqrt-dampened
    # (not linear -- linear growth combined with the broadened whitening
    # gate below pushed C past 30 on real 4x-SR'd full pages, aggressive
    # enough to wash out legitimate thin/antialiased line content on dense
    # schematics, not just clean up boundary halos) and hard-capped at
    # whiten_c_max regardless of resolution. Tune against your own pages
    # with debug_whitening=True — the right value depends on your SR
    # model/strength, not a universal constant.
    #
    # NOTE on whitening scope: this now also fires whenever a block simply
    # has a light background (see _has_light_background in
    # image_enhancer.py), not only when classify_image's is_diagram fires
    # -- that classifier's color_std<80 requirement almost never matches
    # real tightly-cropped high-contrast text/line content (verified:
    # bimodal black-on-white crops read as HIGH std, 96-122 in testing, not
    # low), so whitening was effectively dead code for most real content
    # before this. If output looks over-corrected/washed out, whiten_c_max
    # is the knob to lower first; if grey boundaries are still visible,
    # whiten_c or whiten_c_resolution_scale are the ones to raise.

    # ── Debugging ─────────────────────────────────────────────
    debug_whitening: bool = False  # Dump amplified before/after diff of the
                                    # final diagram-background whitening pass
                                    # to /tmp/pdf_enhancer_debug for inspection


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
            has_anthropic = bool(v.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"))
            has_gemini    = bool(v.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"))
            if not has_anthropic and not has_gemini:
                logger.warning("No ANTHROPIC_API_KEY or GEMINI_API_KEY set — AI features will be limited")
            elif not has_anthropic and has_gemini:
                logger.info("ANTHROPIC_API_KEY not set — using Gemini for AI features")
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