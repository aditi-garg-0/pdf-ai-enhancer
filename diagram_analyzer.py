"""
diagram_analyzer.py
====================
Stage 4: AI diagram intelligence + vectorization.

For every ImageBlock classified as a diagram:
  1. Claude Vision — semantic understanding of diagram content
     (labels, arrows, shapes, relationships, type classification)
  2. Binarization + potrace — raster → clean SVG vector
  3. SVG optimization — remove noise, simplify paths
  4. Optional reconstruction — redraw as clean matplotlib figure
     (for charts, graphs, simple flowcharts)
  5. Produces either enhanced raster OR embedded vector SVG
"""

from __future__ import annotations

import io
import base64
import subprocess
import tempfile
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

import anthropic
import svgwrite

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

from config import Config, DiagramConfig
from extractor import ImageBlock

# Shared across both providers so the JSON schema/parsing stays identical,
# and so the checkpoint API cache key (content + this exact prompt) is
# consistent regardless of which provider actually served a given call.
_DIAGRAM_PROMPT = """Analyze this diagram/image extracted from a PDF document.
Return a JSON object with these exact fields:
{
  "diagram_type": one of [flowchart, graph, schematic, formula, table, chart, map, photo, logo, unknown],
  "title": "inferred title or empty string",
  "description": "concise 1-2 sentence description of what this shows",
  "labels": ["list", "of", "all", "text", "labels", "visible"],
  "relationships": [{"from": "node A", "to": "node B", "label": "arrow label"}],
  "colors_used": ["#hexcolor or color name"],
  "has_text": true or false,
  "is_chart": true or false,
  "is_flowchart": true or false,
  "is_table": true or false,
  "confidence": 0.0 to 1.0
}
Return ONLY valid JSON, no markdown, no explanation."""


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class DiagramAnalysis:
    diagram_type: str                       # flowchart | graph | table | schematic | formula | photo | logo | unknown
    title:        str        = ""
    description:  str        = ""
    labels:       list[str]  = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)  # [{from, to, label}, ...]
    colors_used:  list[str]  = field(default_factory=list)
    has_text:     bool       = False
    is_chart:     bool       = False        # bar/line/pie etc
    is_flowchart: bool       = False
    is_table:     bool       = False
    confidence:   float      = 0.0
    raw_response: str        = ""


@dataclass
class DiagramResult:
    original_block:   ImageBlock
    analysis:         Optional[DiagramAnalysis] = None
    svg_bytes:        Optional[bytes]           = None   # Vectorized SVG
    enhanced_raster:  Optional[bytes]           = None   # Enhanced PNG fallback
    use_vector:       bool                      = False  # Embed as SVG vs raster


# ──────────────────────────────────────────────────────────────
#  Diagram Analyzer
# ──────────────────────────────────────────────────────────────

class DiagramAnalyzer:

    def __init__(self, config: Config):
        self.cfg:      DiagramConfig = config.diagram
        self.api_cfg   = config.api
        self.pipeline_cfg = config.pipeline
        self._client:  Optional[anthropic.Anthropic] = None
        self._gemini_client = None
        self._checkpoint = None

    def _get_checkpoint(self):
        """
        Lazily open the shared checkpoint/API-cache DB (checkpoint.py),
        which existed in this codebase already but was never actually wired
        into anything -- every run re-called the Vision API for every
        diagram block, even re-processing the exact same file repeatedly
        during testing/debugging. Returns None (caching disabled) if
        pipeline.enable_checkpoint is off or checkpoint.py isn't importable.
        """
        if self._checkpoint is not None:
            return self._checkpoint or None
        if not getattr(self.pipeline_cfg, "enable_checkpoint", True):
            self._checkpoint = False
            return None
        try:
            from checkpoint import CheckpointDB
            self._checkpoint = CheckpointDB(self.pipeline_cfg.checkpoint_db)
        except Exception as e:
            logger.debug(f"  Checkpoint DB unavailable for diagram analysis cache: {e}")
            self._checkpoint = False  # sentinel: don't retry every call
        return self._checkpoint or None

    # ── Public API ────────────────────────────────────────────

    def analyze(self, block: ImageBlock) -> DiagramResult:
        """
        Full analysis pipeline for a single diagram ImageBlock.
        """
        result = DiagramResult(original_block=block)

        img = block.to_numpy()

        # Skip tiny images
        if img.shape[0] < self.cfg.min_diagram_size_px or \
           img.shape[1] < self.cfg.min_diagram_size_px:
            return result

        # 1. AI Vision analysis (Claude, with Gemini as a free-tier fallback)
        if self.cfg.enable_ai_analysis and (self.api_cfg.anthropic_api_key or self.api_cfg.gemini_api_key):
            result.analysis = self._ai_analyze(block.image_bytes)

        # 2. Vectorization (potrace)
        if self.cfg.enable_vectorization:
            svg = self._vectorize(img)
            if svg:
                result.svg_bytes = svg
                result.use_vector = self.cfg.embed_as_vector

        return result

    def classify_image(self, block: ImageBlock) -> ImageBlock:
        """
        Quick classification: is this a diagram or a photo?
        Uses heuristics (edge density, color variance) + optional Claude.

        is_photo gates real downstream behavior in image_enhancer.py: it
        disables auto-contrast/black-point correction and routes output
        through lossy JPEG/WebP instead of lossless PNG. Getting it wrong
        on ordinary document content is exactly what was producing "washed
        out" output even after the contrast fix -- the fix was correctly
        implemented but silently skipped for any block this function
        mis-tagged as a photo.

        color_std alone is not a safe photo signal for this pipeline: a
        tightly-cropped, high-contrast black-on-white block (a title, a
        stamp, a paragraph of dense text -- exactly the granularity real
        pages get fragmented into during extraction) is maximally BIMODAL,
        which reads as HIGH std (verified empirically at 97-122 on
        synthetic text/title/stamp crops -- see image_enhancer.py's
        _has_light_background docstring for the same finding), not low.
        Combined with edge_density<0.08 (easily true for a block that's
        mostly white margin around a few lines of text), that alone was
        enough to mis-tag ordinary document blocks as photos.

        Fix: gate is_photo on the block NOT having a paper-white
        background. A real photo essentially never has >75th-percentile
        lightness above 225 covering most of the frame; document/diagram
        content printed on white paper almost always does. This reuses
        the same purpose-built test _has_light_background already relies
        on elsewhere in this codebase for the identical reason.
        """
        img = block.to_numpy()

        # Heuristic: photos have high color variance; diagrams have low
        color_std = float(np.std(img))
        gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges     = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        p75_lightness = float(np.percentile(lab[:, :, 0], 75))
        has_paper_white_background = p75_lightness > 225

        # Diagrams: high edge density, moderate color variance
        # Photos: lower edge density, high color variance, and -- critically --
        # not sitting on a paper-white background (a real photo of a scene
        # doesn't look like this; scanned/printed document content does).
        is_diagram = edge_density > 0.05 and color_std < 80
        is_photo   = color_std > 60 and edge_density < 0.08 and not has_paper_white_background

        block.is_diagram = is_diagram
        block.is_photo   = is_photo and not is_diagram
        return block

    # ── Provider dispatch ──────────────────────────────────────

    def _ai_analyze(self, image_bytes: bytes) -> DiagramAnalysis:
        """
        Try the configured primary provider; fall back to the other one if
        the primary has no key set or its call raises after retries.
        Checks the checkpoint API-response cache first (keyed by image
        content + the prompt), and stores successful results there --
        so re-running the same PDF (a normal part of iterating on
        enhancement settings) doesn't re-pay for/re-call Vision analysis on
        unchanged diagram images.
        """
        ckpt = self._get_checkpoint()
        cache_model_key = "diagram_analysis_v1"  # bump if the prompt/schema changes
        if ckpt:
            cached = ckpt.get_cached_api_response(image_bytes, _DIAGRAM_PROMPT, cache_model_key)
            if cached:
                try:
                    data = json.loads(cached)
                    logger.debug("  Diagram analysis: cache hit")
                    return DiagramAnalysis(**data)
                except Exception:
                    pass  # fall through and re-fetch if the cached blob is malformed

        providers = ["anthropic", "gemini"]
        if self.api_cfg.ai_provider_priority == "gemini":
            providers = ["gemini", "anthropic"]

        for provider in providers:
            if provider == "anthropic" and self.api_cfg.anthropic_api_key:
                try:
                    result = self._claude_analyze(image_bytes)
                    if ckpt:
                        ckpt.cache_api_response(image_bytes, _DIAGRAM_PROMPT, cache_model_key, json.dumps(result.__dict__))
                    return result
                except Exception as e:
                    logger.warning(f"  Claude diagram analysis failed, trying next provider: {e}")
            elif provider == "gemini" and self.api_cfg.gemini_api_key and GEMINI_AVAILABLE:
                try:
                    result = self._gemini_analyze(image_bytes)
                    if ckpt:
                        ckpt.cache_api_response(image_bytes, _DIAGRAM_PROMPT, cache_model_key, json.dumps(result.__dict__))
                    return result
                except Exception as e:
                    logger.warning(f"  Gemini diagram analysis failed, trying next provider: {e}")
        return DiagramAnalysis(diagram_type="unknown")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _gemini_analyze(self, image_bytes: bytes) -> DiagramAnalysis:
        client = self._get_gemini_client()
        if client is None:
            return DiagramAnalysis(diagram_type="unknown")

        if image_bytes[:4] == b'\x89PNG':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/png"

        # Same schema/prompt as the Claude path so downstream parsing is identical.
        prompt = _DIAGRAM_PROMPT

        response = client.models.generate_content(
            model=self.api_cfg.gemini_model,
            contents=[
                google_genai_types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                prompt,
            ],
        )

        raw = (response.text or "").strip()

        try:
            raw_clean = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE).strip()
            data = json.loads(raw_clean)
            return DiagramAnalysis(
                diagram_type  = data.get("diagram_type", "unknown"),
                title         = data.get("title", ""),
                description   = data.get("description", ""),
                labels        = data.get("labels", []),
                relationships = data.get("relationships", []),
                colors_used   = data.get("colors_used", []),
                has_text      = data.get("has_text", False),
                is_chart      = data.get("is_chart", False),
                is_flowchart  = data.get("is_flowchart", False),
                is_table      = data.get("is_table", False),
                confidence    = float(data.get("confidence", 0.5)),
                raw_response  = raw,
            )
        except json.JSONDecodeError:
            logger.warning(f"  Gemini returned non-JSON diagram analysis: {raw[:200]}")
            return DiagramAnalysis(
                diagram_type = "unknown",
                description  = raw[:500],
                raw_response = raw,
                confidence   = 0.3,
            )

    # ── Claude Vision Analysis ────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _claude_analyze(self, image_bytes: bytes) -> DiagramAnalysis:
        client = self._get_client()
        if client is None:
            return DiagramAnalysis(diagram_type="unknown")

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect media type
        if image_bytes[:4] == b'\x89PNG':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/png"

        prompt = _DIAGRAM_PROMPT

        response = client.messages.create(
            model      = self.api_cfg.claude_model,
            max_tokens = self.api_cfg.max_tokens_analysis,
            messages   = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        raw = response.content[0].text.strip()

        try:
            # Strip markdown fences if Claude wrapped it
            raw_clean = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE).strip()
            data = json.loads(raw_clean)
            return DiagramAnalysis(
                diagram_type  = data.get("diagram_type", "unknown"),
                title         = data.get("title", ""),
                description   = data.get("description", ""),
                labels        = data.get("labels", []),
                relationships = data.get("relationships", []),
                colors_used   = data.get("colors_used", []),
                has_text      = data.get("has_text", False),
                is_chart      = data.get("is_chart", False),
                is_flowchart  = data.get("is_flowchart", False),
                is_table      = data.get("is_table", False),
                confidence    = float(data.get("confidence", 0.5)),
                raw_response  = raw,
            )
        except json.JSONDecodeError:
            logger.warning(f"  Claude returned non-JSON diagram analysis: {raw[:200]}")
            return DiagramAnalysis(
                diagram_type = "unknown",
                description  = raw[:500],
                raw_response = raw,
                confidence   = 0.3,
            )

    # ── Vectorization ─────────────────────────────────────────

    def _vectorize(self, img: np.ndarray) -> Optional[bytes]:
        """
        Convert raster diagram to clean SVG using potrace.
        Pipeline:
          1. Convert to grayscale
          2. Adaptive threshold → clean binary image
          3. Write BMP → potrace → SVG
          4. Return SVG bytes
        """
        # Check potrace is available
        if not self._potrace_available():
            logger.debug("  potrace not found — skipping vectorization")
            return None

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Adaptive threshold — handles uneven lighting better than global
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize = 11,
                C         = 2,
            )

            # Morphological cleanup
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            with tempfile.TemporaryDirectory() as tmpdir:
                bmp_path = Path(tmpdir) / "input.bmp"
                svg_path = Path(tmpdir) / "output.svg"

                # Save as BMP (potrace native format)
                pil_bin = Image.fromarray(binary)
                pil_bin.save(str(bmp_path), format="BMP")

                # Run potrace
                result = subprocess.run(
                    [
                        "potrace",
                        str(bmp_path),
                        "--svg",
                        "-o", str(svg_path),
                        "--turdsize", str(self.cfg.vectorize_min_area),  # Remove noise
                        "--alphamax", "1.0",                              # Smooth corners
                        "--opttolerance", "0.2",                          # Path accuracy
                    ],
                    capture_output = True,
                    timeout        = 30,
                )

                if result.returncode != 0:
                    logger.debug(f"  potrace error: {result.stderr.decode()}")
                    return None

                if svg_path.exists():
                    svg_bytes = svg_path.read_bytes()
                    logger.debug(f"  Vectorized: {len(svg_bytes)} bytes SVG")
                    return svg_bytes

        except subprocess.TimeoutExpired:
            logger.warning("  potrace timeout")
        except Exception as e:
            logger.debug(f"  Vectorization failed: {e}")

        return None

    def _potrace_available(self) -> bool:
        try:
            r = subprocess.run(["potrace", "--version"], capture_output=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── Utilities ─────────────────────────────────────────────

    def _get_client(self) -> Optional[anthropic.Anthropic]:
        if self._client is None and self.api_cfg.anthropic_api_key:
            self._client = anthropic.Anthropic(
                api_key=self.api_cfg.anthropic_api_key
            )
        return self._client

    def _get_gemini_client(self):
        if self._gemini_client is None and self.api_cfg.gemini_api_key and GEMINI_AVAILABLE:
            self._gemini_client = google_genai.Client(api_key=self.api_cfg.gemini_api_key)
        return self._gemini_client