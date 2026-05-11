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

from config import Config, DiagramConfig
from extractor import ImageBlock


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
        self._client:  Optional[anthropic.Anthropic] = None

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

        # 1. Claude Vision analysis
        if self.cfg.enable_ai_analysis and self.api_cfg.anthropic_api_key:
            result.analysis = self._claude_analyze(block.image_bytes)

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
        """
        img = block.to_numpy()

        # Heuristic: photos have high color variance; diagrams have low
        color_std = float(np.std(img))
        gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges     = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # Diagrams: high edge density, moderate color variance
        # Photos: lower edge density, high color variance
        is_diagram = edge_density > 0.05 and color_std < 80
        is_photo   = color_std > 60 and edge_density < 0.08

        block.is_diagram = is_diagram
        block.is_photo   = is_photo and not is_diagram
        return block

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

        prompt = """Analyze this diagram/image extracted from a PDF document.
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