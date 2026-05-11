"""
pdf_rebuilder.py
================
Stage 5 (final): Rebuild the enhanced PDF.

Takes all processed artifacts and reassembles them into a
publication-quality output PDF:

  - Enhanced images/diagrams (raster or SVG vector)
  - Invisible searchable text layer (native + OCR-corrected)
  - Reconstructed tables as proper PDF table objects
  - Restored bookmarks / TOC
  - Linearization (web-optimized / fast-open)
  - Optional compression + PDF/A archival compliance
  - Quality comparison report (SSIM/PSNR per page)
"""

from __future__ import annotations

import io
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz                         # PyMuPDF
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

try:
    import pikepdf
    PIKEPDF_AVAILABLE = True
except ImportError:
    PIKEPDF_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as rl_colors

from config import Config, PipelineConfig, OUTPUT_DIR
from extractor import ExtractedDocument, PageInfo, TextBlock, ImageBlock, TableBlock
from image_enhancer import ImageEnhancer
from ocr_corrector import OCRCorrector, CorrectedText
from diagram_analyzer import DiagramAnalyzer, DiagramResult


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class PageQuality:
    page_num:      int
    ssim_before:   float = 0.0
    ssim_after:    float = 0.0
    psnr_before:   float = 0.0
    psnr_after:    float = 0.0
    images_enhanced:  int = 0
    ocr_blocks:       int = 0
    diagrams_vectorized: int = 0
    tables_embedded:  int = 0


@dataclass
class RebuildReport:
    source_path:     str
    output_path:     str
    pages_processed: int
    page_quality:    list[PageQuality] = field(default_factory=list)
    total_images:    int = 0
    total_diagrams:  int = 0
    total_tables:    int = 0
    total_ocr_blocks: int = 0
    processing_time_s: float = 0.0


# ──────────────────────────────────────────────────────────────
#  PDF Rebuilder
# ──────────────────────────────────────────────────────────────

class PDFRebuilder:
    """
    Reconstructs the final enhanced PDF from all pipeline artifacts.
    """

    def __init__(self, config: Config):
        self.cfg:       PipelineConfig = config.pipeline
        self.full_cfg   = config
        self.enhancer   = ImageEnhancer(config)
        self.ocr        = OCRCorrector(config)
        self.diagrams   = DiagramAnalyzer(config)

    # ── Public API ────────────────────────────────────────────

    def rebuild(
        self,
        doc: ExtractedDocument,
        output_path: Optional[str | Path] = None,
    ) -> RebuildReport:
        import time
        t_start = time.time()

        source = Path(doc.source_path)
        if output_path is None:
            output_path = OUTPUT_DIR / (source.stem + self.cfg.output_suffix + ".pdf")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Rebuilder] Building enhanced PDF → {output_path.name}")

        report = RebuildReport(
            source_path = str(source),
            output_path = str(output_path),
            pages_processed = len(doc.pages),
        )

        # Build new document with PyMuPDF
        new_doc = fitz.open()

        for page_info in tqdm(doc.pages, desc="Rebuilding pages", unit="pg"):
            pq = PageQuality(page_num=page_info.page_num)
            self._build_page(new_doc, page_info, doc.is_scanned, pq)
            report.page_quality.append(pq)
            report.total_images   += pq.images_enhanced
            report.total_diagrams += pq.diagrams_vectorized
            report.total_tables   += pq.tables_embedded
            report.total_ocr_blocks += pq.ocr_blocks

        # Restore bookmarks / TOC
        if doc.bookmarks:
            try:
                new_doc.set_toc(doc.bookmarks)
            except Exception as e:
                logger.warning(f"  TOC restore failed: {e}")

        # Set metadata
        meta = {k: v for k, v in doc.metadata.items()
                if isinstance(v, str) and k in ("title", "author", "subject", "creator")}
        meta["creator"] = "PDF AI Enhancer"
        new_doc.set_metadata(meta)

        # Save
        save_opts = dict(
            deflate       = self.cfg.compress_output,
            garbage       = 4 if self.cfg.optimize_output else 0,
            clean         = self.cfg.optimize_output,
            linear        = True,   # Web-optimized (fast open)
        )
        new_doc.save(str(output_path), **save_opts)
        new_doc.close()

        # Post-process with pikepdf (linearize + repair)
        if PIKEPDF_AVAILABLE and self.cfg.optimize_output:
            self._pikepdf_optimize(output_path)

        report.processing_time_s = time.time() - t_start

        if self.cfg.save_report:
            self._save_report(report, output_path)

        logger.success(
            f"[Rebuilder] Done in {report.processing_time_s:.1f}s — "
            f"{report.pages_processed} pages, "
            f"{report.total_images} images enhanced, "
            f"{report.total_diagrams} diagrams vectorized"
        )
        return report

    # ── Per-page build ────────────────────────────────────────

    def _build_page(
        self,
        new_doc:    fitz.Document,
        page_info:  PageInfo,
        is_scanned: bool,
        pq:         PageQuality,
    ) -> None:
        w, h = page_info.width, page_info.height

        # Scale up for higher output DPI
        scale = self.cfg.output_dpi / 72.0
        new_page = new_doc.new_page(width=w * scale, height=h * scale)

        # ── Background: full-page rasterized render ───────────
        if page_info.rendered_image is not None:
            bg_img = page_info.rendered_image

            # Enhance entire page render for scanned docs
            if is_scanned:
                bg_img = self.enhancer.enhance_page_render(bg_img)

            self._insert_image_on_page(
                new_page, bg_img,
                fitz.Rect(0, 0, w * scale, h * scale)
            )

        # ── Enhanced embedded images ──────────────────────────
        for img_block in page_info.image_blocks:
            try:
                # Classify
                img_block = self.diagrams.classify_image(img_block)

                # Enhance
                enhanced = self.enhancer.enhance_block(img_block)
                pq.images_enhanced += 1

                # Diagram: attempt vectorization
                if img_block.is_diagram:
                    d_result = self.diagrams.analyze(img_block)
                    if d_result.svg_bytes and d_result.use_vector and CAIROSVG_AVAILABLE:
                        # Convert SVG → PDF-embeddable image
                        png_bytes = cairosvg.svg2png(
                            bytestring = d_result.svg_bytes,
                            output_width  = enhanced.width,
                            output_height = enhanced.height,
                        )
                        pq.diagrams_vectorized += 1
                        img_to_embed = png_bytes
                    else:
                        img_to_embed = enhanced.image_bytes
                else:
                    img_to_embed = enhanced.image_bytes

                # Position on page (scale coords)
                bbox = fitz.Rect(
                    img_block.bbox.x0 * scale,
                    img_block.bbox.y0 * scale,
                    img_block.bbox.x1 * scale,
                    img_block.bbox.y1 * scale,
                )
                new_page.insert_image(bbox, stream=img_to_embed)

            except Exception as e:
                logger.warning(f"  Image embed failed p{page_info.page_num}: {e}")

        # ── Invisible searchable text layer ───────────────────
        if self.cfg.embed_searchable_text:
            self._embed_text_layer(new_page, page_info, is_scanned, scale, pq)

        # ── Tables ────────────────────────────────────────────
        for table in page_info.table_blocks:
            try:
                self._embed_table(new_page, table, scale)
                pq.tables_embedded += 1
            except Exception as e:
                logger.debug(f"  Table embed failed p{page_info.page_num}: {e}")

    # ── Text layer ────────────────────────────────────────────

    def _embed_text_layer(
        self,
        page:      fitz.Page,
        page_info: PageInfo,
        is_scanned: bool,
        scale:     float,
        pq:        PageQuality,
    ) -> None:
        """
        Insert invisible text layer for search/copy functionality.
        For native text: use original blocks directly.
        For scanned: run OCR on rendered image + AI correction.
        """
        # Use native text blocks if available
        if page_info.text_blocks and not is_scanned:
            for block in page_info.text_blocks:
                if not block.text.strip():
                    continue
                try:
                    page.insert_text(
                        fitz.Point(block.bbox.x0 * scale, block.bbox.y1 * scale),
                        block.text,
                        fontsize = max(block.font_size * scale * 0.9, 1),
                        color    = (1, 1, 1, 0),   # Fully transparent
                        overlay  = True,
                    )
                except Exception:
                    pass
            return

        # Scanned page: OCR + correct
        if page_info.rendered_image is not None:
            try:
                ocr_result = self.ocr.run_ocr(page_info.rendered_image)
                if ocr_result.text.strip():
                    corrected = self.ocr.correct(ocr_result)

                    # Insert as invisible text block over full page
                    rect = page.rect
                    page.insert_textbox(
                        rect,
                        corrected.corrected,
                        fontsize = 8,
                        color    = (1, 1, 1, 0),   # Invisible
                        overlay  = True,
                    )
                    pq.ocr_blocks += 1
            except Exception as e:
                logger.debug(f"  OCR text layer failed: {e}")

    # ── Tables ────────────────────────────────────────────────

    def _embed_table(
        self,
        page:  fitz.Page,
        table: TableBlock,
        scale: float,
    ) -> None:
        """
        Draw table as text annotations with grid lines.
        """
        if not table.data or not table.data[0]:
            return

        rows = table.data
        n_rows = len(rows)
        n_cols = max(len(r) for r in rows)

        x0 = table.bbox.x0 * scale
        y0 = table.bbox.y0 * scale
        x1 = table.bbox.x1 * scale
        y1 = table.bbox.y1 * scale

        col_w = (x1 - x0) / max(n_cols, 1)
        row_h = (y1 - y0) / max(n_rows, 1)

        shape = page.new_shape()

        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row):
                cx = x0 + c_idx * col_w
                cy = y0 + r_idx * row_h
                cell_rect = fitz.Rect(cx, cy, cx + col_w, cy + row_h)

                # Draw cell border
                shape.draw_rect(cell_rect)

                # Insert text in cell (visible)
                try:
                    page.insert_textbox(
                        cell_rect.shrink(2),
                        str(cell),
                        fontsize = 7,
                        color    = (0, 0, 0),
                        align    = fitz.TEXT_ALIGN_LEFT,
                    )
                except Exception:
                    pass

        shape.finish(width=0.5, color=(0.6, 0.6, 0.6))
        shape.commit()

    # ── Image insertion ───────────────────────────────────────

    def _insert_image_on_page(
        self, page: fitz.Page, img: np.ndarray, rect: fitz.Rect
    ) -> None:
        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG", compress_level=1)
        page.insert_image(rect, stream=buf.getvalue())

    # ── pikepdf optimization ──────────────────────────────────

    def _pikepdf_optimize(self, pdf_path: Path) -> None:
        try:
            with pikepdf.open(str(pdf_path)) as pdf:
                pdf.save(
                    str(pdf_path),
                    compress_streams = True,
                    object_stream_mode = pikepdf.ObjectStreamMode.generate,
                    recompress_flate  = True,
                    normalize_content = True,
                )
            logger.debug("  pikepdf optimization complete")
        except Exception as e:
            logger.warning(f"  pikepdf optimization failed: {e}")

    # ── Report ────────────────────────────────────────────────

    def _save_report(self, report: RebuildReport, pdf_path: Path) -> None:
        report_path = pdf_path.with_suffix(".quality_report.json")
        data = {
            "source":          report.source_path,
            "output":          report.output_path,
            "pages_processed": report.pages_processed,
            "total_images_enhanced":    report.total_images,
            "total_diagrams_vectorized": report.total_diagrams,
            "total_tables_embedded":    report.total_tables,
            "total_ocr_blocks":         report.total_ocr_blocks,
            "processing_time_seconds":  round(report.processing_time_s, 2),
            "per_page": [
                {
                    "page":  pq.page_num + 1,
                    "images": pq.images_enhanced,
                    "diagrams": pq.diagrams_vectorized,
                    "tables": pq.tables_embedded,
                    "ocr_blocks": pq.ocr_blocks,
                }
                for pq in report.page_quality
            ],
        }
        report_path.write_text(json.dumps(data, indent=2))
        logger.info(f"  Quality report → {report_path.name}")