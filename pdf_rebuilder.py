# pdf_rebuilder.py
# Full replacement with rotation-aware transforms, duplicate-image skipping,
# and PyMuPDF prerotate/prerotate compatibility guard.

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
        new_doc.save(
            str(output_path),
            garbage=4,
            deflate=True,
        )
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

    # ── Heuristics ────────────────────────────────────────────

    def _should_skip_block(self, page_info: PageInfo, img_block: ImageBlock) -> bool:
        """
        Heuristic to decide whether an image block is already present in the
        full-page rendered image and should be skipped to avoid duplication.
        Prefer setting img_block.from_rendered in the extractor for deterministic behavior.
        """
        # If extractor explicitly marked it as coming from the rendered image, skip.
        if getattr(img_block, "from_rendered", False):
            return True

        # If there's no rendered image, don't skip.
        if page_info.rendered_image is None:
            return False

        # If the image block covers a large fraction of the page, assume it's part of the render.
        page_area = page_info.width * page_info.height
        b = img_block.bbox
        block_area = max((b.x1 - b.x0) * (b.y1 - b.y0), 0.0)
        if page_area > 0 and (block_area / page_area) > 0.6:
            return True

        return False

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

        # Create new page sized for scaled output
        new_page = new_doc.new_page(width=w * scale, height=h * scale)

        # Determine page rotation (0 if not provided)
        rotation = getattr(page_info, "rotation", 0) or 0

        # Build a single matrix that scales then rotates to map source coords -> new_page visual coords
        m = fitz.Matrix(scale, scale)
        # Compatibility: some PyMuPDF versions expose 'prerotate', others 'preRotate'
        if hasattr(m, "prerotate"):
            try:
                transform = m.prerotate(rotation)
            except Exception:
                transform = m
        elif hasattr(m, "preRotate"):
            try:
                transform = m.preRotate(rotation)
            except Exception:
                transform = m
        else:
            transform = m

        # ── Background: full-page rasterized render ───────────
        if page_info.rendered_image is not None:
            bg_img = page_info.rendered_image

            # Enhance entire page render for scanned docs
            if is_scanned:
                bg_img = self.enhancer.enhance_page_render(bg_img)

            # Insert the full-page background using the full target rect
            self._insert_image_on_page(
                new_page, bg_img,
                fitz.Rect(0, 0, w * scale, h * scale)
            )

        # ── Enhanced embedded images ──────────────────────────
        for img_block in page_info.image_blocks:
            try:
                # Skip blocks that are effectively part of the full-page render
                if self._should_skip_block(page_info, img_block):
                    logger.debug(
                        f"  Skipping image block on p{page_info.page_num} (covered by render)"
                    )
                    continue

                # Classify
                img_block = self.diagrams.classify_image(img_block)

                # Enhance
                enhanced = self.enhancer.enhance_block(img_block)
                pq.images_enhanced += 1

                # Diagram: attempt vectorization
                if img_block.is_diagram:
                    d_result = self.diagrams.analyze(img_block)
                    if d_result.svg_bytes and d_result.use_vector and CAIROSVG_AVAILABLE:
                        # Convert SVG → PNG sized to enhanced image dimensions
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

                # Position on page: transform the original bbox using the combined matrix
                src_rect = fitz.Rect(
                    img_block.bbox.x0,
                    img_block.bbox.y0,
                    img_block.bbox.x1,
                    img_block.bbox.y1,
                )

                # Apply transform: matrix * rect -> visual coordinates on new_page
                try:
                    dst_rect = transform * src_rect
                except Exception:
                    # Fallback: manually scale coords if transform multiplication not supported
                    dst_rect = fitz.Rect(
                        src_rect.x0 * scale,
                        src_rect.y0 * scale,
                        src_rect.x1 * scale,
                        src_rect.y1 * scale,
                    )

                # Defensive clamp to page rect
                dst_rect = dst_rect & fitz.Rect(0, 0, w * scale, h * scale)

                logger.debug(
                    f"  Embedding image p{page_info.page_num} src={src_rect} dst={dst_rect} rot={rotation} scale={scale}"
                )

                # Insert image bytes into the computed rectangle
                new_page.insert_image(dst_rect, stream=img_to_embed)

            except Exception as e:
                logger.warning(f"  Image embed failed p{page_info.page_num}: {e}")

        # ── Invisible searchable text layer ───────────────────
        if self.cfg.embed_searchable_text:
            self._embed_text_layer(new_page, page_info, is_scanned, scale, pq)

        # ── Tables ────────────────────────────────────────────
        for table in page_info.table_blocks:
            try:
                # Transform table bbox similarly before embedding
                src_t = fitz.Rect(table.bbox.x0, table.bbox.y0, table.bbox.x1, table.bbox.y1)
                try:
                    dst_t = transform * src_t
                except Exception:
                    dst_t = fitz.Rect(
                        src_t.x0 * scale,
                        src_t.y0 * scale,
                        src_t.x1 * scale,
                        src_t.y1 * scale,
                    )

                # Convert transformed rect back into a TableBlock-like bbox for _embed_table
                scaled_table = TableBlock(
                    bbox = type(table.bbox)(
                        x0 = dst_t.x0,
                        y0 = dst_t.y0,
                        x1 = dst_t.x1,
                        y1 = dst_t.y1,
                    ),
                    data = table.data,
                )
                # _embed_table expects a scale factor; since we've already transformed coords,
                # call with scale=1.0 and let _embed_table use the bbox directly.
                self._embed_table(new_page, scaled_table, scale=1.0)
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
        Expects table.bbox to already be in page coordinates (i.e., transformed if needed).
        If scale != 1.0, the bbox will be scaled by that factor.
        """
        if not table.data or not table.data[0]:
            return

        rows = table.data
        n_rows = len(rows)
        n_cols = max(len(r) for r in rows)

        # If scale was provided, apply it to bbox coordinates
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
        self, page: fitz.Page, img: np.ndarray | bytes, rect: fitz.Rect
    ) -> None:
        """
        Insert a numpy image (H,W,3/4) or raw bytes into the page rect. Convert to PNG bytes if needed.
        """
        # If img is already bytes (e.g., PNG bytes), accept that too
        if isinstance(img, (bytes, bytearray)):
            page.insert_image(rect, stream=bytes(img))
            return

        pil = Image.fromarray(img.astype(np.uint8))
        # Flatten alpha if present to avoid blending artifacts
        if pil.mode in ("RGBA", "LA"):
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        # Use PNG with low compression for speed; set optimize if you prefer smaller size
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
