

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import fitz                         # PyMuPDF
import numpy as np
from PIL import Image, UnidentifiedImageError
from loguru import logger
from tqdm import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

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
    page_quality:    List[PageQuality] = field(default_factory=list)
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

        # Whether to enhance full-page images (images that cover most of page)
        self.enhance_full_page_images = getattr(self.full_cfg.enhancement, "enhance_full_page_images", True)

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

        # Set metadata (keep safe subset)
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

    # ── Heuristics ───────────────────────────────────────────

    def _should_skip_block(self, page_info: PageInfo, img_block: ImageBlock) -> bool:
        if getattr(img_block, "from_rendered", False):
            return True
        if page_info.rendered_image is None:
            return False
        page_area = page_info.width * page_info.height
        b = img_block.bbox
        block_area = max((b.x1 - b.x0) * (b.y1 - b.y0), 0.0)
        frac = (block_area / page_area) if page_area > 0 else 0.0
        if frac > 0.9:
            return not self.enhance_full_page_images
        if frac > 0.6 and not self.enhance_full_page_images:
            return True
        return False

    # ── Per-page build ───────────────────────────────────────

    def _build_page(
        self,
        new_doc:    fitz.Document,
        page_info:  PageInfo,
        is_scanned: bool,
        pq:         PageQuality,
    ) -> None:
        w, h = page_info.width, page_info.height
        scale = self.cfg.output_dpi / 72.0
        new_page = new_doc.new_page(width=w * scale, height=h * scale)
        rotation = getattr(page_info, "rotation", 0) or 0

        m = fitz.Matrix(scale, scale)
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

        # Background render
        if page_info.rendered_image is not None:
            bg_img = page_info.rendered_image
            if is_scanned:
                try:
                    bg_img = self.enhancer.enhance_page_render(bg_img)
                except Exception as e:
                    logger.debug(f"  Page render enhancement failed p{page_info.page_num}: {e}")
            try:
                self._insert_image_on_page(new_page, bg_img, fitz.Rect(0, 0, w * scale, h * scale))
            except Exception as e:
                logger.debug(f"  Failed to insert page background p{page_info.page_num}: {e}")

        # Embedded images
        for img_block in page_info.image_blocks:
            try:
                if self._should_skip_block(page_info, img_block):
                    logger.debug(f"  Skipping image block on p{page_info.page_num} (covered by render)")
                    continue

                # Classify
                try:
                    img_block = self.diagrams.classify_image(img_block)
                except Exception as e:
                    logger.debug(f"  Diagram classification failed for xref={img_block.xref}: {e}")

                # Ensure image_bytes exists; synthesize from page render if missing
                if not getattr(img_block, "image_bytes", None):
                    self._synthesize_image_bytes_from_render(img_block, page_info)

                # Try enhancement with one retry
                enhanced = None
                try:
                    enhanced = self.enhancer.enhance_block(img_block)
                    pq.images_enhanced += 1
                    logger.debug(f"  Enhanced image xref={img_block.xref} p{page_info.page_num}")
                except Exception as e:
                    logger.debug(f"  Enhancement attempt failed for xref={img_block.xref} p{page_info.page_num}: {e}")
                    # Retry by synthesizing bytes from render (if available) and retry once
                    if page_info.rendered_image is not None:
                        self._synthesize_image_bytes_from_render(img_block, page_info)
                        try:
                            enhanced = self.enhancer.enhance_block(img_block)
                            pq.images_enhanced += 1
                            logger.debug(f"  Enhancement retry succeeded xref={img_block.xref} p{page_info.page_num}")
                        except Exception as e2:
                            logger.warning(f"  Enhancement retry failed xref={img_block.xref} p{page_info.page_num}: {e2}")
                            enhanced = img_block
                    else:
                        enhanced = img_block

                if enhanced is None:
                    enhanced = img_block

                # If full-page and configured to replace, do so
                page_area = page_info.width * page_info.height
                b = img_block.bbox
                block_area = max((b.x1 - b.x0) * (b.y1 - b.y0), 0.0)
                frac = (block_area / page_area) if page_area > 0 else 0.0

                if frac > 0.9 and self.enhance_full_page_images:
                    try:
                        img_bytes = getattr(enhanced, "image_bytes", None) or enhanced
                        normalized = self._ensure_valid_image_bytes(img_bytes, enhanced, img_block, page_info)
                        if normalized:
                            pil = Image.open(io.BytesIO(normalized)).convert("RGB")
                            arr = np.array(pil)
                            self._insert_image_on_page(new_page, arr, fitz.Rect(0, 0, w * scale, h * scale))
                            logger.debug(f"  Replaced page render with enhanced full-page image p{page_info.page_num}")
                            if getattr(img_block, "is_diagram", False):
                                pq.diagrams_vectorized += 1 if getattr(enhanced, "is_diagram", False) else 0
                            continue
                    except Exception as e:
                        logger.debug(f"  Failed to replace page render with enhanced image p{page_info.page_num}: {e}")

                # Prepare bytes to embed
                img_to_embed = getattr(enhanced, "image_bytes", None) or enhanced
                normalized = self._ensure_valid_image_bytes(img_to_embed, enhanced, img_block, page_info)
                if not normalized:
                    logger.warning(f"  Skipping embedding for xref={img_block.xref} p{page_info.page_num}: invalid image bytes")
                    continue

                # Transform bbox -> destination rect
                src_rect = fitz.Rect(img_block.bbox.x0, img_block.bbox.y0, img_block.bbox.x1, img_block.bbox.y1)
                try:
                    dst_rect = transform * src_rect
                except Exception:
                    dst_rect = fitz.Rect(src_rect.x0 * scale, src_rect.y0 * scale, src_rect.x1 * scale, src_rect.y1 * scale)
                dst_rect = dst_rect & fitz.Rect(0, 0, w * scale, h * scale)

                # Diagram vectorization (best-effort)
                if getattr(img_block, "is_diagram", False):
                    try:
                        d_result: DiagramResult = self.diagrams.analyze(enhanced)
                        if getattr(d_result, "svg_bytes", None) and getattr(d_result, "use_vector", False) and CAIROSVG_AVAILABLE:
                            try:
                                png_bytes = cairosvg.svg2png(bytestring=d_result.svg_bytes, output_width=getattr(enhanced, "width", None) or img_block.width, output_height=getattr(enhanced, "height", None) or img_block.height)
                                png_norm = self._ensure_valid_image_bytes(png_bytes, enhanced, img_block, page_info)
                                if png_norm:
                                    normalized = png_norm
                                    pq.diagrams_vectorized += 1
                                    logger.info(f"  Vectorized diagram xref={img_block.xref} p{page_info.page_num}")
                            except Exception as e:
                                logger.debug(f"  SVG rasterization failed for xref={img_block.xref}: {e}")
                    except Exception as e:
                        logger.debug(f"  Diagram analysis failed for xref={img_block.xref}: {e}")

                # Insert image
                try:
                    new_page.insert_image(dst_rect, stream=normalized)
                except Exception as e:
                    logger.debug(f"  insert_image failed for p{page_info.page_num} xref={img_block.xref}: {e}")
                    # Fallback: open with PIL and insert via _insert_image_on_page
                    try:
                        pil = Image.open(io.BytesIO(normalized)).convert("RGB")
                        arr = np.array(pil)
                        self._insert_image_on_page(new_page, arr, dst_rect)
                    except Exception as e2:
                        logger.warning(f"  Fallback embed failed for p{page_info.page_num} xref={img_block.xref}: {e2}")

            except Exception as e:
                logger.warning(f"  Image embed failed p{page_info.page_num}: {e}")

        # Text layer
        if self.cfg.embed_searchable_text:
            self._embed_text_layer(new_page, page_info, is_scanned, scale, pq)

        # Tables
        for table in page_info.table_blocks:
            try:
                src_t = fitz.Rect(table.bbox.x0, table.bbox.y0, table.bbox.x1, table.bbox.y1)
                try:
                    dst_t = transform * src_t
                except Exception:
                    dst_t = fitz.Rect(src_t.x0 * scale, src_t.y0 * scale, src_t.x1 * scale, src_t.y1 * scale)

                scaled_table = TableBlock(
                    bbox = type(table.bbox)(
                        x0 = dst_t.x0,
                        y0 = dst_t.y0,
                        x1 = dst_t.x1,
                        y1 = dst_t.y1,
                    ),
                    data = table.data,
                )
                self._embed_table(new_page, scaled_table, scale=1.0)
                pq.tables_embedded += 1
            except Exception as e:
                logger.debug(f"  Table embed failed p{page_info.page_num}: {e}")

    # ── Helpers ───────────────────────────────────────────────

    def _synthesize_image_bytes_from_render(self, img_block: ImageBlock, page_info: PageInfo) -> None:
        """Try to crop page_info.rendered_image to produce PNG bytes for img_block."""
        try:
            if page_info.rendered_image is None:
                return
            rend = page_info.rendered_image
            rh, rw = rend.shape[:2]
            px_scale_x = rw / page_info.width if page_info.width else 1.0
            px_scale_y = rh / page_info.height if page_info.height else 1.0
            x0 = int(max(0, round(img_block.bbox.x0 * px_scale_x)))
            y0 = int(max(0, round(img_block.bbox.y0 * px_scale_y)))
            x1 = int(min(rw, round(img_block.bbox.x1 * px_scale_x)))
            y1 = int(min(rh, round(img_block.bbox.y1 * px_scale_y)))
            if x1 <= x0 or y1 <= y0:
                return
            crop = rend[y0:y1, x0:x1]
            pil = Image.fromarray(crop)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_block.image_bytes = buf.getvalue()
            logger.debug(f"  Synthesized image_bytes from page render for xref={img_block.xref}")
        except Exception as e:
            logger.debug(f"  Synthesis from render failed for xref={img_block.xref}: {e}")

    def _ensure_valid_image_bytes(
        self,
        candidate,
        enhanced_obj,
        img_block: ImageBlock,
        page_info: PageInfo,
    ) -> Optional[bytes]:
        """
        Normalize candidate into valid PNG/JPEG bytes suitable for fitz.insert_image.
        Tries multiple strategies:
          - If bytes: validate with PIL; if invalid, try OpenCV imdecode.
          - If numpy array or PIL Image: convert to PNG bytes.
          - If enhanced_obj provides to_pil/to_numpy/image_bytes, use them.
          - As last resort, synthesize from page render crop.
        Returns bytes or None.
        """
        # 1) If bytes-like, try PIL open/verify
        try:
            if isinstance(candidate, (bytes, bytearray)):
                try:
                    # PIL verify may not detect all issues; attempt open+convert
                    pil = Image.open(io.BytesIO(candidate))
                    pil = pil.convert("RGB")
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    return buf.getvalue()
                except Exception:
                    # Try OpenCV decode if available (handles more raw formats)
                    if CV2_AVAILABLE:
                        try:
                            arr = np.frombuffer(candidate, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                # convert BGR -> RGB
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                pil = Image.fromarray(img)
                                buf = io.BytesIO()
                                pil.save(buf, format="PNG")
                                return buf.getvalue()
                        except Exception:
                            pass
                    # fall through to other attempts
            # 2) If numpy array
            if isinstance(candidate, np.ndarray):
                try:
                    pil = Image.fromarray(candidate.astype(np.uint8))
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    return buf.getvalue()
                except Exception:
                    pass
            # 3) If PIL Image
            if isinstance(candidate, Image.Image):
                try:
                    pil = candidate.convert("RGB")
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    return buf.getvalue()
                except Exception:
                    pass
            # 4) If enhanced_obj has helpers
            if enhanced_obj is not None:
                try:
                    if hasattr(enhanced_obj, "to_pil"):
                        pil = enhanced_obj.to_pil()
                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        return buf.getvalue()
                    if hasattr(enhanced_obj, "to_numpy"):
                        arr = enhanced_obj.to_numpy()
                        pil = Image.fromarray(arr.astype(np.uint8))
                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        return buf.getvalue()
                    if hasattr(enhanced_obj, "image_bytes") and enhanced_obj.image_bytes:
                        try:
                            pil = Image.open(io.BytesIO(enhanced_obj.image_bytes))
                            pil = pil.convert("RGB")
                            buf = io.BytesIO()
                            pil.save(buf, format="PNG")
                            return buf.getvalue()
                        except Exception:
                            # try OpenCV decode
                            if CV2_AVAILABLE:
                                try:
                                    arr = np.frombuffer(enhanced_obj.image_bytes, dtype=np.uint8)
                                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                    if img is not None:
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        pil = Image.fromarray(img)
                                        buf = io.BytesIO()
                                        pil.save(buf, format="PNG")
                                        return buf.getvalue()
                                except Exception:
                                    pass
                except Exception:
                    pass
            # 5) Synthesize from page render crop
            try:
                if page_info and page_info.rendered_image is not None:
                    rend = page_info.rendered_image
                    rh, rw = rend.shape[:2]
                    px_scale_x = rw / page_info.width if page_info.width else 1.0
                    px_scale_y = rh / page_info.height if page_info.height else 1.0
                    x0 = int(max(0, round(img_block.bbox.x0 * px_scale_x)))
                    y0 = int(max(0, round(img_block.bbox.y0 * px_scale_y)))
                    x1 = int(min(rw, round(img_block.bbox.x1 * px_scale_x)))
                    y1 = int(min(rh, round(img_block.bbox.y1 * px_scale_y)))
                    if x1 > x0 and y1 > y0:
                        crop = rend[y0:y1, x0:x1]
                        pil = Image.fromarray(crop)
                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        return buf.getvalue()
            except Exception:
                pass
        except Exception:
            pass

        return None

    # ── Text layer ────────────────────────────────────────────

    def _embed_text_layer(
        self,
        page:      fitz.Page,
        page_info: PageInfo,
        is_scanned: bool,
        scale:     float,
        pq:        PageQuality,
    ) -> None:
        if page_info.text_blocks and not is_scanned:
            for block in page_info.text_blocks:
                if not block.text.strip():
                    continue
                try:
                    page.insert_text(
                        fitz.Point(block.bbox.x0 * scale, block.bbox.y1 * scale),
                        block.text,
                        fontsize = max(block.font_size * scale * 0.9, 1),
                        color    = (1, 1, 1, 0),
                        overlay  = True,
                    )
                except Exception:
                    pass
            return

        if page_info.rendered_image is not None:
            try:
                ocr_result = self.ocr.run_ocr(page_info.rendered_image)
                if ocr_result.text.strip():
                    corrected = self.ocr.correct(ocr_result)
                    rect = page.rect
                    page.insert_textbox(
                        rect,
                        corrected.corrected,
                        fontsize = 8,
                        color    = (1, 1, 1, 0),
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
                shape.draw_rect(cell_rect)
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

    # ── Image insertion ─────────────────────────────────────

    def _insert_image_on_page(
        self, page: fitz.Page, img: np.ndarray | bytes, rect: fitz.Rect
    ) -> None:
        if isinstance(img, (bytes, bytearray)):
            page.insert_image(rect, stream=bytes(img))
            return

        pil = Image.fromarray(img.astype(np.uint8))
        if pil.mode in ("RGBA", "LA"):
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG", compress_level=1)
        page.insert_image(rect, stream=buf.getvalue())

    # ── pikepdf optimization ──────────────────────────────────

    def _pikepdf_optimize(self, pdf_path: Path) -> None:
        try:
            import pikepdf as _pike
            # allow overwriting input file if needed
            with _pike.open(str(pdf_path), allow_overwriting_input=True) as pdf:
                pdf.save(
                    str(pdf_path),
                    compress_streams = True,
                    object_stream_mode = _pike.ObjectStreamMode.generate,
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
