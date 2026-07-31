from __future__ import annotations

import io
import os
import logging
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageFilter, UnidentifiedImageError
import numpy as np
import fitz  # PyMuPDF

# Optional OCR engines
try:
    import pytesseract
    _OCR_ENGINE = "pytesseract"
except Exception:
    pytesseract = None
    _OCR_ENGINE = None

_paddleocr_instance = None
if _OCR_ENGINE is None:
    try:
        from paddleocr import PaddleOCR
        _paddleocr_instance = PaddleOCR(use_angle_cls=False, lang="en")
        _OCR_ENGINE = "paddleocr"
    except Exception:
        _paddleocr_instance = None
        _OCR_ENGINE = None

# Optional OpenCV for extra decode robustness
try:
    import cv2
    _CV2 = True
except Exception:
    cv2 = None
    _CV2 = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cfg_get(cfg, key_path, default=None):
    """
    Safe config getter that supports:
      - dicts: cfg_get(cfg, "pipeline.min_block_area_ratio")
      - attribute objects: cfg_get(cfg, "pipeline.min_block_area_ratio")
    key_path may be dotted (e.g., "pipeline.min_block_area_ratio").
    """
    if cfg is None:
        return default
    parts = key_path.split(".") if isinstance(key_path, str) else [key_path]
    cur = cfg
    for p in parts:
        if cur is None:
            return default
        # dict-like
        try:
            if isinstance(cur, dict):
                cur = cur.get(p, None)
                continue
        except Exception:
            pass
        # attribute-style
        try:
            cur = getattr(cur, p)
            continue
        except Exception:
            try:
                cur = cur[p]
                continue
            except Exception:
                return default
    return cur if cur is not None else default


def pil_to_png_bytes(pil_img: Image.Image, compress_level: int = 1) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def pil_remove_alpha_and_to_png_bytes(pil_img: Image.Image, compress_level: int = 1) -> bytes:
    """
    Ensure image has no alpha by compositing onto white, then return PNG bytes.
    """
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, pil_img.convert("RGBA")).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")
    return pil_to_png_bytes(pil_img, compress_level=compress_level)


def estimate_dpi_from_block(page_rect: fitz.Rect, pixel_width: int) -> Optional[float]:
    try:
        page_inch_w = page_rect.width / 72.0
        if page_inch_w <= 0:
            return None
        return float(pixel_width / page_inch_w)
    except Exception:
        return None


def compute_edge_density(gray_pil: Image.Image) -> float:
    edges = gray_pil.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype(float)
    return float((edge_arr > 20).mean())


def compute_text_density(gray_pil: Image.Image) -> float:
    """
    Try pytesseract first, then paddleocr. Return fraction of pixels covered by text.
    If no OCR engine available, return 0.0 (treated as low text density).
    """
    try:
        px_w, px_h = gray_pil.size
        total_pixels = px_w * px_h
        if total_pixels == 0:
            return 0.0

        if _OCR_ENGINE == "pytesseract" and pytesseract is not None:
            data = pytesseract.image_to_data(gray_pil, output_type=pytesseract.Output.DICT)
            text_pixels = 0
            n = len(data.get("width", []))
            for i in range(n):
                w = int(data["width"][i] or 0)
                h = int(data["height"][i] or 0)
                text_pixels += w * h
            return float(min(text_pixels, total_pixels)) / float(total_pixels)

        if _OCR_ENGINE == "paddleocr" and _paddleocr_instance is not None:
            arr = np.array(gray_pil.convert("RGB"))
            try:
                result = _paddleocr_instance.ocr(arr, cls=False)
            except Exception:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                    gray_pil.convert("RGB").save(tmp.name)
                    result = _paddleocr_instance.ocr(tmp.name, cls=False)
            text_pixels = 0
            for line in result:
                for box_info in line:
                    box = box_info[0]
                    xs = [int(p[0]) for p in box]
                    ys = [int(p[1]) for p in box]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    if w > 0 and h > 0:
                        text_pixels += w * h
            return float(min(text_pixels, total_pixels)) / float(total_pixels)
    except Exception:
        logger.debug("OCR text density failed, treating as low text density")
        return 0.0
    return 0.0


def compute_cdr(
    image_bytes: bytes,
    page_rect: fitz.Rect,
    block_rect: fitz.Rect,
    min_area_ratio: float = 0.05,
    min_dpi: int = 150,
    edge_threshold: float = 0.06,
    text_threshold: float = 0.02,
) -> Tuple[bool, Dict]:
    diagnostics: Dict = {}
    try:
        page_area = page_rect.width * page_rect.height
        block_area = block_rect.width * block_rect.height
        area_ratio = block_area / page_area if page_area > 0 else 0.0
        diagnostics["area_ratio"] = float(area_ratio)

        pil_gray = Image.open(io.BytesIO(image_bytes)).convert("L")
        px_w, px_h = pil_gray.size
        diagnostics["pixel_size"] = (px_w, px_h)

        est_dpi = estimate_dpi_from_block(page_rect, px_w)
        diagnostics["est_dpi"] = float(est_dpi) if est_dpi is not None else None

        edge_density = compute_edge_density(pil_gray)
        diagnostics["edge_density"] = float(edge_density)

        text_density = compute_text_density(pil_gray)
        diagnostics["text_density"] = float(text_density)

        score = 0.0
        score += 1.0 if area_ratio >= min_area_ratio else 0.0
        score += 1.0 if (est_dpi is None or est_dpi >= min_dpi) else 0.0
        score += 1.0 if edge_density > edge_threshold else 0.0
        score += 1.0 if text_density < text_threshold else 0.0

        diagnostics["score"] = float(score)
        should = score >= 2.5
        return should, diagnostics
    except Exception as e:
        logger.exception("compute_cdr failed: %s", e)
        fallback = area_ratio >= min_area_ratio if "area_ratio" in locals() else True
        diagnostics["error"] = str(e)
        return fallback, diagnostics


class PDFRebuilder:
    def __init__(self, config):
        """
        config: your Config object or dict
        """
        self.cfg = config
        # pipeline-level config may be nested; use cfg_get to read safely
        self.min_block_area_ratio = float(cfg_get(self.cfg, "pipeline.min_block_area_ratio", 0.05))
        self.min_resolution_dpi = int(cfg_get(self.cfg, "pipeline.min_resolution_dpi", 150))
        self.jpeg_quality = int(cfg_get(self.cfg, "enhancement.jpeg_quality", 95))
        self.debug_dir = cfg_get(self.cfg, "pipeline.debug_dir", None)
        self.enhancer = None
        self.ocr = None
        self.diagram = None

        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    def _init_components(self):
        if self.enhancer is None:
            from image_enhancer import ImageEnhancer
            self.enhancer = ImageEnhancer(self.cfg)
        if self.ocr is None:
            try:
                from ocr_corrector import OCRCorrector
                self.ocr = OCRCorrector(self.cfg)
            except Exception:
                self.ocr = None
        if self.diagram is None:
            try:
                from diagram_analyzer import DiagramAnalyzer
                self.diagram = DiagramAnalyzer(self.cfg)
            except Exception:
                self.diagram = None

    def _save_debug(self, prefix: str, image_bytes: bytes):
        if not self.debug_dir:
            return
        try:
            idx = len(os.listdir(self.debug_dir))
            path = os.path.join(self.debug_dir, f"{prefix}_{idx}.png")
            with open(path, "wb") as f:
                f.write(image_bytes)
            logger.debug("Saved debug image %s", path)
        except Exception:
            logger.exception("Failed to save debug image")

    def rebuild(self, doc, output_path: str):
        import time
        t0 = time.time()
        self._init_components()

        new_doc = fitz.open()
        for page_info in doc.pages:
            page_doc = self._build_page_doc(doc, page_info)
            new_doc.insert_pdf(page_doc)
            page_doc.close()

        # restore TOC/metadata if present
        try:
            if getattr(doc, "bookmarks", None):
                new_doc.set_toc(doc.bookmarks)
        except Exception:
            logger.debug("Failed to restore TOC")

        try:
            meta = {}
            if getattr(doc, "metadata", None):
                for k, v in doc.metadata.items():
                    if isinstance(v, str) and k in ("title", "author", "subject", "creator"):
                        meta[k] = v
            meta["creator"] = "PDF AI Enhancer"
            new_doc.set_metadata(meta)
        except Exception:
            logger.debug("Failed to set metadata")

        new_doc.save(str(output_path), garbage=4, deflate=True)
        new_doc.close()
        logger.info("Saved rebuilt PDF to %s (%.1fs)", output_path, time.time() - t0)
        return output_path

    def _page_is_scanned(self, page_info, new_page) -> bool:
        """Return True only for pages with no native text and no vector drawings."""
        try:
            text = new_page.get_text("text") or ""
            if text.strip():
                return False
        except Exception:
            pass
        try:
            drawings = new_page.get_drawings()
            if drawings:
                return False
        except Exception:
            pass
        try:
            text_blocks = getattr(page_info, "text_blocks", None)
            if text_blocks:
                for tb in text_blocks:
                    if getattr(tb, "text", "").strip():
                        return False
        except Exception:
            pass
        return True

    def _build_page_doc(self, doc, page_info):
        self._init_components()
        src_path = doc.source_path
        src = fitz.open(src_path)
        try:
            new_doc = fitz.open()
            new_doc.insert_pdf(src, from_page=page_info.page_num, to_page=page_info.page_num)
            new_page = new_doc.load_page(0)
            page_rect = new_page.rect

            is_scanned = self._page_is_scanned(page_info, new_page)

            # A scanned page's image_blocks (as extracted) commonly cover
            # the page cumulatively even though no single block is
            # individually "full page" -- extractor logs on this doc showed
            # image_area=1.0 via 15-153 smaller blocks per page. The old
            # code always painted a full-page background from
            # enhance_page_render() AND THEN pasted every one of those
            # blocks on top of it, each independently re-enhanced through a
            # *different* pipeline -- two independently processed copies of
            # the same content, positioned through two different coordinate
            # paths, is a double-exposure (grey "ghost" offset across every
            # piece of content on the page).
            #
            # First attempt at a fix (raw cumulative bbox-area fraction)
            # caused a regression: blank pages. Reason -- raw bbox coverage
            # doesn't mean those blocks will actually survive compute_cdr's
            # min-area/min-DPI filtering below. On a page fragmented into
            # 15-153 tiny extraction artifacts, most/all of them can get
            # rejected by CDR, so "skip the background because bboxes add
            # up to >90%" left nothing drawn at all. Fixed by actually
            # dry-running the same compute_cdr check used by the real
            # drawing loop, so we only skip the background when blocks are
            # genuinely going to be drawn, not just nominally present.
            # (_synthesize_image_bytes_from_render caches bytes onto the
            # block object, so this doesn't duplicate that work in the real
            # loop below -- only compute_cdr itself runs twice, which is
            # cheap/metadata-level.)
            blocks_cover_page = False
            if is_scanned and getattr(page_info, "image_blocks", None):
                try:
                    page_area = page_info.width * page_info.height
                    covered = 0.0
                    for _blk in page_info.image_blocks:
                        try:
                            if not getattr(_blk, "image_bytes", None):
                                self._synthesize_image_bytes_from_render(_blk, page_info)
                            _bts = getattr(_blk, "image_bytes", None)
                            if not _bts:
                                continue
                            _b = _blk.bbox
                            _rect = fitz.Rect(_b.x0, _b.y0, _b.x1, _b.y1)
                            _should, _ = compute_cdr(
                                _bts, page_rect, _rect,
                                min_area_ratio=self.min_block_area_ratio,
                                min_dpi=self.min_resolution_dpi,
                            )
                            if _should:
                                covered += max((_b.x1 - _b.x0) * (_b.y1 - _b.y0), 0.0)
                        except Exception:
                            continue
                    estimated_frac = (covered / page_area) if page_area > 0 else 0.0
                    blocks_cover_page = estimated_frac > 0.9
                    logger.debug(
                        "Page %d: %.0f%% coverage from blocks that will actually survive "
                        "CDR (blocks_cover_page=%s)", page_info.page_num,
                        estimated_frac * 100, blocks_cover_page,
                    )
                except Exception:
                    blocks_cover_page = False

            if is_scanned:
                # No native text/vector content -- effectively just a raster
                # scan, so it's safe to replace the visual wholesale.
                try:
                    shape = new_page.new_shape()
                    shape.draw_rect(new_page.rect)
                    shape.finish(fill=(1, 1, 1), color=None, width=0)
                    shape.commit()
                except Exception:
                    pass

                if blocks_cover_page:
                    logger.debug(
                        "Page %d: CDR-verified block coverage sufficient -- skipping "
                        "full-page background render to avoid double-exposure with "
                        "per-block enhancement below.", page_info.page_num,
                    )
                elif getattr(page_info, "rendered_image", None) is not None:
                    try:
                        if getattr(self.enhancer, "enhance_page_render", None):
                            bg = self.enhancer.enhance_page_render(page_info.rendered_image)
                            pil = Image.fromarray(bg) if isinstance(bg, np.ndarray) else Image.open(io.BytesIO(bg))
                            new_page.insert_image(new_page.rect, stream=pil_remove_alpha_and_to_png_bytes(pil))
                    except Exception:
                        logger.debug("Background render enhancement failed for page %d", page_info.page_num)
            else:
                # Native text/vector content present: leave the imported page
                # untouched. Do NOT paint a white rect or paste a re-rasterized
                # full-page background over it -- only enhance/overlay the
                # individual embedded raster image blocks below.
                logger.debug(
                    "Page %d has native text/vector content; preserving original "
                    "page, only enhancing embedded image blocks", page_info.page_num
                )

            _blocks_actually_drawn = 0
            for img_block in page_info.image_blocks:
                try:
                    page_area = page_info.width * page_info.height
                    b = img_block.bbox
                    block_area = max((b.x1 - b.x0) * (b.y1 - b.y0), 0.0)
                    frac = (block_area / page_area) if page_area > 0 else 0.0
                    if frac > 0.9 and not cfg_get(self.cfg, "enhancement.enhance_full_page_images", True):
                        logger.debug("Skipping full-page image block on page %d", page_info.page_num)
                        continue

                    if not getattr(img_block, "image_bytes", None):
                        self._synthesize_image_bytes_from_render(img_block, page_info)

                    orig_bytes = getattr(img_block, "image_bytes", None)
                    if not orig_bytes:
                        logger.debug("No bytes for block xref=%s p%d", getattr(img_block, "xref", "n/a"), page_info.page_num)
                        continue

                    rect = fitz.Rect(b.x0, b.y0, b.x1, b.y1)
                    should, diag = compute_cdr(
                        orig_bytes,
                        page_rect,
                        rect,
                        min_area_ratio=self.min_block_area_ratio,
                        min_dpi=self.min_resolution_dpi,
                    )
                    logger.debug("CDR page %d block diag %s", page_info.page_num, diag)
                    if not should:
                        logger.debug("CDR skipped enhancement for page %d block", page_info.page_num)
                        continue

                    if self.diagram:
                        try:
                            img_block = self.diagram.classify_image(img_block)
                        except Exception:
                            logger.debug("Diagram classification failed for xref=%s", getattr(img_block, "xref", "n/a"))

                    try:
                        enhanced = self.enhancer.enhance_block(img_block)
                        enhanced_bytes = getattr(enhanced, "image_bytes", None) or enhanced
                    except Exception as e:
                        logger.warning("Enhancer failed for page %d block: %s", page_info.page_num, e)
                        enhanced_bytes = orig_bytes

                    # Normalize and remove alpha
                    normalized = self._ensure_valid_image_bytes(enhanced_bytes, enhanced if 'enhanced' in locals() else None, img_block, page_info)
                    if not normalized:
                        logger.warning("Normalized bytes invalid for page %d block", page_info.page_num)
                        continue

                    src_rect = fitz.Rect(b.x0, b.y0, b.x1, b.y1)
                    try:
                        new_page.insert_image(src_rect, stream=normalized, keep_proportion=False)
                        _blocks_actually_drawn += 1
                    except Exception:
                        try:
                            pil = Image.open(io.BytesIO(normalized)).convert("RGB")
                            arr = np.array(pil)
                            self._insert_image_on_page(new_page, arr, src_rect)
                            _blocks_actually_drawn += 1
                        except Exception as e:
                            logger.warning("Fallback embed failed p%d: %s", page_info.page_num, e)

                except Exception as e:
                    logger.debug("Image block processing failed p%d: %s", page_info.page_num, e)

            # Emergency fallback: the CDR dry-run predicted blocks would
            # cover the page well enough to skip the background, but the
            # real loop above (which can fail for reasons the dry-run
            # doesn't check -- enhancer exceptions, normalization failures,
            # insert_image failures) ended up drawing nothing or almost
            # nothing. Don't ship a blank page a second time -- paint the
            # background now, late, as a safety net. This can still exhibit
            # the original double-exposure look if a few blocks DID draw
            # successfully, but a slightly-ghosted page beats a blank one.
            if is_scanned and blocks_cover_page and _blocks_actually_drawn == 0 \
                    and getattr(page_info, "rendered_image", None) is not None:
                logger.warning(
                    "Page %d: background was skipped (blocks_cover_page=True) but "
                    "0 blocks actually drew -- painting background as emergency "
                    "fallback to avoid a blank page. Investigate why CDR dry-run "
                    "predicted coverage that didn't materialize.", page_info.page_num,
                )
                try:
                    if getattr(self.enhancer, "enhance_page_render", None):
                        bg = self.enhancer.enhance_page_render(page_info.rendered_image)
                        pil = Image.fromarray(bg) if isinstance(bg, np.ndarray) else Image.open(io.BytesIO(bg))
                        new_page.insert_image(new_page.rect, stream=pil_remove_alpha_and_to_png_bytes(pil), overlay=False)
                except Exception:
                    logger.debug("Emergency background fallback also failed for page %d", page_info.page_num)

            if cfg_get(self.cfg, "pipeline.embed_searchable_text", True):
                try:
                    self._embed_text_layer(new_page, page_info)
                except Exception:
                    logger.debug("Text layer embed failed p%d", page_info.page_num)

            return new_doc
        finally:
            src.close()

    def _synthesize_image_bytes_from_render(self, img_block, page_info):
        try:
            if getattr(page_info, "rendered_image", None) is None:
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
            logger.debug("Synthesized image_bytes from render for xref=%s", getattr(img_block, "xref", "n/a"))
        except Exception as e:
            logger.debug("Synthesis from render failed: %s", e)

    def _ensure_valid_image_bytes(self, candidate, enhanced_obj, img_block, page_info) -> Optional[bytes]:
        try:
            if isinstance(candidate, (bytes, bytearray)):
                try:
                    pil = Image.open(io.BytesIO(candidate))
                    # remove alpha and return PNG bytes
                    return pil_remove_alpha_and_to_png_bytes(pil)
                except Exception:
                    if _CV2:
                        try:
                            arr = np.frombuffer(candidate, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                pil = Image.fromarray(img)
                                return pil_remove_alpha_and_to_png_bytes(pil)
                        except Exception:
                            pass
            if isinstance(candidate, np.ndarray):
                try:
                    pil = Image.fromarray(candidate.astype(np.uint8))
                    return pil_remove_alpha_and_to_png_bytes(pil)
                except Exception:
                    pass
            if isinstance(candidate, Image.Image):
                try:
                    return pil_remove_alpha_and_to_png_bytes(candidate)
                except Exception:
                    pass
            if enhanced_obj is not None:
                try:
                    if hasattr(enhanced_obj, "to_pil"):
                        pil = enhanced_obj.to_pil()
                        return pil_remove_alpha_and_to_png_bytes(pil)
                    if hasattr(enhanced_obj, "to_numpy"):
                        arr = enhanced_obj.to_numpy()
                        pil = Image.fromarray(arr.astype(np.uint8))
                        return pil_remove_alpha_and_to_png_bytes(pil)
                    if hasattr(enhanced_obj, "image_bytes") and enhanced_obj.image_bytes:
                        try:
                            pil = Image.open(io.BytesIO(enhanced_obj.image_bytes)).convert("RGB")
                            return pil_remove_alpha_and_to_png_bytes(pil)
                        except Exception:
                            if _CV2:
                                try:
                                    arr = np.frombuffer(enhanced_obj.image_bytes, dtype=np.uint8)
                                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                    if img is not None:
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        pil = Image.fromarray(img)
                                        return pil_remove_alpha_and_to_png_bytes(pil)
                                except Exception:
                                    pass
                except Exception:
                    pass
            if page_info and getattr(page_info, "rendered_image", None) is not None:
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
                    return pil_remove_alpha_and_to_png_bytes(pil)
        except Exception:
            logger.exception("Normalization failed for block xref=%s", getattr(img_block, "xref", "n/a"))
        return None

    def _embed_text_layer(self, page: fitz.Page, page_info):
        # render_mode=3 = PDF spec "invisible text" (neither fill nor
        # stroke) -- the standard technique real OCR tools (ocrmypdf, etc.)
        # use for a searchable-but-invisible text layer. The previous
        # color=(1,1,1,0) here was an attempt at "white, fully transparent",
        # but insert_text's `color` has no alpha channel: a 4-tuple is read
        # as CMYK, so (1,1,1,0) is C=1 M=1 Y=1 K=0 -- a real, visible
        # mid-grey ink color (measured: renders as RGB ~153,153,155), not
        # invisible at all. That grey copy of the text was being stamped
        # directly on top of every page with extracted/OCR'd text, in a
        # different font/position than the underlying scanned glyphs --
        # two overlapping, slightly mismatched letterforms is exactly what
        # produced the crisp grey "emboss" edges around text (unrelated to
        # the SR/whitening/vectorization work in image_enhancer.py -- this
        # is a separate bug, in the text layer, not the image pipeline).
        if getattr(page_info, "text_blocks", None):
            for block in page_info.text_blocks:
                if not getattr(block, "text", "").strip():
                    continue
                try:
                    page.insert_text(
                        fitz.Point(block.bbox.x0, block.bbox.y1),
                        block.text,
                        fontsize=max(getattr(block, "font_size", 8), 6),
                        render_mode=3,
                        overlay=True,
                    )
                except Exception:
                    pass
            return
        if self.ocr and getattr(page_info, "rendered_image", None) is not None:
            try:
                ocr_result = self.ocr.run_ocr(page_info.rendered_image)
                corrected = self.ocr.correct(ocr_result)
                if corrected and corrected.corrected.strip():
                    page.insert_textbox(page.rect, corrected.corrected, fontsize=8, render_mode=3, overlay=True)
            except Exception:
                logger.debug("OCR text layer failed for page %d", page_info.page_num)

    def _insert_image_on_page(self, page: fitz.Page, img: np.ndarray | bytes, rect: fitz.Rect):
        if isinstance(img, (bytes, bytearray)):
            page.insert_image(rect, stream=bytes(img))
            return
        pil = Image.fromarray(img.astype(np.uint8))
        if pil.mode in ("RGBA", "LA"):
            pil = pil.convert("RGB")
        page.insert_image(rect, stream=pil_to_png_bytes(pil))