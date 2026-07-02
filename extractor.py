# extractor.py
"""
Stage 1 of the pipeline — advanced extractor.

Features:
  - Robust extraction of text spans, embedded images, tables, fonts, bookmarks
  - Verbose per-page logging so you can see why blocks are/aren't passed downstream
  - Relaxed thresholds by default to avoid filtering out diagrams and small images
  - Defensive handling of large images and rendered pages
  - Page classification tuned to surface diagrams even in mixed pages
  - Fallback rendering when embedded image bytes are missing (prevents empty image_bytes)
"""

from __future__ import annotations

import io
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import fitz                     # PyMuPDF
import pdfplumber
import pypdfium2 as pdfium
import numpy as np
import cv2
from PIL import Image
from loguru import logger
from tqdm import tqdm
import warnings

try:
    import camelot
    CAMELOT_AVAILABLE = True
except Exception:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except Exception:
    TABULA_AVAILABLE = False

from config import Config

# Safety: avoid PIL DecompressionBomb spam; still keep a reasonable cap
Image.MAX_IMAGE_PIXELS = 200_000_000  # 200M pixels cap


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class BBox:
    x0: float; y0: float; x1: float; y1: float

    @property
    def width(self) -> float: return self.x1 - self.x0

    @property
    def height(self) -> float: return self.y1 - self.y0

    @property
    def area(self) -> float: return max(0.0, self.width * self.height)

    def to_tuple(self):
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TextBlock:
    text: str
    bbox: BBox
    font_name: str = ""
    font_size: float = 12.0
    color: tuple = (0.0, 0.0, 0.0)
    bold: bool = False
    italic: bool = False
    page_num: int = 0
    block_id: int = 0
    confidence: float = 1.0
    source: str = "native"


@dataclass
class ImageBlock:
    image_bytes: bytes
    bbox: BBox
    page_num: int
    xref: int
    width: int = 0
    height: int = 0
    dpi_x: float = 72.0
    dpi_y: float = 72.0
    colorspace: str = "RGB"
    ext: str = "png"
    sha256: str = ""
    is_diagram: bool = False
    is_photo: bool = False

    def __post_init__(self):
        if not self.sha256:
            try:
                self.sha256 = hashlib.sha256(self.image_bytes).hexdigest()
            except Exception:
                self.sha256 = ""

    def to_pil(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_pil())


@dataclass
class TableBlock:
    data: List[List[str]]
    bbox: BBox
    page_num: int
    method: str
    accuracy: float = 1.0
    dataframe: object = None


@dataclass
class PageInfo:
    page_num: int
    width: float
    height: float
    rotation: int
    has_text: bool = False
    has_images: bool = False
    has_tables: bool = False
    page_type: str = "mixed"
    text_density: float = 0.0
    image_area: float = 0.0
    text_blocks: List[TextBlock] = field(default_factory=list)
    image_blocks: List[ImageBlock] = field(default_factory=list)
    table_blocks: List[TableBlock] = field(default_factory=list)
    rendered_image: Optional[np.ndarray] = None


@dataclass
class ExtractedDocument:
    source_path: str
    page_count: int
    pages: List[PageInfo] = field(default_factory=list)
    fonts: dict = field(default_factory=dict)
    bookmarks: List = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    is_scanned: bool = False


# ──────────────────────────────────────────────────────────────
#  Extractor
# ──────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Advanced PDF extractor. Use Config to tune thresholds.
    """

    # Maximum pixels for stored rendered_image to avoid huge pickles
    MAX_RENDERED_PIXELS = 8_000_000  # ~8MP default

    def __init__(self, config: Config):
        self.cfg = config.extraction
        self.pipeline_cfg = config.pipeline

    # ── Public entry point ───────────────────────────────────

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        logger.info(f"[Extractor] Opening: {pdf_path.name}")

        doc_mupdf = fitz.open(str(pdf_path))
        doc_plumber = pdfplumber.open(str(pdf_path))
        doc_pdfium = pdfium.PdfDocument(str(pdf_path))

        try:
            meta = self._extract_metadata(doc_mupdf)
            fonts = self._extract_fonts(doc_mupdf)
            bookmarks = self._extract_bookmarks(doc_mupdf)

            pages_to_process = self._parse_page_range(self.pipeline_cfg.pages, doc_mupdf.page_count)
            pages: List[PageInfo] = []

            for page_num in tqdm(pages_to_process, desc="Extracting pages", unit="pg"):
                pg_info = self._extract_page(page_num, doc_mupdf, doc_plumber, doc_pdfium, str(pdf_path))
                pages.append(pg_info)
                logger.info(
                    f"[Extractor] p{page_num+1}: text_blocks={len(pg_info.text_blocks)} "
                    f"image_blocks={len(pg_info.image_blocks)} table_blocks={len(pg_info.table_blocks)} "
                    f"text_density={pg_info.text_density:.4f} image_area={pg_info.image_area:.4f} type={pg_info.page_type}"
                )

            total_native_chars = sum(
                sum(len(b.text) for b in p.text_blocks if b.source == "native")
                for p in pages
            )
            is_scanned = total_native_chars < 50

            page_count = doc_mupdf.page_count

            result = ExtractedDocument(
                source_path=str(pdf_path),
                page_count=page_count,
                pages=pages,
                fonts=fonts,
                bookmarks=bookmarks,
                metadata=meta,
                is_scanned=is_scanned,
            )

            logger.success(f"[Extractor] Done: {len(pages)} pages, {'SCANNED' if is_scanned else 'NATIVE TEXT'} document")
            return result

        finally:
            try:
                doc_mupdf.close()
            except Exception:
                logger.debug("Failed to close MuPDF document cleanly.")
            try:
                doc_plumber.close()
            except Exception:
                logger.debug("Failed to close pdfplumber document cleanly.")
            try:
                close_fn = getattr(doc_pdfium, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                logger.debug("Failed to close pdfium document cleanly.")

    # ── Per-page extraction ───────────────────────────────────

    def _extract_page(
        self,
        page_num: int,
        doc_mupdf: fitz.Document,
        doc_plumber: pdfplumber.PDF,
        doc_pdfium: pdfium.PdfDocument,
        pdf_path: str,
    ) -> PageInfo:
        mupdf_page = doc_mupdf[page_num]
        plumb_page = doc_plumber.pages[page_num]
        rect = mupdf_page.rect

        pg = PageInfo(
            page_num=page_num,
            width=rect.width,
            height=rect.height,
            rotation=mupdf_page.rotation,
        )

        # Render page (PDFium for fidelity)
        rendered = self._render_page(doc_pdfium, page_num)
        try:
            h, w = rendered.shape[:2]
            if h * w > self.MAX_RENDERED_PIXELS:
                scale = (self.MAX_RENDERED_PIXELS / (h * w)) ** 0.5
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                pil = Image.fromarray(rendered)
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                rendered = np.array(pil)
        except Exception:
            pass
        pg.rendered_image = rendered

        # Text
        pg.text_blocks = self._extract_text_blocks(mupdf_page, page_num)
        pg.has_text = bool(pg.text_blocks)

        # Images
        pg.image_blocks = self._extract_images(mupdf_page, doc_mupdf, page_num)
        pg.has_images = bool(pg.image_blocks)

        # Tables
        pg.table_blocks = self._extract_tables(plumb_page, pdf_path, page_num, rect)
        pg.has_tables = bool(pg.table_blocks)

        # Metrics & classification
        pg.text_density = self._compute_text_density(pg.text_blocks, rect)
        pg.image_area = self._compute_image_area(pg.image_blocks, rect)
        pg.page_type = self._classify_page(pg)

        return pg

    # ── Text extraction ───────────────────────────────────────

    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        blocks: List[TextBlock] = []

        if getattr(self.cfg, "text_extract_mode", "full") == "fast":
            raw = page.get_text("text")
            if raw.strip():
                blocks.append(TextBlock(
                    text=raw,
                    bbox=BBox(0, 0, page.rect.width, page.rect.height),
                    page_num=page_num,
                    source="native",
                ))
            return blocks

        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        for b_idx, block in enumerate(raw_blocks):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    r = span.get("bbox", (0, 0, page.rect.width, page.rect.height))
                    color_int = span.get("color", 0)
                    color_rgb = (
                        ((color_int >> 16) & 0xFF) / 255.0,
                        ((color_int >> 8) & 0xFF) / 255.0,
                        (color_int & 0xFF) / 255.0,
                    )
                    font_name = span.get("font", "")
                    font_flags = span.get("flags", 0)
                    blocks.append(TextBlock(
                        text=text,
                        bbox=BBox(*r),
                        font_name=font_name,
                        font_size=round(span.get("size", 12.0), 2),
                        color=color_rgb,
                        bold=bool(font_flags & 2**4),
                        italic=bool(font_flags & 2**1),
                        page_num=page_num,
                        block_id=b_idx,
                        source="native",
                    ))
        return blocks

    # ── Image extraction ──────────────────────────────────────

    def _extract_images(self, page: fitz.Page, doc: fitz.Document, page_num: int) -> List[ImageBlock]:
        blocks: List[ImageBlock] = []
        seen_xrefs = set()

        # Extract images referenced by the page
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base = {}
            try:
                base = doc.extract_image(xref) or {}
            except Exception as e:
                logger.debug(f"  Could not extract image xref={xref}: {e}")
                base = {}

            img_bytes = base.get("image")
            # If no raw bytes, attempt to render the image bbox from the page as a fallback
            if not img_bytes:
                try:
                    bbox = self._get_image_bbox(page, xref)
                    if bbox:
                        # Render the bbox region using MuPDF at extractor render DPI
                        mat = fitz.Matrix(getattr(self.cfg, "render_dpi", 300) / 72.0,
                                          getattr(self.cfg, "render_dpi", 300) / 72.0)
                        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1))
                        if pix and pix.samples:
                            mode = "RGB" if pix.n < 4 else "RGBA"
                            pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                            if pil.mode != "RGB":
                                pil = pil.convert("RGB")
                            buf = io.BytesIO()
                            pil.save(buf, format="PNG")
                            img_bytes = buf.getvalue()
                except Exception as e:
                    logger.debug(f"  Fallback render for xref={xref} failed: {e}")
                    img_bytes = None

            if not img_bytes:
                # still nothing — skip this xref but log it
                logger.debug(f"  No image bytes for xref={xref}; skipping")
                continue

            # Defensive open to get width/height
            try:
                img_pil = Image.open(io.BytesIO(img_bytes))
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")
                w, h = img_pil.size
            except Exception as e:
                logger.debug(f"  Failed to open image bytes for xref={xref}: {e}")
                continue

            # Skip extremely tiny icons (but keep small diagrams)
            min_dim = getattr(self.cfg, "min_image_dim", 24)
            if w < min_dim and h < min_dim:
                if getattr(self.cfg, "skip_tiny_images", False):
                    continue

            # Compute effective DPI using page size in inches
            page_w_in = page.rect.width / 72.0 if page.rect.width else 1.0
            page_h_in = page.rect.height / 72.0 if page.rect.height else 1.0
            dpi_x = w / page_w_in if page_w_in > 0 else 72.0
            dpi_y = h / page_h_in if page_h_in > 0 else 72.0

            # Get bbox for this image if available
            bbox = self._get_image_bbox(page, xref) or BBox(0, 0, page.rect.width, page.rect.height)

            # Heuristic: classify as photo vs diagram by color variance and palette
            try:
                arr = np.array(img_pil)
                var = float(np.var(arr.astype(np.float32)))
                unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))
                is_photo = (var > 500.0) or (unique_colors > 256)
            except Exception:
                is_photo = False

            blocks.append(ImageBlock(
                image_bytes=img_bytes,
                bbox=bbox,
                page_num=page_num,
                xref=xref,
                width=w,
                height=h,
                dpi_x=round(dpi_x, 1),
                dpi_y=round(dpi_y, 1),
                colorspace=str(base.get("colorspace", "")) or "RGB",
                ext=base.get("ext", "png"),
                is_photo=is_photo,
            ))

        # Additionally, detect images that are not embedded but rendered (vector drawings)
        # by scanning the rendered page for large connected components that look like images.
        if getattr(self.cfg, "detect_rendered_images", True):
            try:
                rendered = self._render_page_from_fitz(page)
                if rendered is not None:
                    gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
                    _, th = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        page_area = int(page.rect.width * page.rect.height)
                        if area > max(5000, 0.005 * page_area):
                            # Map pixel bbox back to PDF coordinates approximately
                            rh, rw = rendered.shape[:2]
                            pdf_x0 = (x / rw) * page.rect.width
                            pdf_y0 = (y / rh) * page.rect.height
                            pdf_x1 = ((x + w) / rw) * page.rect.width
                            pdf_y1 = ((y + h) / rh) * page.rect.height
                            bbox = BBox(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                            crop = rendered[y:y + h, x:x + w]
                            try:
                                pil = Image.fromarray(crop)
                                buf = io.BytesIO()
                                pil.save(buf, format="PNG")
                                img_bytes = buf.getvalue()
                                blocks.append(ImageBlock(
                                    image_bytes=img_bytes,
                                    bbox=bbox,
                                    page_num=page_num,
                                    xref=-1,
                                    width=w,
                                    height=h,
                                    dpi_x=round(getattr(self.cfg, "render_dpi", 300), 1),
                                    dpi_y=round(getattr(self.cfg, "render_dpi", 300), 1),
                                    colorspace="RGB",
                                    ext="png",
                                    is_photo=False,
                                ))
                            except Exception:
                                pass
            except Exception:
                pass

        return blocks

    def _get_image_bbox(self, page: fitz.Page, xref: int) -> Optional[BBox]:
        """Try to find bbox for a given image xref on the page."""
        try:
            for item in page.get_image_info():
                if item.get("xref") == xref:
                    r = item.get("bbox", (0, 0, page.rect.width, page.rect.height))
                    return BBox(*r)
        except Exception:
            pass
        return None

    # ── Table extraction ──────────────────────────────────────

    def _extract_tables(self, plumb_page: pdfplumber.page.Page, pdf_path: str, page_num: int, rect: fitz.Rect) -> List[TableBlock]:
        tables: List[TableBlock] = []
        method = getattr(self.cfg, "table_extraction_method", "all")

        # pdfplumber
        if method in ("pdfplumber", "all"):
            try:
                raw_tables = plumb_page.extract_tables({
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 3,
                    "intersection_tolerance": 15,
                })
                for t in raw_tables:
                    if t and len(t) > 0:
                        tables.append(TableBlock(
                            data=[[str(c or "") for c in row] for row in t],
                            bbox=BBox(0, 0, rect.width, rect.height),
                            page_num=page_num,
                            method="pdfplumber",
                        ))
            except Exception as e:
                logger.debug(f"  pdfplumber table extraction failed p{page_num}: {e}")

        # Camelot (if available)
        if method in ("camelot", "all") and CAMELOT_AVAILABLE:
            try:
                flavor = getattr(self.cfg, "camelot_flavor", "lattice")
                ctables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor=flavor, suppress_stdout=True)
                for ct in ctables:
                    try:
                        df = ct.df
                        x1, y1, x2, y2 = ct._bbox
                        tables.append(TableBlock(
                            data=df.values.tolist(),
                            bbox=BBox(x1, y1, x2, y2),
                            page_num=page_num,
                            method=f"camelot_{flavor}",
                            accuracy=ct.accuracy / 100.0 if hasattr(ct, "accuracy") else 1.0,
                            dataframe=df,
                        ))
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"  camelot failed p{page_num}: {e}")

        return self._dedup_tables(tables)

    def _dedup_tables(self, tables: List[TableBlock]) -> List[TableBlock]:
        seen = set()
        unique: List[TableBlock] = []
        for t in sorted(tables, key=lambda x: x.accuracy, reverse=True):
            key = hashlib.md5(json.dumps(t.data, sort_keys=True).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    # ── Page rendering ───────────────────────────────────────

    def _render_page(self, doc: pdfium.PdfDocument, page_num: int) -> np.ndarray:
        """Render page to numpy array using PDFium (highest fidelity)."""
        page = doc[page_num]
        scale = getattr(self.cfg, "render_dpi", 300) / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img)

    def _render_page_from_fitz(self, page: fitz.Page) -> Optional[np.ndarray]:
        """Fallback render using MuPDF if needed for rendered-image heuristics."""
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(getattr(self.cfg, "render_dpi", 300) / 72.0,
                                                     getattr(self.cfg, "render_dpi", 300) / 72.0))
            mode = "RGB" if pix.n < 4 else "RGBA"
            pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            return np.array(pil)
        except Exception:
            return None

    # ── Classification ───────────────────────────────────────

    def _classify_page(self, pg: PageInfo) -> str:
        # Use relaxed thresholds to detect diagrams
        td = pg.text_density
        ia = pg.image_area
        has_tables = pg.has_tables

        if td > 0.15 and ia < 0.1:
            return "text"
        if ia > 0.5 and td < 0.05:
            return "image"
        # If there is moderate image area and not many tables, treat as diagram
        if ia > 0.15 and not has_tables:
            return "diagram"
        # If many small images and text, mixed
        return "mixed"

    def _compute_text_density(self, blocks: List[TextBlock], rect: fitz.Rect) -> float:
        try:
            if rect.width * rect.height == 0:
                return 0.0
            total_chars = sum(len(b.text) for b in blocks)
            return total_chars / (rect.width * rect.height)
        except Exception:
            return 0.0

    def _compute_image_area(self, blocks: List[ImageBlock], rect: fitz.Rect) -> float:
        try:
            if rect.width * rect.height == 0:
                return 0.0
            total_area = sum(b.bbox.area for b in blocks)
            return min(total_area / (rect.width * rect.height), 1.0)
        except Exception:
            return 0.0

    # ── Document-level metadata ─────────────────────────────

    def _extract_metadata(self, doc: fitz.Document) -> dict:
        try:
            meta = dict(doc.metadata or {})
        except Exception:
            meta = {}
        meta["page_count"] = getattr(doc, "page_count", 0)
        meta["is_pdf"] = getattr(doc, "is_pdf", True)
        meta["needs_pass"] = getattr(doc, "needs_pass", False)
        return meta

    def _extract_fonts(self, doc: fitz.Document) -> dict:
        fonts = {}
        try:
            for page in doc:
                try:
                    for font in page.get_fonts(full=True):
                        name = font[3] if len(font) > 3 else None
                        if name and name not in fonts:
                            fonts[name] = {
                                "xref": font[0] if len(font) > 0 else None,
                                "type": font[1] if len(font) > 1 else None,
                                "encoding": font[2] if len(font) > 2 else None,
                                "name": name,
                            }
                except Exception:
                    continue
        except Exception:
            pass
        return fonts

    def _extract_bookmarks(self, doc: fitz.Document) -> List:
        try:
            toc = doc.get_toc()
            return toc if toc else []
        except Exception:
            return []

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _parse_page_range(pages_str: Optional[str], total: int) -> List[int]:
        if not pages_str:
            return list(range(total))
        result = set()
        for part in pages_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    start = max(1, int(a))
                    end = min(int(b), total)
                    if start <= end:
                        result.update(range(start - 1, end))
                except Exception:
                    continue
            else:
                try:
                    idx = int(part)
                    result.add(idx - 1)
                except Exception:
                    continue
        return sorted(p for p in result if 0 <= p < total)
