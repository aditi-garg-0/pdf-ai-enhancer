# extractor.py
"""
Stage 1 of the pipeline.
Extracts EVERYTHING from a PDF with maximum fidelity:
  - Text blocks with precise coordinates, fonts, sizes, colors
  - Embedded images (raw bytes, position, resolution)
  - Tables (via camelot + pdfplumber with fusion)
  - Fonts (embedded subsets, metrics)
  - Annotations, links, bookmarks
  - Page geometry (size, rotation, crop boxes)
  - Page classification (text-heavy / image-heavy / mixed / diagram)
"""

from __future__ import annotations

import io
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz                     # PyMuPDF
import pdfplumber
import pypdfium2 as pdfium
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm
import warnings

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

from config import Config, ExtractionConfig

# Safety: avoid PIL DecompressionBomb spam; still keep a reasonable cap
# If you trust sources, you can increase or disable this.
Image.MAX_IMAGE_PIXELS = 200_000_000  # 200M pixels cap


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class BBox:
    x0: float; y0: float; x1: float; y1: float

    @property
    def width(self): return self.x1 - self.x0

    @property
    def height(self): return self.y1 - self.y0

    @property
    def area(self): return self.width * self.height

    def to_tuple(self): return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TextBlock:
    text:      str
    bbox:      BBox
    font_name: str     = ""
    font_size: float   = 12.0
    color:     tuple   = (0, 0, 0)
    bold:      bool    = False
    italic:    bool    = False
    page_num:  int     = 0
    block_id:  int     = 0
    confidence: float  = 1.0          # 1.0 for native text; <1.0 for OCR
    source:    str     = "native"     # "native" | "ocr"


@dataclass
class ImageBlock:
    image_bytes: bytes
    bbox:        BBox
    page_num:    int
    xref:        int               # PyMuPDF internal reference
    width:       int   = 0
    height:      int   = 0
    dpi_x:       float = 72.0
    dpi_y:       float = 72.0
    colorspace:  str   = "RGB"
    ext:         str   = "png"
    sha256:      str   = ""        # Dedup key
    is_diagram:  bool  = False     # Classified by DiagramAnalyzer later
    is_photo:    bool  = False

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.image_bytes).hexdigest()

    def to_pil(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_pil())


@dataclass
class TableBlock:
    data:       list[list[str]]   # rows × cols
    bbox:       BBox
    page_num:   int
    method:     str               # "camelot_lattice" | "camelot_stream" | "pdfplumber"
    accuracy:   float = 1.0
    dataframe:  object = None     # pandas DataFrame (optional)


@dataclass
class PageInfo:
    page_num:     int
    width:        float
    height:       float
    rotation:     int              # 0 | 90 | 180 | 270
    has_text:     bool   = False
    has_images:   bool   = False
    has_tables:   bool   = False
    page_type:    str    = "mixed" # "text" | "image" | "diagram" | "mixed"
    text_density: float  = 0.0
    image_area:   float  = 0.0
    text_blocks:  list[TextBlock]  = field(default_factory=list)
    image_blocks: list[ImageBlock] = field(default_factory=list)
    table_blocks: list[TableBlock] = field(default_factory=list)
    rendered_image: Optional[np.ndarray] = None  # Full-page rasterized (may be downscaled)


@dataclass
class ExtractedDocument:
    source_path:  str
    page_count:   int
    pages:        list[PageInfo] = field(default_factory=list)
    fonts:        dict           = field(default_factory=dict)   # name → metrics
    bookmarks:    list           = field(default_factory=list)
    metadata:     dict           = field(default_factory=dict)
    is_scanned:   bool           = False   # True if mostly raster, no native text


# ──────────────────────────────────────────────────────────────
#  Extractor
# ──────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Full-spectrum PDF extractor.
    Produces an ExtractedDocument with every piece of content
    needed for downstream enhancement stages.
    """

    # Maximum pixels for stored rendered_image to avoid huge pickles
    MAX_RENDERED_PIXELS = 8_000_000  # ~8MP default (adjustable via config if needed)

    def __init__(self, config: Config):
        self.cfg = config.extraction
        self.pipeline_cfg = config.pipeline

    # ── Public entry point ───────────────────────────────────

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        logger.info(f"[Extractor] Opening: {pdf_path.name}")

        doc_mupdf     = fitz.open(str(pdf_path))
        doc_plumber   = pdfplumber.open(str(pdf_path))
        doc_pdfium    = pdfium.PdfDocument(str(pdf_path))

        try:
            meta = self._extract_metadata(doc_mupdf)
            fonts = self._extract_fonts(doc_mupdf)
            bookmarks = self._extract_bookmarks(doc_mupdf)

            pages_to_process = self._parse_page_range(
                self.pipeline_cfg.pages, doc_mupdf.page_count
            )

            pages: list[PageInfo] = []

            for page_num in tqdm(pages_to_process, desc="Extracting pages", unit="pg"):
                pg_info = self._extract_page(
                    page_num, doc_mupdf, doc_plumber, doc_pdfium, str(pdf_path)
                )
                pages.append(pg_info)

            # Detect if the whole document is scanned (no native text anywhere)
            total_native_chars = sum(
                sum(len(b.text) for b in p.text_blocks if b.source == "native")
                for p in pages
            )
            is_scanned = total_native_chars < 50

            # Capture page_count while document is still open
            page_count = doc_mupdf.page_count

            result = ExtractedDocument(
                source_path  = str(pdf_path),
                page_count   = page_count,
                pages        = pages,
                fonts        = fonts,
                bookmarks    = bookmarks,
                metadata     = meta,
                is_scanned   = is_scanned,
            )

            logger.success(
                f"[Extractor] Done: {len(pages)} pages, "
                f"{'SCANNED' if is_scanned else 'NATIVE TEXT'} document"
            )
            return result

        finally:
            # Ensure resources are always closed, but only after we've read needed attributes
            try:
                doc_mupdf.close()
            except Exception:
                logger.debug("Failed to close MuPDF document cleanly.")
            try:
                doc_plumber.close()
            except Exception:
                logger.debug("Failed to close pdfplumber document cleanly.")
            # pdfium PdfDocument may not require explicit close, but attempt if available
            try:
                close_fn = getattr(doc_pdfium, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                logger.debug("Failed to close pdfium document cleanly.")

    # ── Per-page extraction ───────────────────────────────────

    def _extract_page(
        self,
        page_num:   int,
        doc_mupdf:  fitz.Document,
        doc_plumber: pdfplumber.PDF,
        doc_pdfium: pdfium.PdfDocument,
        pdf_path:   str,
    ) -> PageInfo:

        mupdf_page  = doc_mupdf[page_num]
        plumb_page  = doc_plumber.pages[page_num]
        rect        = mupdf_page.rect

        pg = PageInfo(
            page_num = page_num,
            width    = rect.width,
            height   = rect.height,
            rotation = mupdf_page.rotation,
        )

        # ── Rasterize full page ──────────────────────────────
        rendered = self._render_page(doc_pdfium, page_num)
        # Downscale rendered image if too large to keep memory/pickle sizes reasonable
        try:
            h, w = rendered.shape[:2]
            if h * w > self.MAX_RENDERED_PIXELS:
                scale = (self.MAX_RENDERED_PIXELS / (h * w)) ** 0.5
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                pil = Image.fromarray(rendered)
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                rendered = np.array(pil)
        except Exception:
            # If anything goes wrong, keep the original rendered image
            pass
        pg.rendered_image = rendered

        # ── Text extraction ──────────────────────────────────
        pg.text_blocks = self._extract_text_blocks(mupdf_page, page_num)
        pg.has_text = bool(pg.text_blocks)

        # ── Image extraction ─────────────────────────────────
        pg.image_blocks = self._extract_images(mupdf_page, doc_mupdf, page_num)
        pg.has_images = bool(pg.image_blocks)

        # ── Table extraction ─────────────────────────────────
        pg.table_blocks = self._extract_tables(
            plumb_page, pdf_path, page_num, rect
        )
        pg.has_tables = bool(pg.table_blocks)

        # ── Page classification ───────────────────────────────
        pg.text_density = self._compute_text_density(pg.text_blocks, rect)
        pg.image_area   = self._compute_image_area(pg.image_blocks, rect)
        pg.page_type    = self._classify_page(pg)

        return pg

    # ── Text extraction ───────────────────────────────────────

    def _extract_text_blocks(
        self, page: fitz.Page, page_num: int
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []

        if self.cfg.text_extract_mode == "fast":
            # pypdf-style: raw text, no coords
            raw = page.get_text("text")
            if raw.strip():
                blocks.append(TextBlock(
                    text=raw, bbox=BBox(0, 0, page.rect.width, page.rect.height),
                    page_num=page_num
                ))
            return blocks

        # Full dict extraction: every span with font metadata
        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for b_idx, block in enumerate(raw_blocks):
            if block.get("type") != 0:
                continue  # Skip image-type blocks

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    r = span["bbox"]
                    color_int = span.get("color", 0)
                    color_rgb = (
                        (color_int >> 16 & 0xFF) / 255,
                        (color_int >> 8  & 0xFF) / 255,
                        (color_int       & 0xFF) / 255,
                    )
                    font_name  = span.get("font", "")
                    font_flags = span.get("flags", 0)

                    blocks.append(TextBlock(
                        text      = text,
                        bbox      = BBox(*r),
                        font_name = font_name,
                        font_size = round(span.get("size", 12.0), 2),
                        color     = color_rgb,
                        bold      = bool(font_flags & 2**4),
                        italic    = bool(font_flags & 2**1),
                        page_num  = page_num,
                        block_id  = b_idx,
                        source    = "native",
                    ))

        return blocks

    # ── Image extraction ──────────────────────────────────────

    def _extract_images(
        self, page: fitz.Page, doc: fitz.Document, page_num: int
    ) -> list[ImageBlock]:
        blocks: list[ImageBlock] = []
        seen_xrefs: set[int] = set()

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base = doc.extract_image(xref)
            except Exception as e:
                logger.warning(f"  Failed to extract image xref={xref}: {e}")
                continue

            img_bytes = base.get("image")
            if not img_bytes:
                continue

            # Get bounding box from page image list
            bbox = self._get_image_bbox(page, xref)

            try:
                # Defensive: open image and downscale if extremely large
                img_pil = Image.open(io.BytesIO(img_bytes))
                w, h = img_pil.size

                # If image is enormous, downscale to a safer size for storage/processing
                max_pixels = 50_000_000  # 50MP
                if w * h > max_pixels:
                    scale = (max_pixels / (w * h)) ** 0.5
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
                    buf = io.BytesIO()
                    img_pil.save(buf, format=base.get("ext", "PNG"))
                    img_bytes = buf.getvalue()
                    w, h = img_pil.size

            except Exception as e:
                logger.debug(f"  Could not open image xref={xref}: {e}")
                continue

            # Compute effective DPI
            page_w_in = page.rect.width / 72
            page_h_in = page.rect.height / 72
            dpi_x = w / page_w_in if page_w_in > 0 else 72
            dpi_y = h / page_h_in if page_h_in > 0 else 72

            # Skip tiny images (icons, bullets)
            if w < 30 or h < 30:
                continue

            cs_name = base.get("colorspace", "")
            if hasattr(cs_name, "name"):
                cs_name = cs_name.name

            blocks.append(ImageBlock(
                image_bytes = img_bytes,
                bbox        = bbox,
                page_num    = page_num,
                xref        = xref,
                width       = w,
                height      = h,
                dpi_x       = round(dpi_x, 1),
                dpi_y       = round(dpi_y, 1),
                colorspace  = str(cs_name),
                ext         = base.get("ext", "png"),
            ))

        return blocks

    def _get_image_bbox(self, page: fitz.Page, xref: int) -> BBox:
        """Try to find bbox for a given image xref on the page."""
        for item in page.get_image_info():
            if item.get("xref") == xref:
                r = item.get("bbox", (0, 0, page.rect.width, page.rect.height))
                return BBox(*r)
        return BBox(0, 0, page.rect.width, page.rect.height)

    # ── Table extraction ──────────────────────────────────────

    def _extract_tables(
        self,
        plumb_page: pdfplumber.page.Page,
        pdf_path: str,
        page_num: int,
        rect: fitz.Rect,
    ) -> list[TableBlock]:
        tables: list[TableBlock] = []
        method = self.cfg.table_extraction_method

        # ── pdfplumber ────────────────────────────────────────
        if method in ("pdfplumber", "all"):
            try:
                settings = {
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 3,
                    "intersection_tolerance": 15,
                }
                raw_tables = plumb_page.extract_tables(settings)
                for t in raw_tables:
                    if t and len(t) > 1:
                        tables.append(TableBlock(
                            data     = [[str(c or "") for c in row] for row in t],
                            bbox     = BBox(0, 0, rect.width, rect.height),
                            page_num = page_num,
                            method   = "pdfplumber",
                        ))
            except Exception as e:
                logger.debug(f"  pdfplumber table extraction failed p{page_num}: {e}")

        # ── Camelot (lattice — best for bordered tables) ──────
        if method in ("camelot", "all") and CAMELOT_AVAILABLE:
            try:
                flavor = self.cfg.camelot_flavor
                ctables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor=flavor,
                    suppress_stdout=True,
                )
                for ct in ctables:
                    if ct.accuracy > 50:
                        df   = ct.df
                        x1, y1, x2, y2 = ct._bbox
                        tables.append(TableBlock(
                            data      = df.values.tolist(),
                            bbox      = BBox(x1, y1, x2, y2),
                            page_num  = page_num,
                            method    = f"camelot_{flavor}",
                            accuracy  = ct.accuracy / 100,
                            dataframe = df,
                        ))
            except Exception as e:
                logger.debug(f"  camelot failed p{page_num}: {e}")

        # Deduplicate (if both methods found same table)
        return self._dedup_tables(tables)

    def _dedup_tables(self, tables: list[TableBlock]) -> list[TableBlock]:
        """Remove duplicate tables by comparing data hashes."""
        seen: set[str] = set()
        unique: list[TableBlock] = []
        for t in sorted(tables, key=lambda x: x.accuracy, reverse=True):
            key = hashlib.md5(json.dumps(t.data, sort_keys=True).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    # ── Page rendering ────────────────────────────────────────

    def _render_page(self, doc: pdfium.PdfDocument, page_num: int) -> np.ndarray:
        """Render page to numpy array using PDFium (highest fidelity)."""
        page = doc[page_num]
        scale = self.cfg.render_dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img)

    # ── Classification ────────────────────────────────────────

    def _classify_page(self, pg: PageInfo) -> str:
        if pg.text_density > 0.15 and pg.image_area < 0.1:
            return "text"
        if pg.image_area > 0.5 and pg.text_density < 0.05:
            return "image"
        if pg.image_area > 0.3 and not pg.has_tables:
            return "diagram"
        return "mixed"

    def _compute_text_density(
        self, blocks: list[TextBlock], rect: fitz.Rect
    ) -> float:
        if rect.width * rect.height == 0:
            return 0.0
        total_chars = sum(len(b.text) for b in blocks)
        return total_chars / (rect.width * rect.height)

    def _compute_image_area(
        self, blocks: list[ImageBlock], rect: fitz.Rect
    ) -> float:
        if rect.width * rect.height == 0:
            return 0.0
        total_area = sum(b.bbox.area for b in blocks)
        return min(total_area / (rect.width * rect.height), 1.0)

    # ── Document-level metadata ─────────────────────────────

    def _extract_metadata(self, doc: fitz.Document) -> dict:
        meta = dict(doc.metadata)
        meta["page_count"] = doc.page_count
        meta["is_pdf"]     = doc.is_pdf
        meta["needs_pass"] = doc.needs_pass
        return meta

    def _extract_fonts(self, doc: fitz.Document) -> dict:
        fonts = {}
        for page in doc:
            for font in page.get_fonts(full=True):
                name = font[3]
                if name and name not in fonts:
                    fonts[name] = {
                        "xref": font[0],
                        "type": font[1],
                        "encoding": font[2],
                        "name": name,
                    }
        return fonts

    def _extract_bookmarks(self, doc: fitz.Document) -> list:
        try:
            return doc.get_toc()  # [[level, title, page, dest], ...]
        except Exception:
            return []

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _parse_page_range(pages_str: Optional[str], total: int) -> list[int]:
        """Parse "1-5,8,10-15" → [0,1,2,3,4,7,9,10,11,12,13,14] (0-indexed)."""
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
