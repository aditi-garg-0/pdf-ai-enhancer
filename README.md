# PDF AI Enhancer

**AI pipeline for enhancing PDFs with text and diagrams.**

Processes every page through a 5-stage AI pipeline: extraction → image super-resolution → multi-engine OCR correction → diagram vectorization → PDF reconstruction.

---

## What It Does

| Feature | Technology |
|---|---|
| 4× image super-resolution | Real-ESRGAN (anime model for diagrams) |
| Diagram vectorization | potrace — raster → clean SVG |
| Multi-engine OCR | PaddleOCR + Tesseract + EasyOCR (ensemble) |
| AI OCR correction | Claude API (context-aware error fixing) |
| Diagram understanding | Claude Vision (labels, relationships, type) |
| Face/portrait restoration | GFPGAN |
| Table extraction | Camelot (lattice) + pdfplumber |
| Searchable text layer | Invisible OCR overlay in output PDF |
| Checkpoint/resume | SQLite — resume interrupted runs |
| API response cache | Never pay twice for the same image |
| Quality metrics | SSIM + PSNR per page, JSON report |

---

## Installation

### 1. System dependencies

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr poppler-utils ghostscript potrace \
                 libcairo2-dev libgl1-mesa-glx
```

**macOS:**
```bash
brew install tesseract poppler ghostscript potrace cairo
```

**Windows:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows
- Ghostscript: https://www.ghostscript.com/download.html
- potrace: https://potrace.sourceforge.net

### 2. Python dependencies
```bash
pip install -r requirements.txt
```

### 3. API key (for AI features)
```bash
export ANTHROPIC_API_KEY=your_key_here
# or add to .env file
```

---

## Quick Start

```bash
# Enhance a single PDF
python pipeline.py enhance my_document.pdf

# Specify output path
python pipeline.py enhance input.pdf -o output_enhanced.pdf

# Process all PDFs in a folder
python pipeline.py batch ./pdfs/ -o ./enhanced/

# Generate and customize config
python pipeline.py config --save config.yaml
# edit config.yaml...
python pipeline.py enhance input.pdf --config config.yaml
```

---

## CLI Options

### `enhance` command
```
python pipeline.py enhance INPUT_PDF [OPTIONS]

  -o, --output PATH       Output PDF path
  -c, --config PATH       Custom config.yaml
  --dpi INT               Output DPI (default: 300)
  --pages TEXT            Page range e.g. "1-5,8,10-15"
  --no-sr                 Disable AI super-resolution
  --no-ocr                Disable OCR correction
  --no-ai                 Disable all API calls (offline mode)
  --no-vectorize          Disable diagram vectorization
  --workers INT           Parallel worker count
  --api-key TEXT          Anthropic API key (or set ANTHROPIC_API_KEY)
```

### `batch` command
```
python pipeline.py batch INPUT_DIR [OPTIONS]

  -o, --output-dir PATH   Output directory
  -c, --config PATH       Custom config.yaml
  --pattern TEXT          Glob pattern (default: *.pdf)
```

---

## Configuration

Generate a full config file:
```bash
python pipeline.py config --save config.yaml
```

Key settings in `config.yaml`:

```yaml
enhancement:
  sr_model: RealESRGAN_x4plus_anime   # Best for diagrams
  sr_outscale: 4.0                    # 4x upscale
  enable_face_restoration: true

ocr:
  engine_priority: [paddleocr, tesseract, easyocr]
  tesseract_lang: eng                  # Change for other languages
  enable_ai_correction: true           # Claude OCR cleanup

diagram:
  enable_ai_analysis: true
  enable_vectorization: true
  embed_as_vector: true                # SVG in output PDF

pipeline:
  output_dpi: 300
  workers: 4
  enable_checkpoint: true              # Resume interrupted runs
```

---

## Project Structure

```
pdf_ai_enhancer/
├── requirements.txt      All Python dependencies
├── config.py             Typed config with Pydantic validation
├── extractor.py          Stage 1: Full PDF content extraction
├── image_enhancer.py     Stage 2: Real-ESRGAN + CLAHE + denoising
├── ocr_corrector.py      Stage 3: Multi-engine OCR + AI correction
├── diagram_analyzer.py   Stage 4: Claude Vision + vectorization
├── pdf_rebuilder.py      Stage 5: Reconstruct enhanced PDF
├── pipeline.py           Master orchestrator + CLI
├── checkpoint.py         SQLite resume/cache system
└── README.md             This file
```

---

## How Each Stage Works

### Stage 1 — Extraction (`extractor.py`)
Uses **PyMuPDF** (fitz) for full block-level extraction with font metadata, **pdfplumber** for table detection, and **PDFium** for high-fidelity page rendering. Every page is classified as `text`, `image`, `diagram`, or `mixed`.

### Stage 2 — Image Enhancement (`image_enhancer.py`)
Pipeline: denoise (Non-Local Means) → CLAHE → unsharp mask → **Real-ESRGAN 4×** super-resolution → GFPGAN face restoration → post-process. Falls back to bicubic if GPU unavailable.

### Stage 3 — OCR Correction (`ocr_corrector.py`)
Runs **PaddleOCR + Tesseract + EasyOCR** in priority order, ensembles results by confidence, then fixes: encoding artifacts (ftfy) → spelling (SymSpell) → grammar (LanguageTool) → AI context correction (Claude).

### Stage 4 — Diagram Analysis (`diagram_analyzer.py`)
**Claude Vision** classifies each diagram (flowchart/schematic/chart/etc), extracts labels and relationships as JSON. **potrace** traces the binarized image to a clean SVG. Optionally embeds as vector in the output.

### Stage 5 — Rebuild (`pdf_rebuilder.py`)
Assembles all enhanced components into a new PDF using **PyMuPDF**: enhanced images, SVG diagrams, invisible searchable text layer, reconstructed table grids, restored bookmarks. **pikepdf** linearizes the output for fast web viewing.

---

## Offline Mode

Run without any API keys (no Claude calls):
```bash
python pipeline.py enhance input.pdf --no-ai
```
Super-resolution, OCR, denoising, and vectorization still work fully offline.

---

## Troubleshooting

**`ModuleNotFoundError: realesrgan`**
```bash
pip install realesrgan basicsr
```

**`tesseract not found`**
Install Tesseract system binary (see Installation above).

**GPU out of memory during super-resolution**
Reduce tile size in config: `sr_tile: 256`
Or use CPU: `sr_gpu_id: null`

**`potrace not found`**
Install the potrace system binary. Vectorization will be skipped gracefully if missing.

**`ANTHROPIC_API_KEY not set`**
AI correction and diagram analysis are skipped. All other features work offline.

---

## Output Files

For each input `document.pdf`:
- `document_enhanced.pdf` — the enhanced PDF
- `document_enhanced.quality_report.json` — per-page metrics

The quality report contains images enhanced, diagrams vectorized, tables embedded, and OCR blocks corrected per page.