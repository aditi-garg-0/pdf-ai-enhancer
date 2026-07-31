# PDF AI Enhancer

**A five-stage AI pipeline for restoring and enhancing scanned engineering documents: super-resolution, OCR correction, diagram vectorization, and searchable-text reconstruction, backed by Claude with a Gemini fallback for diagram intelligence.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration Reference](#configuration-reference)
6. [Pipeline Internals](#pipeline-internals)
7. [Performance Characteristics](#performance-characteristics)
8. [Known Limitations and Troubleshooting](#known-limitations-and-troubleshooting)
9. [Engineering Notes: Fixes Applied](#engineering-notes-fixes-applied)
10. [Project Structure](#project-structure)
11. [Development](#development)
12. [License](#license)

---

## Overview

PDF AI Enhancer takes low quality scanned PDFs, engineering schematics, as built drawings, relay diagrams, contract documents, and produces a cleaned, super resolved, searchable output PDF. It's built for documents where the source scan quality is poor (generation loss photocopies, faded ink, uneven lighting, paper texture) but the content itself is dense and structurally important: technical line art, tabular data, small font annotations.

The pipeline combines classical image processing (denoising, CLAHE, adaptive thresholding, unsharp masking), deep learning super-resolution (Real-ESRGAN), multi-engine OCR ensembling, and large language model based semantic correction and diagram understanding (Claude, with a Gemini fallback for the free tier).

Three things matter more than anything else here, in this order:

1. **Fidelity.** Never fabricate content. Enhancement should sharpen and clean what's actually on the page, not invent what isn't.
2. **Legibility.** The real measure of success is whether a person can read the result, not a generic image quality score.
3. **Throughput.** Batch processing of large document sets needs to be workable on ordinary hardware, so every stage has a CPU only fallback path.

---

## Architecture

```
                    +-----------------------------------------------------------+
                    |                     pipeline.py (CLI/API)                |
                    +-----------------------------+-----------------------------+
                                                    |
                    +-----------------------------v-----------------------------+
                    |  Stage 1: Extraction (extractor.py)                      |
                    |  PyMuPDF block extraction, pdfplumber tables,            |
                    |  PDFium page rendering, text/image/diagram classify      |
                    +-----------------------------+-----------------------------+
                                                    |  ExtractedDocument
                    +-----------------------------v-----------------------------+
                    |  Stage 2: Image Enhancement (image_enhancer.py)          |
                    |  denoise, CLAHE, auto contrast stretch, unsharp,         |
                    |  selective Real-ESRGAN SR, whitening, vectorization      |
                    +-----------------------------+-----------------------------+
                                                    |
                    +-----------------------------v-----------------------------+
                    |  Stage 3: OCR Correction (ocr_corrector.py)              |
                    |  PaddleOCR + Tesseract + EasyOCR ensemble, ftfy,         |
                    |  SymSpell, LanguageTool, Claude/Gemini context fix       |
                    +-----------------------------+-----------------------------+
                                                    |
                    +-----------------------------v-----------------------------+
                    |  Stage 4: Diagram Analysis (diagram_analyzer.py)         |
                    |  Claude/Gemini Vision classification, potrace raster     |
                    |  to SVG vectorization, optional semantic reconstruction  |
                    +-----------------------------+-----------------------------+
                                                    |
                    +-----------------------------v-----------------------------+
                    |  Stage 5: Rebuild (pdf_rebuilder.py)                     |
                    |  Composite enhanced raster/vector content, invisible     |
                    |  searchable text layer, pikepdf linearization            |
                    +-----------------------------------------------------------+
```

Supporting infrastructure:

- **`checkpoint.py`**: a SQLite backed extraction cache, keyed on the source PDF, so iterative re-runs against the same document skip the (comparatively cheap) extraction step. It does not cache image enhancement or SR, which is genuine per-run compute and the dominant cost.
- **`config.py`**: Pydantic validated typed configuration, loaded from `config.yaml` with environment variable overrides for secrets.

---

## Installation

### 1. System dependencies

**Ubuntu / Debian:**
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

> **A compatibility note worth reading before you skip it:** `basicsr==1.4.2` imports `torchvision.transforms.functional_tensor`, which was removed in `torchvision>=0.17`. With the pinned `torchvision==0.18.1`, that import fails silently unless `image_enhancer.py`'s compatibility shim (it registers the module under its old, pre 0.17 location) runs before `basicsr` is imported. Don't remove that shim without re-testing, whatever `basicsr`/`torchvision` pair you end up on. The failure mode is silent: `REALESRGAN_AVAILABLE` becomes `False` and the pipeline quietly falls back to plain bicubic upscaling with no error raised anywhere.

### 3. API keys

At least one of the following is needed for AI assisted diagram analysis and OCR correction. Everything else, SR, denoising, vectorization, OCR itself, works fully offline.

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
# and/or, for the free tier fallback:
GEMINI_API_KEY=...
```

> **Please don't** put `anthropic_api_key` (or `gemini_api_key`) directly in `config.yaml`, not even as a placeholder. `Config`'s `default_factory` only reads the environment variable when the key is absent from the YAML entirely. A literal value in the YAML, including a placeholder like `"your_key_here"`, overrides the environment variable and gets sent as the actual bearer credential, which produces a quiet `401 Unauthorized` on every single API call. Keep secrets in `.env` only.

`ai_provider_priority: anthropic` (the default) tries Claude first and falls back to Gemini if no Claude key is configured or the Claude call fails. Set it to `gemini` if you'd rather use the free tier even when both keys are present.

---

## Quick Start

```bash
# Enhance a single PDF
python pipeline.py enhance my_document.pdf

# Specify an output path
python pipeline.py enhance input.pdf -o output_enhanced.pdf

# Batch process a folder
python pipeline.py batch ./pdfs/ -o ./enhanced/

# Generate a config file to customize
python pipeline.py config --save config.yaml
python pipeline.py enhance input.pdf --config config.yaml
```

### CLI reference

```
python pipeline.py enhance INPUT_PDF [OPTIONS]

  -o, --output PATH       Output PDF path
  -c, --config PATH       Custom config.yaml
  --dpi INT               Output DPI (default: 300)
  --pages TEXT            Page range, e.g. "1-5,8,10-15"
  --no-sr                 Disable AI super-resolution
  --no-ocr                Disable OCR correction
  --no-ai                 Disable all LLM API calls (fully offline)
  --no-vectorize          Disable diagram vectorization
  --workers INT           Parallel worker count
  --api-key TEXT          Anthropic API key (or set ANTHROPIC_API_KEY)

python pipeline.py batch INPUT_DIR [OPTIONS]

  -o, --output-dir PATH   Output directory
  -c, --config PATH       Custom config.yaml
  --pattern TEXT          Glob pattern (default: *.pdf)
```

---

## Configuration Reference

Configuration is layered: Pydantic defaults in `config.py`, then `config.yaml` overrides, then environment variables for secrets. Every field is typed and validated at load time.

### `api`

| Key | Default | Notes |
|---|---|---|
| `ai_provider_priority` | `anthropic` | `anthropic` or `gemini`. This sets fallback order, not exclusivity. |
| `claude_model` | `claude-opus-4-5` | Used for diagram Vision analysis and OCR context correction. |
| `claude_fast` | `claude-sonnet-4-20250514` | Cheaper, faster calls where full Opus quality isn't needed. |
| `gemini_model` / `gemini_model_fast` | `gemini-2.5-flash` / `gemini-2.5-flash-lite` | Free tier fallback. |
| `max_concurrent_api_calls` | `3` | Rate limit guard. |
| `api_retry_attempts` | `5` | Exponential backoff between `api_retry_min_wait` and `api_retry_max_wait`. |

### `extraction`

| Key | Default | Notes |
|---|---|---|
| `render_dpi` | `400` | A higher DPI gives the SR pass a better starting point. |
| `text_extract_mode` | `full` | `fast` (pypdf), `layout` (pdfplumber with coordinates), or `full` (PyMuPDF dict blocks, most detail). |
| `table_extraction_method` | `all` | `camelot`, `pdfplumber`, `paddleocr`, or `all` for an ensemble. |

### `enhancement`

This is the largest section and the one that matters most day to day. Grouped by what each part actually does:

**Super-resolution**
| Key | Default | Notes |
|---|---|---|
| `enable_super_resolution` | `true` | |
| `sr_model` | `RealESRGAN_x4plus` | A general purpose model. Avoid the `_anime` variant for scanned technical drawings; it carries a cel shading, stylization bias that shows up as hallucinated directional shading on plain line art. |
| `sr_outscale` | `4.0` | |
| `selective_sr` | `true` | Tile grid based: only GAN upscales tiles that actually contain ink, and gives blank tiles a cheap bicubic resize instead. See [Performance Characteristics](#performance-characteristics). |
| `selective_sr_tile_px` | `0` | `0` means match `sr_tile`. Please don't set this within about 32px of `sr_tile`. The padded crop handed to the model needs to stay at or under the model's own internal tile threshold, otherwise `RealESRGANer` re-tiles an already minimal crop a second time, which is pure overhead with no quality benefit. |

**Vectorization**
| Key | Default | Notes |
|---|---|---|
| `enable_vectorization` | `true` | potrace raster to SVG, for line art blocks. |
| `max_vectorize_dim_px` | `2500` | Blocks bigger than this get skipped. Vectorization is for logos and diagram crops, not full scanned pages. |

**Whitening (background cleanup)**
| Key | Default | Notes |
|---|---|---|
| `whiten_block_size` | `15` | The adaptiveThreshold window. Bigger is not more aggressive here, which is counterintuitive; see the engineering notes below. |
| `whiten_c` | `9.0` | The base adaptiveThreshold `C` offset. |
| `whiten_c_resolution_scale` | `1.0` | Grows `C` (dampened by a square root, and capped at `whiten_c_max`) as resolution climbs past a roughly 900px reference, so a 4x SR'd page gets a wider correction net than a native resolution crop. |
| `whiten_c_max` | `14.0` | A hard ceiling regardless of resolution. |
| `debug_whitening` | `false` | Dumps an amplified (8x) before/after diff per block, so you can see exactly what the pass is correcting. |

**Contrast**
| Key | Default | Notes |
|---|---|---|
| `enable_auto_contrast` | `true` | A black point/white point percentile stretch, run before CLAHE and sharpening. This is what fixes genuinely faded source ink, for example a typewriter era scan where "black" text sits around L 140 to 180. Whitening only ever lightens the background; it has no way to darken existing ink. |
| `auto_contrast_gamma` | `1.8` | Applied after the linear stretch, pushes midtones darker without moving true black or true white. Raise it toward 2.2 to 2.5 if text still reads light; lower it toward 1.4 if thin curved lines start looking jagged. |

### `ocr`

| Key | Default | Notes |
|---|---|---|
| `engine_priority` | `[paddleocr, tesseract, easyocr]` | Tried in order, then ensembled by `ensemble_strategy` (`voting`, `confidence`, or `longest`). |
| `enable_ai_correction` | `true` | Sends the OCR text to Claude or Gemini for context aware correction. |

### `pipeline`

| Key | Default | Notes |
|---|---|---|
| `enable_checkpoint` | `true` | The SQLite extraction cache. |
| `output_dpi` | `400` | |
| `workers` | `min(4, cpu_count)` | |

---

## Pipeline Internals

### Stage 1: Extraction

PyMuPDF does block level extraction with font metadata; pdfplumber handles table detection; PDFium renders high fidelity page rasters for fallback and background use. Each page gets classified as `text`, `image`, `diagram`, or `mixed`, based on text density and image area fraction. Purely scanned pages, ones without a native text layer, are flagged `is_scanned` and handled differently during reconstruction. More on that in Stage 5.

### Stage 2: Image Enhancement

The per block pipeline (`enhance_block`) runs: remove marks, then preprocess (denoise, CLAHE, auto contrast, sharpen, and whitening where it applies), then selective super-resolution, then vectorization where it applies, then postprocess (a final sharpen, a final whitening pass, and saturation/brightness tuning).

There's a separate, lighter path (`enhance_page_render`) for whole page background rendering on scanned documents that don't decompose into individual image blocks. These two paths should never both run against the same content. See the engineering notes below for why that matters.

### Stage 3: OCR Correction

Three OCR engines run independently and get ensembled by confidence or plurality vote. The cleanup chain after that is: ftfy for Unicode and encoding artifacts, SymSpell for fast dictionary based spell checking, LanguageTool for grammar, and then an LLM pass for context correction, one that understands "K1.41", "A25", "PMU" and similar are legitimate schematic reference designators, not spelling mistakes.

### Stage 4: Diagram Analysis

A Vision capable LLM classifies each diagram block (flowchart, schematic, graph, table, formula, map, photo, or logo), pulls out visible labels and inferred relationships as structured JSON, and estimates a confidence score. Separately, potrace binarizes and traces the same block into a clean SVG, for optional vector embedding via `diagram.embed_as_vector`.

### Stage 5: Rebuild

This assembles the output PDF page by page. For a scanned page, it picks one source of truth, either a full page background render or the set of individually enhanced extracted image blocks, and never both. Mixing them risks a double exposure, described below. It then embeds an invisible (PDF text rendering mode 3) searchable text layer over the visible content, and finally linearizes the file with pikepdf for fast web viewing.

---

## Performance Characteristics

On CPU, full page Real-ESRGAN inference is the dominant cost by a wide margin. `selective_sr` helps here: it tiles each block into a grid, checks ink density per tile, and only sends tiles that actually have content through the GAN model, giving blank margin or background tiles a cheap bicubic resize instead. On a typical scanned engineering drawing, sparse line art on a mostly white background, this meaningfully cuts down how many tiles hit the slow path.

Extraction results are cached (`checkpoint.py`), keyed on the source PDF, so iterating on enhancement settings against the same input doesn't re-pay the extraction cost every run. Vision and OCR correction API responses are cached the same way, so you don't get billed twice for identical calls across runs.

A GPU is strongly recommended for anything beyond light, occasional use. CPU inference works, but it's roughly one to two orders of magnitude slower per tile.

---

## Known Limitations and Troubleshooting

| Symptom | Likely Cause | Where to Look |
|---|---|---|
| `Real-ESRGAN not available, using bicubic fallback` | The `basicsr`/`torchvision` compatibility shim didn't load, or a dependency is missing | See the compatibility note under Installation |
| `401 Unauthorized` on every LLM call | A placeholder or literal API key sitting in `config.yaml` | Remove it from the YAML, set it in `.env` instead |
| `potrace not found` | The system binary is missing | Vectorization skips gracefully; install it per the platform instructions above |
| Very slow full page runs | `selective_sr` is disabled, or `selective_sr_tile_px` is set too close to `sr_tile` | Check the config; see the double tiling note below |
| Blank output pages | A background/per block source of truth decision predicted content coverage that didn't actually materialize at draw time | Check the logs for the emergency fallback warning in `pdf_rebuilder.py` |
| A grey halo around content edges | Could be a few things: SR/vectorization ordering, whitening gate scope, or whitening intensity | See the engineering notes below |
| Content looks washed out, or thin lines are missing | Whitening applied too broadly or too strongly for how dense the document actually is | Try lowering `whiten_c_max`; see the engineering notes below |

---

## Engineering Notes: Fixes Applied

This section is here on purpose. It documents non-obvious bugs that came up while hardening this pipeline against real scanned engineering documents, because every one of these is easy to quietly reintroduce if a future change doesn't know the history.

1. **Running SR after vectorization softened things back up.** Applying GAN super-resolution to a block that had already been cleanly vectorized (pixel pure, by construction) reintroduced soft, grey edges, because a photographic upscaler has no concept of "this edge is supposed to stay a hard binary transition." Fixed by skipping the GAN pass for any block that vectorized successfully on that call.

2. **`selective_sr` was tiling twice.** The outer tile grid and the cached `RealESRGANer` instance underneath it were both tiling independently, because the outer grid's padded crop size ended up bigger than the model's own internal tile threshold. Fixed by sizing the outer grid so a padded crop never exceeds `sr_tile`.

3. **The "invisible" text layer wasn't actually invisible.** `insert_text(..., color=(1,1,1,0))` was meant to mean "white, fully transparent." PyMuPDF's `color` parameter has no alpha channel though; a four value tuple gets read as CMYK, so `(1,1,1,0)` rendered as a real, visible, mid grey ink color, stamped in a mismatched font right on top of the original scanned glyphs. Fixed by using PDF render mode 3, true invisible text, the standard technique OCR tools use for a searchable but invisible layer.

4. **Scanned pages were being rendered twice.** Once as a full page background (`enhance_page_render`), then again per extracted image block (`enhance_block`) on top of it. Two independently processed copies of the same content, placed through two different coordinate paths. Fixed by choosing one source of truth per page, based on whether the extracted blocks will actually survive content decision filtering, not just their nominal bounding box coverage. An earlier version of this fix checked coverage only, which caused blank pages by skipping the background without confirming the blocks would really draw.

5. **The whitening pass was gated shut for almost all real content.** Background whitening was gated on `classify_image`'s `is_diagram` flag, which requires `color_std < 80`. Real, tightly cropped, high contrast black on white content, text, line art, stamps, is about as bimodal as an image gets, and that reads as a high std, not a low one. Tested against synthetic samples matching real title text, body text, and stamp text: none of them qualified. Fixed by gating on a direct, purpose built background lightness test instead of repurposing a classifier that was tuned for a different decision entirely.

6. **Once whitening actually ran, it overcorrected.** After the gate above was broadened, a linear formula for scaling `C` with resolution pushed the adaptive threshold constant past 30 on 4x SR'd full pages, aggressive enough to erase legitimate thin, antialiased line content, not just clean up boundary halos. Fixed by dampening that scaling curve with a square root and putting a hard cap on it (`whiten_c_max`).

7. **Solid filled areas were getting hollowed out.** This one is root caused, with the fix pending integration. `adaptiveThreshold` makes its decision relative to the local window mean, not against an absolute brightness value. When a window sits entirely inside a large solid filled region, a bold letter stroke, a filled diagram element, the local mean is already dark, so subtracting `C` from an already low mean leaves a threshold low enough that ordinary pixel level noise pushes a real share of true interior ink pixels above it, misclassifying them as background. In a synthetic solid fill test, 100% of the deep interior pixels were misclassified. The fix needs an absolute brightness floor as an additional condition alongside the existing relative check, so a pixel only ever gets whitened if it's both above the local threshold and genuinely light in absolute terms, never based on the relative computation alone.

---

## Project Structure

```
pdf_ai_enhancer/
|-- requirements.txt      All Python dependencies, with compatibility notes
|-- config.py              Typed config (Pydantic), the single source of truth for defaults
|-- config.yaml             Runtime overrides
|-- extractor.py            Stage 1: PDF content extraction
|-- image_enhancer.py       Stage 2: Real-ESRGAN, CLAHE, denoising, whitening
|-- ocr_corrector.py        Stage 3: multi engine OCR plus AI correction
|-- diagram_analyzer.py     Stage 4: Vision LLM plus vectorization
|-- pdf_rebuilder.py        Stage 5: reconstructing the enhanced PDF
|-- pipeline.py             Master orchestrator and CLI entrypoint
|-- checkpoint.py           SQLite extraction cache and resume support
|-- README.md               This file
```

---

## Development

```bash
pytest                       # run the test suite
pytest --cov                 # with coverage
```

When you change the whitening or sharpening logic in `image_enhancer.py`, please test it against both a synthetic single edge case and a dense, solid fill case before shipping. Several of the fixes above only showed up on one of those two, not both, which is exactly why they weren't caught sooner.

When adding a new config field, add it to the matching Pydantic model in `config.py`, not just to `config.yaml`, so it stays validated and typed, and write down its default and the reasoning behind it right there in the code. Several of the fixes above depended on constants like these, and whoever touches this next deserves to know why a value was chosen, not just what it is.

---

## License

Internal, proprietary. Reach out to the project maintainer for usage terms.
