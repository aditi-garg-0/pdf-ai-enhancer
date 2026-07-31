# pipeline.py
"""
run_pipeline wrapper that uses your existing extractor, enhancer and the advanced pdf_rebuilder.
This function is the entrypoint used by the FastAPI app and CLI.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import get_config
from extractor import PDFExtractor
from pdf_rebuilder import PDFRebuilder, cfg_get

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_pipeline(input_pdf: str, output_pdf: str, cfg: Optional[object] = None):
    """
    High-level pipeline entrypoint.
    - input_pdf: path to source PDF
    - output_pdf: path to write enhanced PDF
    - cfg: optional Config object or dict; if None, get_config() is used
    """
    cfg = cfg or get_config()
    logger.info("Pipeline start: %s -> %s", input_pdf, output_pdf)

    # Extract (checkpoint.py's extraction cache existed in this codebase but
    # was never actually called from here -- every single run re-rendered
    # every page from scratch even when re-processing the exact same PDF,
    # e.g. during iterative testing of enhancement settings. This adds a
    # cache check/store around just the extraction step; it does NOT cache
    # image enhancement/SR itself, which is genuine per-run compute and the
    # dominant cost -- see the selective_sr and double-tiling fixes in
    # image_enhancer.py for that side of runtime, and the API response cache
    # in diagram_analyzer.py/ocr_corrector.py for the Vision/OCR-correction
    # calls, which ARE now cached.)
    checkpoint = None
    if getattr(cfg.pipeline, "enable_checkpoint", True):
        try:
            from checkpoint import CheckpointDB
            checkpoint = CheckpointDB(cfg.pipeline.checkpoint_db)
        except Exception as e:
            logger.debug(f"Checkpoint DB unavailable for extraction cache: {e}")

    doc = checkpoint.get_extraction(input_pdf) if checkpoint else None
    if doc is None:
        extractor = PDFExtractor(cfg)
        doc = extractor.extract(input_pdf)
        if checkpoint:
            try:
                checkpoint.save_extraction(input_pdf, doc)
            except Exception as e:
                logger.debug(f"Failed to cache extraction: {e}")
    else:
        logger.info("Extraction: using cached result (source PDF unchanged since last run)")

    # Rebuild (internally calls enhancer, OCR, diagram analyzer)
    rebuilder = PDFRebuilder(cfg)
    rebuilder.rebuild(doc, output_pdf)

    logger.info("Pipeline finished: %s", output_pdf)
    return output_pdf