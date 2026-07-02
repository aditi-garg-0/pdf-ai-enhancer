

from __future__ import annotations

import os
import sys
import signal
import time
import traceback
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TaskID
)
from rich import print as rprint

from config import Config, get_config, OUTPUT_DIR
from extractor import PDFExtractor, ExtractedDocument
from pdf_rebuilder import PDFRebuilder, RebuildReport
from checkpoint import CheckpointDB   # defined below in checkpoint.py


console = Console()

# ──────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────

class EnhancementPipeline:
    """
    Runs the full PDF AI enhancement pipeline for one or many files.
    """

    STAGES = ["extract", "enhance", "ocr", "diagrams", "rebuild"]

    def __init__(self, config: Config):
        self.config     = config
        self.extractor  = PDFExtractor(config)
        self.rebuilder  = PDFRebuilder(config)
        self._shutdown  = False

        # Handle Ctrl-C gracefully
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ── Single file ───────────────────────────────────────────

    def run(
        self,
        input_path:  str | Path,
        output_path: Optional[str | Path] = None,
    ) -> RebuildReport:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        logger.info(f"{'='*60}")
        logger.info(f"  PDF AI Enhancer — {input_path.name}")
        logger.info(f"{'='*60}")

        checkpoint = CheckpointDB(self.config.pipeline.checkpoint_db) \
            if self.config.pipeline.enable_checkpoint else None

        report = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console = console,
            transient = False,
        ) as progress:

            # Stage 1: Extract
            task = progress.add_task("Stage 1/5 — Extracting content …", total=1)
            doc = self._stage_extract(input_path, checkpoint, progress, task)
            progress.update(task, completed=1)

            if self._shutdown:
                logger.warning("Shutdown requested — stopping after extraction")
                return None

            # Stage 2–5: Rebuild (handles enhance + ocr + diagrams + rebuild internally)
            task2 = progress.add_task(
                "Stage 2–5 — Enhancing + OCR + Diagrams + Rebuilding …",
                total = len(doc.pages),
            )
            report = self._stage_rebuild(doc, output_path, checkpoint, progress, task2)

        self._print_summary(report)
        return report

    # ── Batch mode ────────────────────────────────────────────

    def run_batch(
        self,
        input_dir:  str | Path,
        output_dir: Optional[str | Path] = None,
        pattern:    str = "*.pdf",
    ) -> list[RebuildReport]:
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(input_dir.glob(pattern))
        if not pdf_files:
            logger.warning(f"No PDFs found in {input_dir} matching '{pattern}'")
            return []

        logger.info(f"Batch mode: {len(pdf_files)} files in {input_dir}")
        reports: list[RebuildReport] = []

        for i, pdf in enumerate(pdf_files, 1):
            if self._shutdown:
                break
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf.name}")
            try:
                out = output_dir / (pdf.stem + self.config.pipeline.output_suffix + ".pdf")
                report = self.run(pdf, out)
                if report:
                    reports.append(report)
            except Exception as e:
                logger.error(f"  Failed: {pdf.name} — {e}")
                logger.debug(traceback.format_exc())

        self._print_batch_summary(reports)
        return reports

    # ── Internal stage runners ────────────────────────────────

    def _stage_extract(
        self,
        input_path: Path,
        checkpoint: Optional["CheckpointDB"],
        progress: Progress,
        task: TaskID,
    ) -> ExtractedDocument:
        if checkpoint:
            cached = checkpoint.get_extraction(str(input_path))
            if cached:
                logger.info("  [cache] Using cached extraction")
                return cached

        doc = self.extractor.extract(input_path)

        if checkpoint:
            checkpoint.save_extraction(str(input_path), doc)

        return doc

    def _stage_rebuild(
        self,
        doc:         ExtractedDocument,
        output_path: Optional[Path],
        checkpoint:  Optional["CheckpointDB"],
        progress:    Progress,
        task:        TaskID,
    ) -> RebuildReport:
        # Wire up progress callback into rebuilder
        original_method = self.rebuilder._build_page

        def tracked_build_page(new_doc, page_info, is_scanned, pq):
            result = original_method(new_doc, page_info, is_scanned, pq)
            progress.advance(task)
            return result

        self.rebuilder._build_page = tracked_build_page
        report = self.rebuilder.rebuild(doc, output_path)
        self.rebuilder._build_page = original_method
        return report

    # ── Display ───────────────────────────────────────────────

    def _print_summary(self, report: Optional[RebuildReport]) -> None:
        if not report:
            return

        table = Table(title="Enhancement Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric",  style="bold")
        table.add_column("Value",   justify="right")

        table.add_row("Pages processed",       str(report.pages_processed))
        table.add_row("Images enhanced",       str(report.total_images))
        table.add_row("Diagrams vectorized",   str(report.total_diagrams))
        table.add_row("Tables embedded",       str(report.total_tables))
        table.add_row("OCR blocks corrected",  str(report.total_ocr_blocks))
        table.add_row("Processing time",       f"{report.processing_time_s:.1f}s")
        table.add_row("Output",                Path(report.output_path).name)

        console.print(table)
        console.print(Panel(
            f"[green bold]✓ Enhancement complete[/]\n"
            f"[dim]{report.output_path}[/]",
            border_style="green"
        ))

    def _print_batch_summary(self, reports: list[RebuildReport]) -> None:
        console.print(f"\n[bold]Batch complete: {len(reports)} files processed[/]")
        for r in reports:
            console.print(f"  ✓ {Path(r.output_path).name} "
                          f"({r.pages_processed} pages, {r.processing_time_s:.0f}s)")

    def _handle_shutdown(self, signum, frame):
        logger.warning("\nShutdown signal received — finishing current page …")
        self._shutdown = True


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

@click.group()
@click.version_option("1.0.0", prog_name="pdf-enhance")
def cli():
    """PDF AI Enhancer — production-grade AI PDF enhancement pipeline."""
    pass


@cli.command("enhance")
@click.argument("input_pdf", type=click.Path(exists=True))
@click.option("-o", "--output",     default=None,  help="Output PDF path")
@click.option("-c", "--config",     default=None,  help="Path to config.yaml")
@click.option("--dpi",              default=None,  type=int, help="Output DPI (default: 300)")
@click.option("--pages",            default=None,  help="Page range e.g. '1-5,8'")
@click.option("--no-sr",            is_flag=True,  help="Disable super-resolution")
@click.option("--no-ocr",           is_flag=True,  help="Disable OCR correction")
@click.option("--no-ai",            is_flag=True,  help="Disable all AI/API calls")
@click.option("--no-vectorize",     is_flag=True,  help="Disable diagram vectorization")
@click.option("--workers",          default=None,  type=int, help="Parallel workers")
@click.option("--api-key",          default=None,  envvar="ANTHROPIC_API_KEY")
def enhance_cmd(input_pdf, output, config, dpi, pages, no_sr, no_ocr,
                no_ai, no_vectorize, workers, api_key):
    """Enhance a single PDF file."""
    cfg = get_config(Path(config) if config else None)

    # Apply CLI overrides
    if dpi:             cfg.pipeline.output_dpi = dpi
    if pages:           cfg.pipeline.pages = pages
    if no_sr:           cfg.enhancement.enable_super_resolution = False
    if no_ocr:          cfg.ocr.enable_ai_correction = False
    if no_ai:
        cfg.ocr.enable_ai_correction   = False
        cfg.diagram.enable_ai_analysis = False
    if no_vectorize:    cfg.diagram.enable_vectorization = False
    if workers:         cfg.pipeline.workers = workers
    if api_key:         cfg.api.anthropic_api_key = api_key

    _setup_logging()
    pipeline = EnhancementPipeline(cfg)
    pipeline.run(input_pdf, output)


@cli.command("batch")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", default=None, help="Output directory")
@click.option("-c", "--config",     default=None, help="Path to config.yaml")
@click.option("--pattern",          default="*.pdf", help="Glob pattern (default: *.pdf)")
@click.option("--api-key",          default=None,  envvar="ANTHROPIC_API_KEY")
def batch_cmd(input_dir, output_dir, config, pattern, api_key):
    """Enhance all PDFs in a directory."""
    cfg = get_config(Path(config) if config else None)
    if api_key:
        cfg.api.anthropic_api_key = api_key

    _setup_logging()
    pipeline = EnhancementPipeline(cfg)
    pipeline.run_batch(input_dir, output_dir, pattern)


@cli.command("config")
@click.option("--save", default="config.yaml", help="Save default config to this path")
def config_cmd(save):
    """Print or save the default configuration."""
    cfg = Config()
    cfg.to_yaml(save)
    rprint(f"[green]Default config saved → {save}[/]")
    rprint("[dim]Edit it then pass with --config config.yaml[/]")


# ──────────────────────────────────────────────────────────────
#  Logging setup
# ──────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level    = level,
        colorize = True,
        format   = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        OUTPUT_DIR / "pipeline.log",
        level    = "DEBUG",
        rotation = "10 MB",
        retention = "7 days",
        encoding = "utf-8",
    )


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()