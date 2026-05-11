# checkpoint.py
"""
SQLite-backed checkpoint + disk cache system.

Allows the pipeline to:
  - Resume an interrupted run from the last processed page
  - Cache Claude API responses (avoid re-paying for same image)
  - Track per-page status across runs
  - Store extraction results so re-runs skip Stage 1 entirely
"""

from __future__ import annotations

import json
import pickle
import gzip
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from loguru import logger

# Local cache directory for externalized blobs
CACHE_DIR = Path.home() / ".pdf_enhancer_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Threshold (bytes) above which we externalize the blob to disk
BLOB_EXTERNALIZE_THRESHOLD = 50 * 1024 * 1024  # 50 MB


# ──────────────────────────────────────────────────────────────
#  Helpers for pickling / externalization
# ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()

def _safe_pickle_and_maybe_externalize(obj: object, pdf_path: str) -> Tuple[bytes, Optional[str]]:
    """
    Pickle and gzip the object. If compressed size > threshold, write to file and return (b'', filepath).
    Returns (gz_blob_or_empty, external_path_or_None).
    """
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    gz = gzip.compress(blob)
    if len(gz) > BLOB_EXTERNALIZE_THRESHOLD:
        fname = hashlib.sha256(pdf_path.encode()).hexdigest()[:24]
        out_path = CACHE_DIR / f"{fname}.extraction.pickle.gz"
        with open(out_path, "wb") as f:
            f.write(gz)
        return b"", str(out_path)
    return gz, None

def _load_pickled_extraction(blob: bytes, external_path: Optional[str]) -> object:
    """
    Load pickled object either from blob (gz) or from external file path.
    """
    if external_path:
        with open(external_path, "rb") as f:
            gz = f.read()
    else:
        gz = blob
    raw = gzip.decompress(gz)
    return pickle.loads(raw)

def _strip_heavy_fields_for_pickle(doc: Any) -> Any:
    """
    Return a deep-copied doc with heavy fields removed or replaced by lightweight placeholders.
    - Remove page.rendered_image
    - Replace ImageBlock.image_bytes with empty bytes
    """
    try:
        # Deep copy via pickle to avoid mutating original
        doc_copy = pickle.loads(pickle.dumps(doc, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        # If deep copy fails, operate on original (best-effort)
        doc_copy = doc

    try:
        for p in getattr(doc_copy, "pages", []):
            # Remove large numpy arrays
            try:
                p.rendered_image = None
            except Exception:
                pass
            # Replace image bytes with empty bytes to keep metadata
            for ib in getattr(p, "image_blocks", []):
                try:
                    ib.image_bytes = b""
                except Exception:
                    pass
    except Exception:
        # If anything fails, return the best-effort copy
        pass

    return doc_copy


# ──────────────────────────────────────────────────────────────
#  CheckpointDB
# ──────────────────────────────────────────────────────────────

class CheckpointDB:
    """
    Lightweight SQLite database for pipeline state management.

    Tables:
      - extractions   : cached ExtractedDocument objects (pickle or externalized file)
      - page_status   : per-page completion status
      - api_cache     : Claude/OpenAI response cache (keyed by content hash)
      - run_log       : timestamped run history
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS extractions (
        pdf_path    TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL,
        data_blob   BLOB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS page_status (
        run_id      TEXT NOT NULL,
        pdf_path    TEXT NOT NULL,
        page_num    INTEGER NOT NULL,
        status      TEXT NOT NULL,  -- pending | done | error
        updated_at  TEXT NOT NULL,
        PRIMARY KEY (run_id, pdf_path, page_num)
    );

    CREATE TABLE IF NOT EXISTS api_cache (
        content_hash  TEXT PRIMARY KEY,
        model         TEXT NOT NULL,
        prompt_hash   TEXT NOT NULL,
        response      TEXT NOT NULL,
        created_at    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS run_log (
        run_id       TEXT PRIMARY KEY,
        pdf_path     TEXT NOT NULL,
        started_at   TEXT NOT NULL,
        finished_at  TEXT,
        status       TEXT NOT NULL,  -- running | complete | failed
        pages_total  INTEGER,
        pages_done   INTEGER DEFAULT 0
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.db_path.with_suffix(".extraction_meta.json")
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    # ── Extraction Cache ──────────────────────────────────────

    def save_extraction(self, pdf_path: str, doc: Any) -> None:
        """Pickle and store an ExtractedDocument. Externalize large blobs to disk."""
        try:
            # Make a safe copy with heavy fields removed
            doc_for_pickle = _strip_heavy_fields_for_pickle(doc)

            gz_blob, external_path = _safe_pickle_and_maybe_externalize(doc_for_pickle, pdf_path)

            with self._conn() as conn:
                if external_path:
                    # store an empty blob but record external path in sidecar meta
                    conn.execute(
                        "INSERT OR REPLACE INTO extractions (pdf_path, created_at, data_blob) VALUES (?, ?, ?)",
                        (pdf_path, _now(), sqlite3.Binary(b""))
                    )
                    # Update sidecar meta map
                    meta_map = {}
                    if self._meta_path.exists():
                        try:
                            meta_map = json.loads(self._meta_path.read_text())
                        except Exception:
                            meta_map = {}
                    meta_map[pdf_path] = {"external_path": external_path, "created_at": _now()}
                    self._meta_path.write_text(json.dumps(meta_map))
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO extractions (pdf_path, created_at, data_blob) VALUES (?, ?, ?)",
                        (pdf_path, _now(), sqlite3.Binary(gz_blob))
                    )
            logger.debug(f"  Checkpoint: saved extraction for {Path(pdf_path).name}")
        except Exception as e:
            logger.warning(f"  Checkpoint save failed: {e}")

    def get_extraction(self, pdf_path: str) -> Optional[Any]:
        """Retrieve a cached ExtractedDocument if it exists and PDF hasn't changed."""
        try:
            pdf = Path(pdf_path)
            if not pdf.exists():
                return None

            with self._conn() as conn:
                row = conn.execute(
                    "SELECT data_blob, created_at FROM extractions WHERE pdf_path = ?",
                    (pdf_path,)
                ).fetchone()

            # If no DB row, check sidecar meta for externalized file
            if row is None:
                if self._meta_path.exists():
                    try:
                        meta_map = json.loads(self._meta_path.read_text())
                        meta = meta_map.get(pdf_path)
                        if meta:
                            external_path = meta.get("external_path")
                            cached_at = datetime.fromisoformat(meta.get("created_at"))
                            pdf_mtime = datetime.fromtimestamp(pdf.stat().st_mtime)
                            if pdf_mtime > cached_at:
                                logger.debug(f"  Checkpoint: PDF modified since cache — re-extracting")
                                return None
                            doc = _load_pickled_extraction(b"", external_path)
                            logger.info(f"  Checkpoint: loaded cached extraction (external) ({Path(pdf_path).name})")
                            return doc
                    except Exception:
                        pass
                return None

            # Invalidate if PDF was modified after caching
            cached_at = datetime.fromisoformat(row["created_at"])
            pdf_mtime = datetime.fromtimestamp(pdf.stat().st_mtime)
            if pdf_mtime > cached_at:
                logger.debug(f"  Checkpoint: PDF modified since cache — re-extracting")
                return None

            blob = row["data_blob"]
            if blob is None or len(blob) == 0:
                # Look up sidecar meta for external path
                if self._meta_path.exists():
                    try:
                        meta_map = json.loads(self._meta_path.read_text())
                        meta = meta_map.get(pdf_path)
                        if meta and meta.get("external_path"):
                            doc = _load_pickled_extraction(b"", meta.get("external_path"))
                            logger.info(f"  Checkpoint: loaded cached extraction (external) ({Path(pdf_path).name})")
                            return doc
                    except Exception:
                        pass
                return None

            doc = _load_pickled_extraction(blob, None)
            logger.info(f"  Checkpoint: loaded cached extraction ({Path(pdf_path).name})")
            return doc

        except Exception as e:
            logger.debug(f"  Checkpoint load failed: {e}")
            return None

    # ── Page Status ───────────────────────────────────────────

    def mark_page_done(self, run_id: str, pdf_path: str, page_num: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO page_status "
                "(run_id, pdf_path, page_num, status, updated_at) VALUES (?,?,?,?,?)",
                (run_id, pdf_path, page_num, "done", _now())
            )

    def mark_page_error(self, run_id: str, pdf_path: str, page_num: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO page_status "
                "(run_id, pdf_path, page_num, status, updated_at) VALUES (?,?,?,?,?)",
                (run_id, pdf_path, page_num, "error", _now())
            )

    def get_done_pages(self, run_id: str, pdf_path: str) -> set[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT page_num FROM page_status WHERE run_id=? AND pdf_path=? AND status='done'",
                (run_id, pdf_path)
            ).fetchall()
        return {r["page_num"] for r in rows}

    # ── API Response Cache ────────────────────────────────────

    def cache_api_response(
        self,
        content: bytes,
        prompt:  str,
        model:   str,
        response: str,
    ) -> None:
        """Cache a Claude/OpenAI response keyed by content + prompt hash."""
        content_hash = hashlib.sha256(content).hexdigest()
        prompt_hash  = hashlib.sha256(prompt.encode()).hexdigest()
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO api_cache "
                    "(content_hash, model, prompt_hash, response, created_at) VALUES (?,?,?,?,?)",
                    (content_hash, model, prompt_hash, response, _now())
                )
        except Exception as e:
            logger.debug(f"  API cache save error: {e}")

    def get_cached_api_response(
        self, content: bytes, prompt: str, model: str
    ) -> Optional[str]:
        content_hash = hashlib.sha256(content).hexdigest()
        prompt_hash  = hashlib.sha256(prompt.encode()).hexdigest()
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT response FROM api_cache "
                    "WHERE content_hash=? AND prompt_hash=? AND model=?",
                    (content_hash, prompt_hash, model)
                ).fetchone()
            return row["response"] if row else None
        except Exception:
            return None

    # ── Run Log ───────────────────────────────────────────────

    def start_run(self, run_id: str, pdf_path: str, pages_total: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_log "
                "(run_id, pdf_path, started_at, status, pages_total) VALUES (?,?,?,?,?)",
                (run_id, pdf_path, _now(), "running", pages_total)
            )

    def finish_run(self, run_id: str, pages_done: int, success: bool = True) -> None:
        status = "complete" if success else "failed"
        with self._conn() as conn:
            conn.execute(
                "UPDATE run_log SET finished_at=?, status=?, pages_done=? WHERE run_id=?",
                (_now(), status, pages_done, run_id)
            )

    def get_run_history(self, pdf_path: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if pdf_path:
                rows = conn.execute(
                    "SELECT * FROM run_log WHERE pdf_path=? ORDER BY started_at DESC",
                    (pdf_path,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM run_log ORDER BY started_at DESC LIMIT 50"
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Maintenance ─────────────────────────────────────────

    def clear_extraction_cache(self, pdf_path: Optional[str] = None) -> None:
        with self._conn() as conn:
            if pdf_path:
                conn.execute("DELETE FROM extractions WHERE pdf_path=?", (pdf_path,))
                # remove sidecar meta entry and external file if present
                if self._meta_path.exists():
                    try:
                        meta_map = json.loads(self._meta_path.read_text())
                        meta = meta_map.pop(pdf_path, None)
                        if meta and meta.get("external_path"):
                            try:
                                Path(meta["external_path"]).unlink(missing_ok=True)
                            except Exception:
                                pass
                        self._meta_path.write_text(json.dumps(meta_map))
                    except Exception:
                        pass
            else:
                conn.execute("DELETE FROM extractions")
                # remove all externalized files referenced in meta
                if self._meta_path.exists():
                    try:
                        meta_map = json.loads(self._meta_path.read_text())
                        for meta in meta_map.values():
                            try:
                                Path(meta.get("external_path", "")).unlink(missing_ok=True)
                            except Exception:
                                pass
                        self._meta_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        logger.info("Extraction cache cleared")

    def clear_api_cache(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM api_cache")
        logger.info("API response cache cleared")

    def stats(self) -> dict:
        with self._conn() as conn:
            n_extractions = conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
            n_api         = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
            n_runs        = conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0]
            n_pages       = conn.execute("SELECT COUNT(*) FROM page_status WHERE status='done'").fetchone()[0]
        return {
            "cached_extractions": n_extractions,
            "cached_api_responses": n_api,
            "total_runs": n_runs,
            "total_pages_processed": n_pages,
        }
