"""Bulk document ingestion pipeline for Remy.

Accepts arbitrary files (PDF, DOCX, XLSX, CSV, HTML, XML, TXT, MD, ZIP)
and streams them into the brain using the Aura-clean corpus preprocessor.

Entry point: ingest_files_to_brain()
Progress is reported via an optional async callback for SSE streaming.

The corpus preprocessor is bundled (remy.core.corpus_preprocessor). If it
is not available, a built-in fallback handles .txt/.md/.csv/.pdf/.docx
using the dependencies already installed in Remy.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Callable

logger = logging.getLogger("BulkIngestion")

# ── Chunking parameters ──────────────────────────────────────────────────────
_CHUNK_SIZE = 1500       # characters per chunk
_CHUNK_OVERLAP = 150     # overlap between adjacent chunks
_MIN_CHUNK = 64          # discard chunks shorter than this

# ── Supported extensions ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".pdf", ".docx", ".xlsx", ".csv",
    ".html", ".htm", ".xml", ".json", ".jsonl",
    ".zip",
}


# ── Preprocessor ─────────────────────────────────────────────────────────────

def _load_preprocessor():
    """Import build_clean_corpus from the bundled corpus preprocessor."""
    try:
        from remy.core.corpus_preprocessor import build_clean_corpus
        return build_clean_corpus
    except Exception as exc:
        logger.warning("Could not load corpus preprocessor: %s", exc)
        return None


# ── Text chunker ─────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of ~size chars at paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text] if len(text) >= _MIN_CHUNK else []

    # Try to split on paragraph boundaries first
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current and len(current) >= _MIN_CHUNK:
                chunks.append(current)
            if len(para) > size:
                # Large paragraph — split by sentences or hard-cut
                pos = 0
                while pos < len(para):
                    end = pos + size
                    if end < len(para):
                        # Try to cut at a sentence boundary
                        cut = para.rfind(". ", pos, end)
                        if cut == -1 or cut <= pos:
                            cut = end
                        else:
                            cut += 1  # include the period
                    else:
                        cut = len(para)
                    chunk = para[pos:cut].strip()
                    if len(chunk) >= _MIN_CHUNK:
                        chunks.append(chunk)
                    pos = max(pos + 1, cut - overlap)
                current = ""
            else:
                current = para

    if current and len(current) >= _MIN_CHUNK:
        chunks.append(current)

    return chunks


# ── Fallback extractors (when Aura-clean preprocessor is not available) ──────

def _extract_pdf_fallback(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n\n".join(parts)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", path, exc)
        return ""


def _extract_docx_fallback(path: Path) -> str:
    try:
        import zipfile as _zf
        import xml.etree.ElementTree as ET
        with _zf.ZipFile(str(path)) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = [node.text or "" for node in tree.findall(".//w:t", ns)]
        return " ".join(t for t in texts if t.strip())
    except Exception as exc:
        logger.warning("DOCX extraction failed for %s: %s", path, exc)
        return ""


def _extract_xlsx_fallback(path: Path) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        parts = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join(str(c) for c in row if c is not None)
                if row_text.strip():
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as exc:
        logger.warning("XLSX extraction failed for %s: %s", path, exc)
        return ""


def _extract_file_fallback(path: Path) -> str:
    """Fallback single-file text extraction without the Aura preprocessor."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf_fallback(path)
    if ext == ".docx":
        return _extract_docx_fallback(path)
    if ext == ".xlsx":
        return _extract_xlsx_fallback(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Text read failed for %s: %s", path, exc)
        return ""


# ── Main ingestion pipeline ───────────────────────────────────────────────────

async def ingest_files_to_brain(
    files: list[Path],
    *,
    tags: list[str] | None = None,
    pin: bool = False,
    source_label: str = "bulk-upload",
    progress_cb: Callable[[dict], None] | None = None,
) -> dict:
    """Ingest a list of files (including ZIPs) into the brain.

    progress_cb receives dicts: {phase, file, chunk, total_chunks, stored, skipped, message}
    Returns {stored, skipped, errors, chunks_total, files_processed}
    """
    from remy.core.agent_tools import Level, brain, brain_lock

    base_tags = list(tags or []) + ["kb-ingested", "bulk-ingested", source_label]
    level = Level.IDENTITY if pin else Level.DOMAIN

    def _report(phase: str, **kwargs):
        if progress_cb:
            try:
                progress_cb({"phase": phase, **kwargs})
            except Exception:
                pass

    # Expand ZIPs and collect all leaf files
    all_files: list[Path] = []
    tmp_dirs: list[Path] = []

    for f in files:
        ext = f.suffix.lower()
        if ext == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="remy_bulk_"))
            tmp_dirs.append(tmp)
            try:
                with zipfile.ZipFile(str(f)) as zf:
                    zf.extractall(str(tmp))
                for child in sorted(tmp.rglob("*")):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS - {".zip"}:
                        all_files.append(child)
            except Exception as exc:
                logger.warning("ZIP extraction failed for %s: %s", f, exc)
                _report("error", file=str(f), message=f"ZIP error: {exc}")
        else:
            all_files.append(f)

    total_files = len(all_files)
    _report("start", total_files=total_files, message=f"Starting bulk ingestion of {total_files} file(s)")

    # Try to use Aura-clean preprocessor for batch extraction
    build_clean_corpus = await asyncio.get_event_loop().run_in_executor(None, _load_preprocessor)

    stored = 0
    skipped = 0
    errors = 0
    chunks_total = 0

    if build_clean_corpus and all_files:
        # Group files into a temp source dir for the preprocessor
        tmp_src = Path(tempfile.mkdtemp(prefix="remy_src_"))
        tmp_out = Path(tempfile.mkdtemp(prefix="remy_out_"))
        tmp_dirs += [tmp_src, tmp_out]

        # Symlink or copy files into temp source dir preserving names
        seen_names: dict[str, int] = {}
        for f in all_files:
            name = f.name
            if name in seen_names:
                seen_names[name] += 1
                stem, suf = name.rsplit(".", 1) if "." in name else (name, "")
                name = f"{stem}_{seen_names[name]}.{suf}" if suf else f"{stem}_{seen_names[name]}"
            else:
                seen_names[name] = 0
            dest = tmp_src / name
            try:
                shutil.copy2(str(f), str(dest))
            except Exception:
                pass

        ext_set = {e.lstrip(".") for e in SUPPORTED_EXTENSIONS - {".zip"}}

        last_progress: dict = {}

        def _sync_progress(payload: dict):
            last_progress.update(payload)
            msg = payload.get("message", "")
            cur = payload.get("current_source_path", "")
            _report(
                "extracting",
                file=Path(cur).name if cur else "",
                message=msg,
                processed=payload.get("processed_file_count", 0),
                total_files=payload.get("selected_file_count", total_files),
            )

        def _run_preprocessor():
            return build_clean_corpus(
                source_roots=[tmp_src],
                output_root=tmp_out,
                manifest_output=None,
                report_output=None,
                progress_output=None,
                extensions=ext_set,
                max_files=0,
                max_bytes_per_file=0,
                min_clean_chars=_MIN_CHUNK,
                source_origin_channel="web_upload",
                source_acquisition_surface="bulk_ingest",
                verification_mode_hint="conflict_scan",
                extractor_provider="built_in",
                clear_output=False,
                progress_callback=_sync_progress,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _run_preprocessor)
        except Exception as exc:
            logger.error("Preprocessor failed: %s", exc)
            _report("error", message=f"Preprocessor error: {exc}")
            build_clean_corpus = None  # fall through to per-file fallback

        if build_clean_corpus:
            # Read all .txt output files and ingest their chunks
            out_files = sorted(tmp_out.rglob("*.txt"))
            total_out = len(out_files)
            _report("ingesting", total_files=total_out, message=f"Ingesting {total_out} extracted text file(s)")

            for idx, out_f in enumerate(out_files, 1):
                try:
                    text = out_f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                chunks = chunk_text(text)
                orig_name = out_f.stem.rsplit(".", 1)[0] if "." in out_f.stem else out_f.stem
                file_tags = base_tags + [f"source:{orig_name}"]
                for ci, chunk in enumerate(chunks):
                    chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:12]
                    meta = {
                        "admission_class": "bulk_ingested",
                        "source_file": orig_name,
                        "chunk_index": ci,
                        "chunk_hash": chunk_hash,
                        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "source_label": source_label,
                    }
                    try:
                        with brain_lock:
                            brain.store(chunk, level=level, tags=file_tags, metadata=meta)
                        stored += 1
                        chunks_total += 1
                    except Exception as exc:
                        logger.warning("Store failed for chunk %d of %s: %s", ci, orig_name, exc)
                        skipped += 1

                _report(
                    "ingesting",
                    file=orig_name,
                    chunk=len(chunks),
                    stored=stored,
                    skipped=skipped,
                    processed=idx,
                    total_files=total_out,
                    message=f"[{idx}/{total_out}] {orig_name}: {len(chunks)} chunks → {stored} stored",
                )

            try:
                with brain_lock:
                    brain.flush()
            except Exception:
                pass

    # Fallback: process remaining files individually (when preprocessor unavailable)
    if not build_clean_corpus:
        for idx, f in enumerate(all_files, 1):
            _report("extracting", file=f.name, processed=idx, total_files=total_files,
                    message=f"[{idx}/{total_files}] Extracting {f.name}")
            text = await asyncio.get_event_loop().run_in_executor(None, _extract_file_fallback, f)
            if not text.strip():
                skipped += 1
                continue
            chunks = chunk_text(text)
            file_tags = base_tags + [f"source:{f.stem}"]
            for ci, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:12]
                meta = {
                    "admission_class": "bulk_ingested",
                    "source_file": f.name,
                    "chunk_index": ci,
                    "chunk_hash": chunk_hash,
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "source_label": source_label,
                }
                try:
                    with brain_lock:
                        brain.store(chunk, level=level, tags=file_tags, metadata=meta)
                    stored += 1
                    chunks_total += 1
                except Exception as exc:
                    logger.warning("Store failed chunk %d of %s: %s", ci, f.name, exc)
                    skipped += 1

            _report(
                "ingesting",
                file=f.name,
                chunk=len(chunks),
                stored=stored,
                skipped=skipped,
                processed=idx,
                total_files=total_files,
                message=f"[{idx}/{total_files}] {f.name}: {len(chunks)} chunks → {stored} stored",
            )

        try:
            with brain_lock:
                brain.flush()
        except Exception:
            pass

    # Cleanup temp dirs
    for d in tmp_dirs:
        try:
            shutil.rmtree(str(d), ignore_errors=True)
        except Exception:
            pass

    summary = {
        "stored": stored,
        "skipped": skipped,
        "errors": errors,
        "chunks_total": chunks_total,
        "files_processed": total_files,
    }
    _report("done", **summary, message=f"Done: {stored} chunks stored from {total_files} file(s)")
    return summary


async def sse_bulk_ingest_stream(
    files: list[Path],
    *,
    tags: list[str] | None = None,
    pin: bool = False,
    source_label: str = "bulk-upload",
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted progress events for bulk ingestion."""
    import json as _json

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _cb(payload: dict):
        queue.put_nowait(payload)

    async def _run():
        try:
            await ingest_files_to_brain(files, tags=tags, pin=pin,
                                        source_label=source_label, progress_cb=_cb)
        finally:
            queue.put_nowait(None)  # sentinel

    task = asyncio.create_task(_run())

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield "data: {\"phase\": \"heartbeat\"}\n\n"
            continue
        if item is None:
            break
        yield f"data: {_json.dumps(item, ensure_ascii=False)}\n\n"

    await task
