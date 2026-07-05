"""
Documents & Reports API routes.

Documents: markdown files in data/documents/ (agent-created .md files)
Reports: PDF files in data/reports/ (generated research reports)
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from remy.config.settings import settings

logger = logging.getLogger("DocumentsRoutes")

router = APIRouter()


def _docs_dir() -> Path:
    return Path(settings.DATA_DIR) / "documents"


def _reports_dir() -> Path:
    return Path(settings.DATA_DIR) / "reports"


def _safe_path(base: Path, filename: str) -> Path:
    """Resolve path and ensure it stays inside base dir (path traversal guard)."""
    base = base.resolve()
    target = (base / filename).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return target


# ── DOCUMENTS ────────────────────────────────────────────────────────────────

@router.get("/documents")
async def list_documents():
    """List all .md files in data/documents/."""
    docs_dir = _docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for f in sorted(docs_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        items.append({
            "name": f.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })

    return {"documents": items}


@router.get("/documents/{filename}")
async def get_document(filename: str):
    """Get document content. Looks in data/documents/ first, then data/ root."""
    # Try documents/ first
    docs_dir = _docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_path(docs_dir, filename)

    if not path.exists():
        # Fallback to data/ root for legacy files
        data_dir = Path(settings.DATA_DIR)
        path = _safe_path(data_dir, filename)

    if not path.exists() or path.suffix != ".md":
        raise HTTPException(status_code=404, detail="Document not found")

    content = path.read_text(encoding="utf-8")
    stat = path.stat()
    return {
        "name": filename,
        "content": content,
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


class DocumentUpdate(BaseModel):
    content: str


@router.put("/documents/{filename}")
async def update_document(filename: str, body: DocumentUpdate):
    """Create or update a .md document in data/documents/."""
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files allowed")

    docs_dir = _docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_path(docs_dir, filename)

    path.write_text(body.content, encoding="utf-8")
    logger.info("Document saved: %s (%d bytes)", filename, len(body.content))
    return {"ok": True, "name": filename, "size": len(body.content.encode())}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document. Checks data/documents/ then data/ root."""
    docs_dir = _docs_dir()
    data_dir = Path(settings.DATA_DIR)

    path = _safe_path(docs_dir, filename)
    if not path.exists():
        path = _safe_path(data_dir, filename)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    if path.suffix != ".md":
        raise HTTPException(status_code=400, detail="Only .md files can be deleted here")

    path.unlink()
    logger.info("Document deleted: %s", filename)
    return {"ok": True, "deleted": filename}


# ── REPORTS ──────────────────────────────────────────────────────────────────

@router.get("/reports")
async def list_reports():
    """List all PDF reports in data/reports/."""
    reports_dir = _reports_dir()
    if not reports_dir.exists():
        return {"reports": []}

    items = []
    for f in sorted(reports_dir.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        items.append({
            "name": f.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return {"reports": items}


@router.delete("/reports/{filename}")
async def delete_report(filename: str):
    """Delete a PDF report from data/reports/."""
    reports_dir = _reports_dir()
    path = _safe_path(reports_dir, filename)

    if not path.exists() or path.suffix != ".pdf":
        raise HTTPException(status_code=404, detail="Report not found")

    path.unlink()
    logger.info("Report deleted: %s", filename)
    return {"ok": True, "deleted": filename}
