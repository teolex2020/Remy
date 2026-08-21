"""
Documents & Reports API routes.

Documents: markdown files in data/documents/ (agent-created .md files)
Reports: PDF files in data/reports/ (generated research reports)
"""

import logging
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from remy.config.settings import settings
from remy.core.microbrain import current_project_id
from remy.core.project_documents import (
    is_agent_accessible,
    remove_document_access,
    set_agent_access,
)
from remy.core.project_store import project_artifact_dir

logger = logging.getLogger("DocumentsRoutes")

router = APIRouter()


def _docs_dir() -> Path:
    return project_artifact_dir(
        "documents",
        legacy_data_dir=settings.DATA_DIR,
    )


def _reports_dir() -> Path:
    return project_artifact_dir(
        "reports",
        legacy_data_dir=settings.DATA_DIR,
    )


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
            "agent_access": is_agent_accessible(docs_dir, f.name),
        })

    return {"project_id": current_project_id(), "documents": items}


@router.get("/documents/{filename}")
async def get_document(filename: str):
    """Get a user-owned project document."""
    docs_dir = _docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_path(docs_dir, filename)

    if not path.exists() or path.suffix.lower() != ".md":
        raise HTTPException(status_code=404, detail="Document not found")

    content = path.read_text(encoding="utf-8")
    stat = path.stat()
    return {
        "name": filename,
        "content": content,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "agent_access": is_agent_accessible(docs_dir, filename),
    }


class DocumentUpdate(BaseModel):
    content: str
    agent_access: bool | None = None


class DocumentAccessUpdate(BaseModel):
    agent_access: bool


@router.put("/documents/{filename}")
async def update_document(filename: str, body: DocumentUpdate):
    """Create or update a .md document in data/documents/."""
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files allowed")

    docs_dir = _docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_path(docs_dir, filename)

    path.write_text(body.content, encoding="utf-8")
    if body.agent_access is not None:
        set_agent_access(docs_dir, filename, body.agent_access)
    logger.info("Document saved: %s (%d bytes)", filename, len(body.content))
    return {
        "ok": True,
        "name": filename,
        "size": len(body.content.encode()),
        "agent_access": is_agent_accessible(docs_dir, filename),
    }


@router.patch("/documents/{filename}/access")
async def update_document_access(filename: str, body: DocumentAccessUpdate):
    """Explicitly share or unshare one project document with the agent."""
    docs_dir = _docs_dir()
    path = _safe_path(docs_dir, filename)
    if not path.exists() or path.suffix.lower() != ".md":
        raise HTTPException(status_code=404, detail="Document not found")
    set_agent_access(docs_dir, filename, body.agent_access)
    logger.info("Document agent access changed: %s -> %s", filename, body.agent_access)
    return {"ok": True, "name": filename, "agent_access": body.agent_access}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a user-owned project document."""
    docs_dir = _docs_dir()

    path = _safe_path(docs_dir, filename)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    if path.suffix.lower() != ".md":
        raise HTTPException(status_code=400, detail="Only .md files can be deleted here")

    path.unlink()
    remove_document_access(docs_dir, filename)
    logger.info("Document deleted: %s", filename)
    return {"ok": True, "deleted": filename}


# ── REPORTS ──────────────────────────────────────────────────────────────────

@router.get("/reports")
async def list_reports():
    """List all PDF reports in data/reports/."""
    reports_dir = _reports_dir()
    if not reports_dir.exists():
        return {"project_id": current_project_id(), "reports": []}

    items = []
    for f in sorted(reports_dir.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        encoded_name = quote(f.name)
        items.append({
            "name": f.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "preview_url": f"/api/reports/{encoded_name}",
            "download_url": f"/api/reports/{encoded_name}?download=1",
        })
    return {"project_id": current_project_id(), "reports": items}


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
