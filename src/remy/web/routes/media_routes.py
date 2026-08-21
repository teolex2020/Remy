"""
Media routes — serve generated images, browser screenshots, PDF reports.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from remy.config.settings import settings
from remy.core.project_store import project_artifact_dir

logger = logging.getLogger("WebAPI")

router = APIRouter()


@router.get("/generated_images/{filename}")
async def serve_generated_image(filename: str):
    """Serve a generated image file."""
    image_dir = project_artifact_dir(
        "generated_images",
        legacy_data_dir=settings.DATA_DIR,
    )
    filepath = (image_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(image_dir.resolve()):
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = "image/png"
    if filepath.suffix.lower() in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif filepath.suffix.lower() == ".webp":
        media_type = "image/webp"
    return FileResponse(filepath, media_type=media_type)


@router.get("/browser_screenshots/{filename}")
async def serve_browser_screenshot(filename: str):
    """Serve a browser screenshot file."""
    image_dir = project_artifact_dir(
        "browser_screenshots",
        legacy_data_dir=settings.DATA_DIR,
    )
    filepath = (image_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(image_dir.resolve()):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(filepath, media_type="image/png")


@router.get("/reports/{filename}")
async def serve_report(filename: str, download: bool = False):
    """Serve a generated PDF report."""
    reports_dir = project_artifact_dir(
        "reports",
        legacy_data_dir=settings.DATA_DIR,
    )
    filepath = (reports_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(reports_dir.resolve()):
        raise HTTPException(status_code=404, detail="Report not found")
    if download:
        return FileResponse(
            filepath,
            media_type="application/pdf",
            filename=filepath.name,
            content_disposition_type="attachment",
        )
    return FileResponse(filepath, media_type="application/pdf")


@router.get("/presentations/{filename}")
async def serve_presentation(filename: str):
    """Serve a generated PPTX presentation."""
    pres_dir = project_artifact_dir(
        "presentations",
        legacy_data_dir=settings.DATA_DIR,
    )
    filepath = (pres_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(pres_dir.resolve()):
        raise HTTPException(status_code=404, detail="Presentation not found")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
