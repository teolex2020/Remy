"""
Media routes — serve generated images, browser screenshots, PDF reports.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from remy.web.routes._helpers import _get_api

logger = logging.getLogger("WebAPI")

router = APIRouter()


@router.get("/generated_images/{filename}")
async def serve_generated_image(filename: str):
    """Serve a generated image file."""
    api = _get_api()
    image_dir = Path(api.settings.DATA_DIR) / "generated_images"
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
    api = _get_api()
    image_dir = Path(api.settings.DATA_DIR) / "browser_screenshots"
    filepath = (image_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(image_dir.resolve()):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(filepath, media_type="image/png")


@router.get("/reports/{filename}")
async def serve_report(filename: str):
    """Serve a generated PDF report."""
    api = _get_api()
    reports_dir = Path(api.settings.DATA_DIR) / "reports"
    filepath = (reports_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(reports_dir.resolve()):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(filepath, media_type="application/pdf")


@router.get("/presentations/{filename}")
async def serve_presentation(filename: str):
    """Serve a generated PPTX presentation."""
    api = _get_api()
    pres_dir = Path(api.settings.DATA_DIR) / "presentations"
    filepath = (pres_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(pres_dir.resolve()):
        raise HTTPException(status_code=404, detail="Presentation not found")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
