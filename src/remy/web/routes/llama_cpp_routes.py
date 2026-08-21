"""Local llama.cpp runtime and GGUF model management routes."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from remy.web.routes._helpers import run_in_thread

router = APIRouter()


def _service():
    from remy.core.llama_cpp_service import llama_cpp_service

    return llama_cpp_service


@router.get("/llamacpp/status")
async def llama_cpp_status():
    return await asyncio.to_thread(_service().status)


@router.post("/llamacpp/runtime/install")
async def install_llama_cpp_runtime():
    async def stream():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def report(payload: dict):
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        async def run():
            try:
                await _service().install_runtime(report)
            except Exception as exc:
                await queue.put({"phase": "error", "message": str(exc)})
            finally:
                await queue.put({"phase": "finished"})

        task = asyncio.create_task(run())
        while True:
            event = await queue.get()
            if event.get("phase") == "finished":
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        await task

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/llamacpp/repository")
async def list_repository_gguf(repo_id: str = Query(min_length=3, max_length=200)):
    try:
        files = await _service().repository_files(repo_id)
        return {"repo_id": repo_id, "files": files}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read Hugging Face repository: {exc}",
        ) from exc


class DownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=200)
    filename: str = Field(min_length=6, max_length=500)


class LocalFolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


class LocalModelRequest(BaseModel):
    relative_path: str = Field(min_length=1, max_length=1000)


def _choose_model_folder() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "The native folder picker is unavailable on this system."
        ) from exc
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        return filedialog.askdirectory(
            title="Choose local models folder",
            mustexist=True,
        ) or ""
    finally:
        root.destroy()


@router.post("/llamacpp/local/select-folder")
async def select_local_gguf_folder():
    try:
        selected = await run_in_thread(_choose_model_folder, timeout=None)
        if not selected:
            return {"cancelled": True, "files": []}
        files = await asyncio.to_thread(_service().configure_models_folder, selected)
        return {"cancelled": False, "path": selected, "files": files}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/llamacpp/local/folder")
async def add_local_gguf_folder(body: LocalFolderRequest):
    try:
        files = await asyncio.to_thread(_service().configure_models_folder, body.path)
        return {"path": body.path, "files": files}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/llamacpp/local/model")
async def add_local_gguf_model(body: LocalModelRequest):
    try:
        model = await asyncio.to_thread(
            _service().add_local_model,
            body.relative_path,
        )
        return {"model": model}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/llamacpp/models/download")
async def download_gguf_model(body: DownloadRequest):
    async def stream():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def report(payload: dict):
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        async def run():
            try:
                await _service().download_model(body.repo_id, body.filename, report)
            except Exception as exc:
                await queue.put({"phase": "error", "message": str(exc)})
            finally:
                await queue.put({"phase": "finished"})

        task = asyncio.create_task(run())
        while True:
            event = await queue.get()
            if event.get("phase") == "finished":
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        await task

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


class ModelRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=200)


@router.post("/llamacpp/models/start")
async def start_gguf_model(body: ModelRequest):
    try:
        await asyncio.to_thread(_service().start_model, body.model_id)
        return _service().status()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/llamacpp/stop")
async def stop_llama_cpp():
    await asyncio.to_thread(_service().stop)
    return {"stopped": True}


@router.delete("/llamacpp/models/{model_id}")
async def delete_gguf_model(model_id: str):
    try:
        deleted = await asyncio.to_thread(_service().delete_model, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Local model not found")
    return {"deleted": True, "model_id": model_id}
