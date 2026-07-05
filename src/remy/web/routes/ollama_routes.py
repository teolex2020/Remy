"""
Ollama management routes — install, pull models, status.
All user-facing: no terminal needed.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("OllamaRoutes")

router = APIRouter()

# Popular models shown in the UI — name, label, size hint, description
POPULAR_MODELS = [
    {
        "name": "gemma3:1b",
        "label": "Gemma 3 1B",
        "size": "0.8 GB",
        "description": "Google's smallest model. Fast, good for simple tasks.",
        "tags": ["fast", "small"],
    },
    {
        "name": "gemma3:4b",
        "label": "Gemma 3 4B",
        "size": "2.5 GB",
        "description": "Google's balanced model. Good quality for most tasks.",
        "tags": ["balanced"],
    },
    {
        "name": "mistral:7b",
        "label": "Mistral 7B",
        "size": "4.1 GB",
        "description": "Fast and capable. Great all-rounder for chat and reasoning.",
        "tags": ["balanced", "popular"],
    },
    {
        "name": "llama3.2:3b",
        "label": "Llama 3.2 3B",
        "size": "2.0 GB",
        "description": "Meta's compact model. Efficient on modest hardware.",
        "tags": ["fast", "small"],
    },
    {
        "name": "llama3.1:8b",
        "label": "Llama 3.1 8B",
        "size": "4.7 GB",
        "description": "Meta's 8B model. Strong reasoning and instruction following.",
        "tags": ["balanced", "popular"],
    },
    {
        "name": "phi4:14b",
        "label": "Phi-4 14B",
        "size": "8.5 GB",
        "description": "Microsoft's research model. Punches above its weight.",
        "tags": ["powerful"],
    },
    {
        "name": "qwen2.5:7b",
        "label": "Qwen 2.5 7B",
        "size": "4.4 GB",
        "description": "Alibaba's multilingual model. Strong in Ukrainian & English.",
        "tags": ["multilingual", "balanced"],
    },
    {
        "name": "deepseek-r1:8b",
        "label": "DeepSeek R1 8B",
        "size": "4.9 GB",
        "description": "Reasoning-focused model. Great for complex analysis.",
        "tags": ["reasoning"],
    },
]


@router.get("/ollama/status")
async def ollama_status():
    """Return Ollama service status and list of installed models."""
    from remy.core.ollama_service import ollama_service
    return await ollama_service.status()


@router.get("/ollama/models/popular")
async def ollama_popular_models():
    """Return curated list of recommended models with metadata."""
    from remy.core.ollama_service import ollama_service

    installed_names: set[str] = set()
    try:
        for m in await ollama_service.list_models():
            # Normalize: "mistral:latest" matches "mistral" and "mistral:7b"
            installed_names.add(m["name"])
            base = m["name"].split(":")[0]
            installed_names.add(base)
    except Exception:
        pass

    result = []
    for m in POPULAR_MODELS:
        base = m["name"].split(":")[0]
        installed = m["name"] in installed_names or base in installed_names
        result.append({**m, "installed": installed})
    return {"models": result}


@router.post("/ollama/start")
async def ollama_start():
    """Start Ollama service (download binary if needed). Returns SSE stream."""
    from remy.core.ollama_service import ollama_service

    async def _stream():
        queue: list[dict] = []
        done = False

        def _cb(payload: dict):
            queue.append(payload)

        import asyncio

        async def _run():
            nonlocal done
            await ollama_service.ensure_running(progress_cb=_cb)
            done = True

        task = asyncio.create_task(_run())

        while not done or queue:
            while queue:
                evt = queue.pop(0)
                yield f"data: {json.dumps(evt)}\n\n"
            if not done:
                await asyncio.sleep(0.1)

        await task
        status = await ollama_service.status()
        yield f"data: {json.dumps({'phase': 'status', **status})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class PullRequest(BaseModel):
    name: str


@router.post("/ollama/pull")
async def ollama_pull(body: PullRequest):
    """Download a model. Returns SSE stream of progress events."""
    from remy.core.ollama_service import ollama_service

    model_name = body.name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    async def _stream():
        queue: list[dict] = []
        done = False

        def _cb(payload: dict):
            queue.append(payload)

        import asyncio

        async def _run():
            nonlocal done
            # Ensure Ollama is running first
            if not ollama_service.is_ready():
                await ollama_service.ensure_running(progress_cb=_cb)
            await ollama_service.pull_model(model_name, progress_cb=_cb)
            done = True

        task = asyncio.create_task(_run())

        while not done or queue:
            while queue:
                evt = queue.pop(0)
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            if not done:
                await asyncio.sleep(0.1)

        await task

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class DeleteRequest(BaseModel):
    name: str


@router.delete("/ollama/model")
async def ollama_delete_model(body: DeleteRequest):
    """Remove an installed model."""
    from remy.core.ollama_service import ollama_service

    ok = await ollama_service.delete_model(body.name.strip())
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete model")
    return {"deleted": True, "name": body.name}
