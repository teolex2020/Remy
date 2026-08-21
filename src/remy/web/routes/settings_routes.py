"""
Settings routes — get/update settings, history, sandbox tools, metrics.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from remy.core.model_registry import _mask_key
from remy.web.routes._helpers import _get_api, run_in_thread

logger = logging.getLogger("WebAPI")

router = APIRouter()


class SettingsPayload(BaseModel):
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    summary_model: str | None = None
    web_search_model: str | None = None
    gemini_model: str | None = None
    gemini_voice: str | None = None
    telegram_bot_token: str | None = None
    proactive_chat_id: int | None = None
    review_model: str | None = None
    browser_vision_model: str | None = None
    browser_backend: str | None = None
    pinchtab_enabled: bool | None = None
    pinchtab_base_url: str | None = None
    pinchtab_timeout_sec: int | None = None
    pinchtab_profile_id: str | None = None
    pinchtab_autostart: bool | None = None
    pinchtab_command: str | None = None
    pinchtab_binary_path: str | None = None
    pinchtab_bootstrap_source: str | None = None
    custom_system_prompt: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None


class WorkspaceGrantPayload(BaseModel):
    path: str | None = None
    name: str | None = None
    read: bool = True
    write: bool = False
    execute: bool = False


class WorkspaceUpdatePayload(BaseModel):
    name: str | None = None
    read: bool = True
    write: bool = False
    execute: bool = False


# Known models per provider — shown in dropdown when provider has a key
class SecretUpdatePayload(BaseModel):
    value: str | None = None


SECRET_DEFINITIONS = {
    "gemini_api_key": {
        "label": "Gemini",
        "setting": "GEMINI_API_KEY",
        "kind": "AI key",
        "description": "Used for chat, summaries, voice, and Gemini-powered workflow blocks.",
    },
    "openrouter_api_key": {
        "label": "OpenRouter",
        "setting": "OPENROUTER_API_KEY",
        "kind": "AI key",
        "description": "Used for OpenRouter models and fallback model routing.",
    },
    "telegram_bot_token": {
        "label": "Telegram bot",
        "setting": "TELEGRAM_BOT_TOKEN",
        "kind": "Notification",
        "description": "Lets automation blocks send messages through your Telegram bot.",
    },
    "smtp_password": {
        "label": "Email app password",
        "setting": "SMTP_PASSWORD",
        "kind": "Notification",
        "description": "Used with your email account for automation email delivery.",
    },
}


def _secret_status(settings, key: str) -> dict:
    definition = SECRET_DEFINITIONS[key]
    setting_name = definition["setting"]
    value = getattr(settings, setting_name, None) or os.environ.get(setting_name, "")
    return {
        "key": key,
        "label": definition["label"],
        "kind": definition["kind"],
        "description": definition["description"],
        "configured": bool(value),
        "masked": _mask_key(str(value or "")),
        "stored": "local runtime settings",
    }


MODEL_PROVIDERS = frozenset(
    {"google", "openai", "anthropic", "deepseek", "xai", "openrouter", "nvidia", "llamacpp"}
)


@router.get("/models")
async def list_available_models():
    """List only explicitly registered models plus linked local GGUF models."""
    from remy.core.model_registry import list_registered_models

    models = [
        {
            "name": str(item.get("name") or ""),
            "provider": str(item.get("provider") or ""),
            "label": str(item.get("name") or ""),
        }
        for item in list_registered_models()
        if item.get("name") and item.get("has_key")
    ]
    seen = {item["name"] for item in models}

    # Managed llama.cpp GGUF models.
    try:
        from remy.core.llama_cpp_service import llama_cpp_service

        for item in llama_cpp_service.list_models():
            name = str(item.get("name") or "")
            if not name or name in seen:
                continue
            size_gb = float(item.get("size_gb") or 0)
            label = f"{item.get('filename')} (llama.cpp, {size_gb:.1f} GB)"
            models.append({"name": name, "provider": "llamacpp", "label": label})
            seen.add(name)
    except Exception:
        pass  # Local model storage is optional.

    return {"models": models}


# ============== MODEL REGISTRY ==============


class ModelRegistryPayload(BaseModel):
    model_name: str
    api_key: str
    provider: str | None = None
    input_price: float | None = None   # $/M input tokens
    output_price: float | None = None  # $/M output tokens
    copy_key_from: str | None = None   # reuse API key from this model name


@router.get("/model-registry")
async def get_model_registry():
    """List all registered models with masked API keys."""
    from remy.core.model_registry import list_registered_models

    return {"models": list_registered_models()}


@router.put("/model-registry")
async def register_model(payload: ModelRegistryPayload):
    """Add or update a model with its API key."""
    from remy.core.model_registry import load_registry, register_model as _register

    provider = str(payload.provider or "").strip().lower() or None
    if provider and provider not in MODEL_PROVIDERS:
        raise HTTPException(status_code=422, detail=f"Unsupported model provider: {provider}")

    api_key = payload.api_key
    if not api_key and payload.copy_key_from:
        # Reuse the key from another registered model
        registry = load_registry()
        src = registry.get(payload.copy_key_from, {})
        api_key = src.get("api_key", "")

    _register(
        payload.model_name,
        api_key,
        provider,
        input_price=payload.input_price,
        output_price=payload.output_price,
    )
    return {
        "ok": True,
        "model": payload.model_name,
        "note": "Model registered. Changes apply immediately.",
    }


@router.delete("/model-registry/{model_name:path}")
async def unregister_model(model_name: str):
    """Remove a model from the registry."""
    from remy.core.model_registry import unregister_model as _unregister

    if _unregister(model_name):
        return {"ok": True, "deleted": model_name}
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in registry")


@router.get("/settings")
async def get_settings():
    """Get current settings (API key masked)."""
    api = _get_api()
    settings = api.settings
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    masked_key = ""
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

    bot_token = settings.TELEGRAM_BOT_TOKEN
    masked_bot = ""
    if bot_token:
        masked_bot = bot_token[:4] + "..." + bot_token[-4:] if len(bot_token) > 10 else "***"

    # Load custom system prompt from file
    custom_prompt = ""
    prompt_path = settings.DATA_DIR / "custom_system_prompt.txt"
    if prompt_path.exists():
        try:
            custom_prompt = prompt_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return {
        "gemini_api_key_masked": masked_key,
        "has_api_key": bool(api_key),
        "openrouter_api_key_masked": _mask_key(settings.OPENROUTER_API_KEY or ""),
        "has_openrouter_key": bool(settings.OPENROUTER_API_KEY),
        "summary_model": settings.SUMMARY_MODEL,
        "web_search_model": settings.WEB_SEARCH_MODEL,
        "gemini_model": settings.GEMINI_MODEL,
        "gemini_voice": settings.GEMINI_VOICE,
        "has_telegram": bool(settings.TELEGRAM_BOT_TOKEN),
        "telegram_bot_masked": masked_bot,
        "proactive_chat_id": settings.PROACTIVE_CHAT_ID,
        "web_host": settings.WEB_HOST,
        "web_port": settings.WEB_PORT,
        "review_model": settings.REVIEW_MODEL,
        "review_enabled": settings.REVIEW_ENABLED,
        "browser_vision_model": settings.BROWSER_VISION_MODEL,
        "browser_backend": settings.BROWSER_BACKEND,
        "pinchtab_enabled": settings.PINCHTAB_ENABLED,
        "pinchtab_base_url": settings.PINCHTAB_BASE_URL,
        "pinchtab_timeout_sec": settings.PINCHTAB_TIMEOUT_SEC,
        "pinchtab_profile_id": settings.PINCHTAB_PROFILE_ID,
        "pinchtab_autostart": settings.PINCHTAB_AUTOSTART,
        "pinchtab_command": settings.PINCHTAB_COMMAND,
        "pinchtab_binary_path": settings.PINCHTAB_BINARY_PATH,
        "pinchtab_bootstrap_source": settings.PINCHTAB_BOOTSTRAP_SOURCE,
        "custom_system_prompt": custom_prompt,
        "smtp_host": getattr(settings, "SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": getattr(settings, "SMTP_PORT", 587),
        "smtp_user": getattr(settings, "SMTP_USER", "") or "",
        "smtp_from": getattr(settings, "SMTP_FROM", "") or "",
        "has_smtp": bool(getattr(settings, "SMTP_USER", None) and getattr(settings, "SMTP_PASSWORD", None)),
    }


@router.get("/secrets")
async def list_secrets():
    """Return local secret configuration status without exposing secret values."""
    api = _get_api()
    return {
        "storage": "local",
        "note": "Secrets are stored on this computer in runtime settings and are never returned in full.",
        "secrets": [_secret_status(api.settings, key) for key in SECRET_DEFINITIONS],
    }


@router.put("/secrets/{secret_key}")
async def update_secret(secret_key: str, payload: SecretUpdatePayload):
    """Set or clear a supported local secret."""
    from remy.config.settings import set_runtime_setting

    if secret_key not in SECRET_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Unknown secret '{secret_key}'")

    api = _get_api()
    settings = api.settings
    setting_name = SECRET_DEFINITIONS[secret_key]["setting"]
    value = payload.value.strip() if isinstance(payload.value, str) else payload.value
    set_runtime_setting(setting_name, value or None, target=settings)

    if setting_name == "GEMINI_API_KEY":
        try:
            manager = api.get_session_manager()
            refresh = getattr(manager, "refresh_credentials", None)
            if refresh:
                refresh()
        except Exception as e:
            logger.warning("Failed to refresh web session credentials: %s", e)

    return {"ok": True, "secret": _secret_status(settings, secret_key)}


@router.post("/secrets/{secret_key}/test")
async def test_secret(secret_key: str):
    """Validate a supported local secret without returning its value."""
    if secret_key not in SECRET_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Unknown secret '{secret_key}'")

    api = _get_api()
    settings = api.settings
    setting_name = SECRET_DEFINITIONS[secret_key]["setting"]
    value = str(getattr(settings, setting_name, None) or os.environ.get(setting_name, "") or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Save this secret before testing it.")

    if secret_key == "gemini_api_key":
        return await _test_gemini_key(value)
    if secret_key == "openrouter_api_key":
        return await _test_openrouter_key(value)
    if secret_key == "telegram_bot_token":
        return await _test_telegram_token(value)
    raise HTTPException(status_code=400, detail="This secret cannot be tested directly from Settings.")


async def _test_gemini_key(api_key: str) -> dict:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key},
            )
        if resp.status_code == 200:
            return {"ok": True, "message": "Gemini key works."}
        return {"ok": False, "message": _provider_status_message("Gemini", resp.status_code)}
    except Exception as exc:
        return {"ok": False, "message": _provider_exception_message("Gemini", exc)}


async def _test_openrouter_key(api_key: str) -> dict:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            return {"ok": True, "message": "OpenRouter key works."}
        return {"ok": False, "message": _provider_status_message("OpenRouter", resp.status_code)}
    except Exception as exc:
        return {"ok": False, "message": _provider_exception_message("OpenRouter", exc)}


async def _test_telegram_token(token: str) -> dict:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
        if resp.status_code == 200:
            return {"ok": True, "message": "Telegram bot token works."}
        return {"ok": False, "message": _provider_status_message("Telegram", resp.status_code)}
    except Exception as exc:
        return {"ok": False, "message": _provider_exception_message("Telegram", exc)}


def _provider_status_message(provider: str, status_code: int) -> str:
    if status_code in {401, 403}:
        return f"{provider} rejected this secret. Check that the key/token is correct."
    if status_code == 404:
        return f"{provider} test endpoint was not found. Check provider settings."
    if status_code >= 500:
        return f"{provider} is returning a server error. Retry later."
    return f"{provider} test failed with HTTP {status_code}."


def _provider_exception_message(provider: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if "timeout" in detail.lower() or "timed out" in detail.lower():
        return f"{provider} test timed out. Check your internet connection and retry."
    return f"{provider} test could not connect. Check your internet connection. Detail: {detail or exc.__class__.__name__}"


@router.put("/settings")
async def update_settings(payload: SettingsPayload):
    """Update runtime settings without rewriting .env."""
    from remy.config.settings import set_runtime_setting

    api = _get_api()
    settings = api.settings

    updated = []

    if payload.gemini_api_key is not None:
        set_runtime_setting("GEMINI_API_KEY", payload.gemini_api_key, target=settings)
        try:
            manager = api.get_session_manager()
            refresh = getattr(manager, "refresh_credentials", None)
            if refresh:
                refresh()
        except Exception as e:
            logger.warning("Failed to refresh web session credentials: %s", e)
        updated.append("GEMINI_API_KEY")

    if payload.openrouter_api_key is not None:
        set_runtime_setting("OPENROUTER_API_KEY", payload.openrouter_api_key, target=settings)
        updated.append("OPENROUTER_API_KEY")

    if payload.summary_model is not None:
        set_runtime_setting("SUMMARY_MODEL", payload.summary_model, target=settings)
        updated.append("SUMMARY_MODEL")

    if payload.web_search_model is not None:
        set_runtime_setting("WEB_SEARCH_MODEL", payload.web_search_model, target=settings)
        updated.append("WEB_SEARCH_MODEL")

    if payload.gemini_model is not None:
        set_runtime_setting("GEMINI_MODEL", payload.gemini_model, target=settings)
        updated.append("GEMINI_MODEL")

    if payload.gemini_voice is not None:
        set_runtime_setting("GEMINI_VOICE", payload.gemini_voice, target=settings)
        updated.append("GEMINI_VOICE")

    if payload.telegram_bot_token is not None:
        set_runtime_setting("TELEGRAM_BOT_TOKEN", payload.telegram_bot_token, target=settings)
        updated.append("TELEGRAM_BOT_TOKEN")

    if payload.proactive_chat_id is not None:
        set_runtime_setting("PROACTIVE_CHAT_ID", payload.proactive_chat_id, target=settings)
        updated.append("PROACTIVE_CHAT_ID")

    if payload.review_model is not None:
        set_runtime_setting("REVIEW_MODEL", payload.review_model, target=settings)
        updated.append("REVIEW_MODEL")

    if payload.browser_vision_model is not None:
        set_runtime_setting("BROWSER_VISION_MODEL", payload.browser_vision_model, target=settings)
        updated.append("BROWSER_VISION_MODEL")

    if payload.browser_backend is not None:
        set_runtime_setting("BROWSER_BACKEND", payload.browser_backend, target=settings)
        updated.append("BROWSER_BACKEND")

    if payload.pinchtab_enabled is not None:
        set_runtime_setting("PINCHTAB_ENABLED", payload.pinchtab_enabled, target=settings)
        updated.append("PINCHTAB_ENABLED")

    if payload.pinchtab_base_url is not None:
        set_runtime_setting("PINCHTAB_BASE_URL", payload.pinchtab_base_url, target=settings)
        updated.append("PINCHTAB_BASE_URL")

    if payload.pinchtab_timeout_sec is not None:
        set_runtime_setting("PINCHTAB_TIMEOUT_SEC", payload.pinchtab_timeout_sec, target=settings)
        updated.append("PINCHTAB_TIMEOUT_SEC")

    if payload.pinchtab_profile_id is not None:
        set_runtime_setting("PINCHTAB_PROFILE_ID", payload.pinchtab_profile_id, target=settings)
        updated.append("PINCHTAB_PROFILE_ID")

    if payload.pinchtab_autostart is not None:
        set_runtime_setting("PINCHTAB_AUTOSTART", payload.pinchtab_autostart, target=settings)
        updated.append("PINCHTAB_AUTOSTART")

    if payload.pinchtab_command is not None:
        set_runtime_setting("PINCHTAB_COMMAND", payload.pinchtab_command, target=settings)
        updated.append("PINCHTAB_COMMAND")

    if payload.pinchtab_binary_path is not None:
        set_runtime_setting("PINCHTAB_BINARY_PATH", payload.pinchtab_binary_path, target=settings)
        updated.append("PINCHTAB_BINARY_PATH")

    if payload.pinchtab_bootstrap_source is not None:
        set_runtime_setting("PINCHTAB_BOOTSTRAP_SOURCE", payload.pinchtab_bootstrap_source, target=settings)
        updated.append("PINCHTAB_BOOTSTRAP_SOURCE")

    if payload.custom_system_prompt is not None:
        prompt_path = settings.DATA_DIR / "custom_system_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(payload.custom_system_prompt, encoding="utf-8")
        updated.append("CUSTOM_SYSTEM_PROMPT")

    if payload.smtp_host is not None:
        set_runtime_setting("SMTP_HOST", payload.smtp_host, target=settings)
        updated.append("SMTP_HOST")

    if payload.smtp_port is not None:
        set_runtime_setting("SMTP_PORT", payload.smtp_port, target=settings)
        updated.append("SMTP_PORT")

    if payload.smtp_user is not None:
        set_runtime_setting("SMTP_USER", payload.smtp_user, target=settings)
        updated.append("SMTP_USER")

    if payload.smtp_password is not None:
        set_runtime_setting("SMTP_PASSWORD", payload.smtp_password, target=settings)
        updated.append("SMTP_PASSWORD")

    if payload.smtp_from is not None:
        set_runtime_setting("SMTP_FROM", payload.smtp_from, target=settings)
        updated.append("SMTP_FROM")

    return {
        "updated": updated,
        "note": "Changes apply immediately and are saved to data/runtime_settings.json.",
    }


# ============== HISTORY ==============


@router.get("/history")
async def list_history():
    """List past session logs."""
    api = _get_api()
    history_dir = api.settings.DATA_DIR / "history"
    if not history_dir.exists():
        return {"sessions": []}

    sessions = []
    for f in history_dir.glob("*.json"):
        try:
            stat = f.stat()
            sessions.append(
                {
                    "filename": f.name,
                    "timestamp": stat.st_mtime,
                    "size": stat.st_size,
                    "date_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
        except Exception:
            pass

    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"sessions": sessions}


@router.get("/history/{filename}")
async def get_history_session(filename: str):
    """Get specific session log."""
    api = _get_api()
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    history_dir = api.settings.DATA_DIR / "history"
    filepath = history_dir / safe_name

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="History not found")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load history: {e}")


@router.delete("/history/{filename}")
async def delete_history_session(filename: str):
    """Delete a specific session log file."""
    api = _get_api()
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    history_dir = api.settings.DATA_DIR / "history"
    filepath = history_dir / safe_name

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="History not found")

    try:
        filepath.unlink()
        return {"ok": True, "deleted": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@router.delete("/history")
async def clear_history():
    """Delete all session log files."""
    api = _get_api()
    history_dir = api.settings.DATA_DIR / "history"
    if not history_dir.exists():
        return {"ok": True, "deleted": 0}

    count = 0
    for f in history_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return {"ok": True, "deleted": count}


# ============== SANDBOX TOOLS ==============


@router.get("/sandbox/tools")
async def list_sandbox_tools():
    """List all sandbox tools and their status."""
    api = _get_api()
    from remy.sandbox.manifest import SandboxManifest

    manifest = SandboxManifest(api.settings.SANDBOX_DIR / "manifest.json")
    return {"tools": manifest.summary()}


@router.put("/sandbox/tools/{tool_name}/toggle")
async def toggle_sandbox_tool(tool_name: str):
    """Toggle a sandbox tool between approved and rejected."""
    api = _get_api()
    from remy.core.brain_tools import reload_tools
    from remy.sandbox.manifest import SandboxManifest

    manifest = SandboxManifest(api.settings.SANDBOX_DIR / "manifest.json")

    tool = manifest.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    current = tool["status"]
    if current == "approved":
        tool["status"] = "rejected"
        manifest.save()
        reload_tools()
        return {"name": tool_name, "status": "rejected", "note": "Tool deactivated."}
    elif current == "rejected":
        tool["status"] = "approved"
        manifest.save()
        reload_tools()
        return {"name": tool_name, "status": "approved", "note": "Tool reactivated."}
    elif current in ("tested", "pending"):
        tool["status"] = "approved"
        manifest.save()
        reload_tools()
        return {"name": tool_name, "status": "approved", "note": "Tool approved and loaded."}
    else:
        return {
            "name": tool_name,
            "status": current,
            "note": f"Cannot toggle from '{current}' status.",
        }


# ============== LOCAL WORKSPACES ==============


def _workspace_permissions(payload: WorkspaceGrantPayload | WorkspaceUpdatePayload) -> set[str]:
    permissions = set()
    if payload.read:
        permissions.add("read")
    if payload.write:
        permissions.add("write")
    if payload.execute:
        permissions.add("execute")
    if not permissions:
        raise HTTPException(status_code=422, detail="Select at least one permission.")
    return permissions


@router.get("/workspaces")
async def list_local_workspaces():
    from remy.core.workspace_permissions import get_workspace_manager

    return {"workspaces": get_workspace_manager().list_grants()}


@router.get("/workspaces/audit")
async def list_workspace_audit(limit: int = 100):
    from remy.core.workspace_permissions import get_workspace_manager

    return {"events": get_workspace_manager().read_audit(limit)}


@router.post("/workspaces")
async def add_local_workspace(payload: WorkspaceGrantPayload):
    from remy.core.workspace_permissions import get_workspace_manager

    if not payload.path:
        raise HTTPException(status_code=422, detail="path is required")
    try:
        grant = get_workspace_manager().add_grant(
            payload.path,
            name=payload.name or "",
            permissions=_workspace_permissions(payload),
        )
        return {"workspace": grant}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspaces/select-folder")
async def select_local_workspace(payload: WorkspaceGrantPayload):
    """Open the native folder chooser on the machine running Remy."""
    from remy.core.workspace_permissions import get_workspace_manager

    def choose_folder() -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            raise RuntimeError("The native folder picker is unavailable. Enter the path manually.") from exc
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.update()
            return filedialog.askdirectory(title="Choose a folder for Remy", mustexist=True) or ""
        finally:
            root.destroy()

    try:
        # Native picker duration is controlled by the user, not by an API
        # operation deadline. Wait until they select a folder or cancel.
        selected = await run_in_thread(choose_folder, timeout=None)
        if not selected:
            return {"cancelled": True}
        grant = get_workspace_manager().add_grant(
            selected,
            name=payload.name or "",
            permissions=_workspace_permissions(payload),
        )
        return {"cancelled": False, "workspace": grant}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.patch("/workspaces/{workspace_id}")
async def update_local_workspace(workspace_id: str, payload: WorkspaceUpdatePayload):
    from remy.core.workspace_permissions import get_workspace_manager

    try:
        grant = get_workspace_manager().update_grant(
            workspace_id,
            name=payload.name,
            permissions=_workspace_permissions(payload),
        )
        return {"workspace": grant}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/workspaces/{workspace_id}")
async def revoke_local_workspace(workspace_id: str):
    from remy.core.workspace_permissions import get_workspace_manager

    try:
        return {"workspace": get_workspace_manager().revoke_grant(workspace_id), "revoked": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ============== PERSONA ==============


class PersonaPayload(BaseModel):
    name: str | None = None
    role: str | None = None
    scope: str | None = None
    tone: str | None = None
    formality: str | None = None
    languages: str | None = None
    motivations: str | None = None
    catchphrases: str | None = None
    avoid: str | None = None
    warmth: float | None = None
    curiosity: float | None = None
    conciseness: float | None = None
    humor: float | None = None
    formality_trait: float | None = None
    reset: bool = False


@router.get("/persona")
async def get_persona():
    """Return current agent persona."""
    import asyncio

    from remy.core.tool_handlers.profile import _get_agent_persona

    def _query():
        return _get_agent_persona()

    persona = await run_in_thread(_query)
    return persona


@router.put("/persona")
async def update_persona(payload: PersonaPayload):
    """Update agent persona fields. Set reset=true to restore defaults."""
    import asyncio

    from remy.core.tool_handlers.profile import update_persona_fields

    updates = {}
    for field in ("name", "role", "scope", "tone", "formality", "languages", "motivations"):
        val = getattr(payload, field)
        if val is not None:
            updates[field] = val

    # List fields (comma-separated strings from frontend)
    for field in ("catchphrases", "avoid"):
        val = getattr(payload, field)
        if val is not None:
            updates[field] = val

    # Traits — map formality_trait to the "formality" key inside traits
    traits = {}
    for trait in ("warmth", "curiosity", "conciseness", "humor"):
        val = getattr(payload, trait)
        if val is not None:
            traits[trait] = val
    if payload.formality_trait is not None:
        traits["formality"] = payload.formality_trait
    if traits:
        updates["traits"] = traits

    def _update():
        return update_persona_fields(updates, channel="desktop", reset=payload.reset)

    persona = await run_in_thread(_update)
    return {"updated": True, "persona": persona}
