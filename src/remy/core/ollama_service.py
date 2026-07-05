"""
Ollama lifecycle manager — find, download, start, and stop Ollama automatically.

The user never needs to know about Ollama. Remy manages everything:
  1. Look for ollama binary (PATH, common install dirs, data/ollama/)
  2. If not found — download from official releases for current platform
  3. Start `ollama serve` as a child process
  4. On Remy shutdown — stop the child process

Usage:
    from remy.core.ollama_service import ollama_service
    await ollama_service.ensure_running()   # idempotent
    await ollama_service.pull_model("mistral", progress_cb=...)
    await ollama_service.delete_model("mistral")
    ollama_service.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

logger = logging.getLogger("OllamaService")

# ── Platform detection ────────────────────────────────────────────────────────

_SYSTEM = platform.system().lower()   # "windows" | "linux" | "darwin"
_ARCH   = platform.machine().lower()  # "amd64" | "x86_64" | "arm64" | "aarch64"

def _platform_key() -> str:
    arch = "arm64" if _ARCH in ("arm64", "aarch64") else "amd64"
    return f"{_SYSTEM}-{arch}"

# Official Ollama release URLs per platform
_OLLAMA_VERSION = "0.9.3"
_OLLAMA_URLS: dict[str, str] = {
    "windows-amd64": f"https://github.com/ollama/ollama/releases/download/v{_OLLAMA_VERSION}/ollama-windows-amd64.zip",
    "linux-amd64":   f"https://github.com/ollama/ollama/releases/download/v{_OLLAMA_VERSION}/ollama-linux-amd64.tgz",
    "linux-arm64":   f"https://github.com/ollama/ollama/releases/download/v{_OLLAMA_VERSION}/ollama-linux-arm64.tgz",
    "darwin-amd64":  f"https://github.com/ollama/ollama/releases/download/v{_OLLAMA_VERSION}/ollama-darwin",
    "darwin-arm64":  f"https://github.com/ollama/ollama/releases/download/v{_OLLAMA_VERSION}/ollama-darwin",
}

_BINARY_NAME = "ollama.exe" if _SYSTEM == "windows" else "ollama"

# Common install locations per platform
_SEARCH_PATHS: list[Path] = []
if _SYSTEM == "windows":
    _SEARCH_PATHS = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama",
        Path("C:/Program Files/Ollama"),
        Path(os.environ.get("USERPROFILE", "")) / ".ollama" / "bin",
    ]
elif _SYSTEM == "darwin":
    _SEARCH_PATHS = [
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
        Path(os.environ.get("HOME", "")) / ".ollama" / "bin",
    ]
else:
    _SEARCH_PATHS = [
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path(os.environ.get("HOME", "")) / ".ollama" / "bin",
    ]


# ── Ollama base URL ───────────────────────────────────────────────────────────

def _base_url() -> str:
    from remy.config.settings import settings
    return settings.OLLAMA_BASE_URL.rstrip("/")


def _models_dir() -> Path:
    """Where Remy-managed models live (OLLAMA_MODELS env override or default)."""
    from remy.config.settings import settings
    custom = os.environ.get("OLLAMA_MODELS", "")
    if custom:
        return Path(custom)
    return settings.DATA_DIR / "ollama" / "models"


def _binary_dir() -> Path:
    """Where we store the downloaded ollama binary."""
    from remy.config.settings import settings
    return settings.DATA_DIR / "ollama" / "bin"


# ── OllamaService ─────────────────────────────────────────────────────────────

class OllamaService:
    """Manages the Ollama process lifecycle for Remy."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._binary: Path | None = None
        self._started_by_us = False
        self._lock = threading.Lock()
        self._ready = False

    # ── Binary discovery & download ───────────────────────────────────────────

    def find_binary(self) -> Path | None:
        """Find ollama binary: PATH → common dirs → data/ollama/bin."""
        # 1. PATH
        found = shutil.which("ollama")
        if found:
            return Path(found)
        # 2. Common install dirs
        for d in _SEARCH_PATHS:
            candidate = d / _BINARY_NAME
            if candidate.exists():
                return candidate
        # 3. Our own downloaded copy
        own = _binary_dir() / _BINARY_NAME
        if own.exists():
            return own
        return None

    async def download_binary(self, progress_cb: Callable[[dict], None] | None = None) -> Path:
        """Download ollama binary for current platform into data/ollama/bin/."""
        key = _platform_key()
        url = _OLLAMA_URLS.get(key)
        if not url:
            raise RuntimeError(f"No Ollama release available for platform: {key}")

        dest_dir = _binary_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_bin = dest_dir / _BINARY_NAME

        def _report(msg: str, pct: int = 0):
            if progress_cb:
                try:
                    progress_cb({"phase": "download", "message": msg, "pct": pct})
                except Exception:
                    pass
            logger.info("Ollama download: %s", msg)

        _report(f"Downloading Ollama {_OLLAMA_VERSION} for {key}…", 0)

        tmp_path = Path(tempfile.mktemp(suffix=Path(url).suffix or ".bin"))
        try:
            # Stream download with progress
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    received = 0
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(65536):
                            f.write(chunk)
                            received += len(chunk)
                            if total:
                                pct = int(received / total * 80)
                                mb = received / 1024 / 1024
                                _report(f"Downloading… {mb:.0f} MB / {total/1024/1024:.0f} MB", pct)

            _report("Extracting…", 85)

            # Extract
            if url.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(tmp_path) as zf:
                    # Find the ollama binary inside the zip
                    for name in zf.namelist():
                        if name.endswith(_BINARY_NAME) or name == _BINARY_NAME:
                            with zf.open(name) as src, open(dest_bin, "wb") as dst:
                                dst.write(src.read())
                            break
                    else:
                        # Fallback: extract all, find binary
                        zf.extractall(dest_dir)
                        for p in dest_dir.rglob(_BINARY_NAME):
                            if p != dest_bin:
                                shutil.move(str(p), str(dest_bin))
                            break
            elif url.endswith(".tgz") or url.endswith(".tar.gz"):
                import tarfile
                with tarfile.open(tmp_path) as tf:
                    for member in tf.getmembers():
                        if member.name.endswith(_BINARY_NAME):
                            f = tf.extractfile(member)
                            if f:
                                dest_bin.write_bytes(f.read())
                            break
                    else:
                        tf.extractall(dest_dir)
            else:
                # Plain binary (macOS)
                shutil.copy2(tmp_path, dest_bin)

            # Make executable on Unix
            if _SYSTEM != "windows":
                dest_bin.chmod(dest_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

            _report("Ollama installed successfully.", 100)
            logger.info("Ollama binary installed at %s", dest_bin)
            return dest_bin

        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Process lifecycle ─────────────────────────────────────────────────────

    async def ensure_running(
        self,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> bool:
        """Ensure Ollama is running. Download binary if needed. Returns True if ready."""
        def _report(msg: str, phase: str = "start"):
            if progress_cb:
                try:
                    progress_cb({"phase": phase, "message": msg})
                except Exception:
                    pass

        # Already running (externally or by us)
        if await self._is_healthy():
            self._ready = True
            return True

        with self._lock:
            # Find or download binary
            binary = self.find_binary()
            if binary is None:
                _report("Ollama not found — downloading…", "download")
                try:
                    binary = await self.download_binary(progress_cb)
                except Exception as exc:
                    logger.error("Failed to download Ollama: %s", exc)
                    _report(f"Download failed: {exc}", "error")
                    return False

            self._binary = binary

            # Start ollama serve
            _report("Starting Ollama…", "start")
            env = os.environ.copy()
            # Point models dir to our data folder so models stay inside Remy
            models_dir = _models_dir()
            models_dir.mkdir(parents=True, exist_ok=True)
            env["OLLAMA_MODELS"] = str(models_dir)
            env["OLLAMA_HOST"] = "127.0.0.1:11434"

            try:
                self._process = subprocess.Popen(
                    [str(binary), "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if _SYSTEM == "windows" else 0,
                )
                self._started_by_us = True
                logger.info("Ollama started (pid=%d)", self._process.pid)
            except Exception as exc:
                logger.error("Failed to start Ollama: %s", exc)
                _report(f"Failed to start: {exc}", "error")
                return False

        # Wait for it to become healthy (up to 15s)
        for i in range(30):
            await asyncio.sleep(0.5)
            if await self._is_healthy():
                self._ready = True
                _report("Ollama is ready.", "ready")
                logger.info("Ollama is ready at %s", _base_url())
                return True

        logger.warning("Ollama did not become healthy within 15s")
        _report("Ollama did not start in time.", "error")
        return False

    async def _is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{_base_url()}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def shutdown(self):
        """Stop the Ollama process if we started it."""
        if self._process and self._started_by_us:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
                logger.info("Ollama stopped.")
            except Exception as exc:
                logger.warning("Ollama stop error: %s", exc)
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._started_by_us = False
            self._ready = False

    # ── Model management ──────────────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        """Return list of installed models with name and size."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{_base_url()}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    return [
                        {
                            "name": m["name"],
                            "size_gb": round(m.get("size", 0) / 1024**3, 1),
                            "modified": m.get("modified_at", ""),
                        }
                        for m in models
                    ]
        except Exception:
            pass
        return []

    async def pull_model(
        self,
        model_name: str,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> bool:
        """Download a model. Streams progress via progress_cb."""
        import json as _json

        def _report(msg: str, pct: int | None = None, status: str = "pulling"):
            if progress_cb:
                payload: dict = {"phase": status, "message": msg}
                if pct is not None:
                    payload["pct"] = pct
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        if not await self._is_healthy():
            ok = await self.ensure_running(progress_cb)
            if not ok:
                return False

        _report(f"Downloading {model_name}…", 0)
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream(
                    "POST",
                    f"{_base_url()}/api/pull",
                    json={"name": model_name, "stream": True},
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            evt = _json.loads(line)
                        except Exception:
                            continue
                        status = evt.get("status", "")
                        total = evt.get("total", 0)
                        completed = evt.get("completed", 0)
                        pct = int(completed / total * 100) if total else None
                        msg = status
                        if total and completed:
                            gb_done = completed / 1024**3
                            gb_total = total / 1024**3
                            msg = f"{status} — {gb_done:.2f} / {gb_total:.2f} GB"
                        _report(msg, pct, "pulling")
                        if status == "success":
                            _report(f"{model_name} installed successfully.", 100, "done")
                            return True
            return True
        except Exception as exc:
            logger.error("pull_model %s failed: %s", model_name, exc)
            _report(f"Download failed: {exc}", status="error")
            return False

    async def delete_model(self, model_name: str) -> bool:
        """Remove an installed model."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.request(
                    "DELETE",
                    f"{_base_url()}/api/delete",
                    json={"name": model_name},
                )
                return resp.status_code in (200, 204)
        except Exception as exc:
            logger.error("delete_model %s failed: %s", model_name, exc)
            return False

    def is_ready(self) -> bool:
        return self._ready

    async def status(self) -> dict:
        """Return service status for API."""
        healthy = await self._is_healthy()
        models = await self.list_models() if healthy else []
        binary = self.find_binary()
        return {
            "running": healthy,
            "managed": self._started_by_us,
            "binary_found": binary is not None,
            "binary_path": str(binary) if binary else None,
            "models": models,
            "models_dir": str(_models_dir()),
        }


# Singleton
ollama_service = OllamaService()
