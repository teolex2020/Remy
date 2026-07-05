"""
PinchTab sidecar lifecycle manager.

Best-effort autostart for the external PinchTab service. If startup fails,
Remy should continue to operate with the Playwright fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time

import httpx

from remy.config.settings import settings
from remy.core.pinchtab_bootstrap import build_pinchtab_command

logger = logging.getLogger("PinchTabService")


def _should_manage_pinchtab() -> bool:
    backend = (settings.BROWSER_BACKEND or "playwright").strip().lower()
    return (
        settings.PINCHTAB_ENABLED
        and backend in {"hybrid", "pinchtab"}
        and bool(settings.PINCHTAB_BASE_URL.strip())
    )

async def _check_health(timeout: float = 1.5) -> bool:
    base = settings.PINCHTAB_BASE_URL.rstrip("/")
    paths = ("/health", "/api/health", "/")
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in paths:
            try:
                resp = await client.get(f"{base}{path}")
                if resp.status_code < 500:
                    return True
            except Exception:
                continue
    return False


class PinchTabServiceManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._started_by_remy = False

    @classmethod
    def get(cls) -> "PinchTabServiceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    async def ensure_running(self) -> bool:
        if not _should_manage_pinchtab():
            return False
        if await _check_health():
            return True
        if not settings.PINCHTAB_AUTOSTART:
            logger.info("PinchTab autostart disabled; keeping Playwright fallback.")
            return False
        if self._process and self._process.poll() is None:
            return await self._wait_ready()
        return await self._start_process()

    async def _start_process(self) -> bool:
        parts = build_pinchtab_command()
        if not parts:
            logger.info("PinchTab binary unavailable; keeping Playwright fallback.")
            return False
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                parts,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._started_by_remy = True
            logger.info("Starting PinchTab sidecar: %s", " ".join(parts))
        except FileNotFoundError:
            logger.info("PinchTab binary not found for command; using Playwright fallback: %s", " ".join(parts))
            self._process = None
            self._started_by_remy = False
            return False
        except Exception as exc:
            logger.warning("PinchTab autostart failed: %s", exc)
            self._process = None
            self._started_by_remy = False
            return False
        return await self._wait_ready()

    async def _wait_ready(self) -> bool:
        deadline = time.time() + max(5, int(settings.PINCHTAB_TIMEOUT_SEC))
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                logger.warning("PinchTab sidecar exited before becoming ready.")
                self._process = None
                self._started_by_remy = False
                return False
            if await _check_health(timeout=2.0):
                logger.info("PinchTab sidecar ready.")
                return True
            await asyncio.sleep(0.5)
        logger.info("PinchTab sidecar did not become ready in time; using Playwright fallback.")
        return False

    async def shutdown(self) -> None:
        proc = self._process
        if not proc or not self._started_by_remy:
            self._process = None
            self._started_by_remy = False
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, 5)
                except Exception:
                    proc.kill()
        finally:
            self._process = None
            self._started_by_remy = False


async def ensure_pinchtab_running() -> bool:
    return await PinchTabServiceManager.get().ensure_running()


async def shutdown_pinchtab() -> None:
    await PinchTabServiceManager.get().shutdown()


def ensure_pinchtab_running_sync() -> bool:
    return asyncio.run(ensure_pinchtab_running())


def shutdown_pinchtab_sync() -> None:
    asyncio.run(shutdown_pinchtab())
