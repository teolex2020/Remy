"""PinchTab binary discovery and best-effort bootstrap."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from remy.config.settings import settings

logger = logging.getLogger("PinchTabBootstrap")


def _binary_name() -> str:
    return "pinchtab.exe" if os.name == "nt" else "pinchtab"


def _is_remote(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _mark_executable(path: Path) -> None:
    try:
        current = path.stat().st_mode
        path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _explicit_binary_path() -> Path | None:
    value = (settings.PINCHTAB_BINARY_PATH or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if path.exists() and path.is_file():
        return path
    return None


def _vendor_candidates() -> list[Path]:
    binary = _binary_name()
    return [
        settings.BASE_DIR / "vendor" / "pinchtab" / binary,
        settings.BASE_DIR / "vendor" / binary,
        settings.PINCHTAB_INSTALL_DIR / binary,
        settings.PINCHTAB_INSTALL_DIR / "bin" / binary,
    ]


def resolve_pinchtab_binary() -> Path | None:
    explicit = _explicit_binary_path()
    if explicit:
        return explicit

    command = (settings.PINCHTAB_COMMAND or "").strip()
    if command:
        first = command.split(maxsplit=1)[0].strip("\"'")
        if first:
            path = Path(first)
            if path.exists() and path.is_file():
                return path
            discovered = shutil.which(first)
            if discovered:
                return Path(discovered)

    for candidate in _vendor_candidates():
        if candidate.exists() and candidate.is_file():
            return candidate

    discovered = shutil.which("pinchtab")
    if discovered:
        return Path(discovered)
    return None


def _extract_archive(archive_path: Path, install_dir: Path) -> Path | None:
    binary_name = _binary_name()
    install_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(binary_name)]
            if not members:
                return None
            member = members[0]
            target = install_dir / binary_name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            _mark_executable(target)
            return target

    suffixes = "".join(archive_path.suffixes[-2:]).lower()
    if suffixes in {".tar.gz", ".tgz"} or archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(binary_name)]
            if not members:
                return None
            member = members[0]
            target = install_dir / binary_name
            extracted = tf.extractfile(member)
            if extracted is None:
                return None
            with extracted as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            _mark_executable(target)
            return target

    return None


def _download_archive(source: str, target: Path) -> Path | None:
    timeout = max(5, int(settings.PINCHTAB_TIMEOUT_SEC))
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(source)
            response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return target
    except Exception as exc:
        logger.warning("PinchTab download failed: %s", exc)
        return None


def bootstrap_pinchtab_binary() -> Path | None:
    binary = resolve_pinchtab_binary()
    if binary:
        return binary

    source = (settings.PINCHTAB_BOOTSTRAP_SOURCE or "").strip() or (settings.PINCHTAB_RELEASE_URL or "").strip()
    if not source:
        return None

    install_dir = settings.PINCHTAB_INSTALL_DIR
    install_dir.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(source).path).name or "pinchtab.zip"
    archive_path = install_dir / archive_name

    if _is_remote(source):
        downloaded = _download_archive(source, archive_path)
        if downloaded is None:
            return None
    else:
        candidate = Path(source).expanduser()
        if not candidate.exists() or not candidate.is_file():
            logger.info("PinchTab bootstrap source not found; optional backend remains disabled: %s", source)
            return None
        archive_path = candidate

    try:
        binary = _extract_archive(archive_path, install_dir)
        if binary:
            logger.info("PinchTab binary bootstrapped to %s", binary)
            return binary
        logger.info("PinchTab archive did not contain a runnable binary; using Playwright fallback.")
        return None
    except Exception as exc:
        logger.info("PinchTab bootstrap extraction failed; using Playwright fallback: %s", exc)
        return None


def build_pinchtab_command() -> list[str]:
    command = (settings.PINCHTAB_COMMAND or "").strip()
    if command:
        parts = shlex.split(command, posix=False)
        first = parts[0].strip("\"'")
        if Path(first).exists() or shutil.which(first):
            return parts

    binary = resolve_pinchtab_binary() or bootstrap_pinchtab_binary()
    if binary is None:
        return []

    parsed = urlparse(settings.PINCHTAB_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return [str(binary), "serve", "--host", host, "--port", str(port)]
