"""Managed llama.cpp runtime and local GGUF model library for Remy."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import httpx

logger = logging.getLogger("LlamaCppService")

_GITHUB_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
_HF_API = "https://huggingface.co/api/models"
_SERVER_NAME = "llama-server.exe" if platform.system().lower() == "windows" else "llama-server"
_MODEL_PREFIX = "llamacpp:"
_MAX_REPO_FILES = 5000


def _root() -> Path:
    from remy.config.settings import settings

    return settings.DATA_DIR / "llama_cpp"


def _runtime_dir() -> Path:
    return _root() / "runtime"


def _local_settings_path() -> Path:
    return _root() / "settings.json"


def _load_local_settings() -> dict:
    try:
        data = json.loads(_local_settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_local_settings(data: dict) -> None:
    path = _local_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _models_dir() -> Path:
    configured = str(_load_local_settings().get("models_dir") or "").strip()
    return Path(configured) if configured else _root() / "models"


def _manifest_path() -> Path:
    return _root() / "models.json"


def _base_url() -> str:
    from remy.config.settings import settings

    return settings.LLAMA_CPP_BASE_URL.rstrip("/")


def _api_headers(*, hugging_face: bool = False) -> dict[str, str]:
    headers = {"User-Agent": "Remy-local-model-manager"}
    token = os.environ.get("HF_TOKEN", "").strip() if hugging_face else ""
    if hugging_face and token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _safe_model_id(repo_id: str, filename: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:48]
    digest = hashlib.sha256(f"{repo_id}/{filename}".encode()).hexdigest()[:10]
    return f"{stem or 'gguf'}-{digest}"


class LlamaCppService:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._active_model = ""
        self._lock = threading.RLock()
        self._log_handle = None

    def find_binary(self) -> Path | None:
        found = shutil.which("llama-server")
        if found:
            return Path(found)
        direct = _runtime_dir() / _SERVER_NAME
        if direct.exists():
            return direct
        if _runtime_dir().exists():
            candidates = list(_runtime_dir().rglob(_SERVER_NAME))
            if candidates:
                return candidates[0]
        return None

    @staticmethod
    def _runtime_asset(assets: list[dict]) -> dict | None:
        system = platform.system().lower()
        arch = platform.machine().lower()
        names: list[str]
        if system == "windows" and arch in {"amd64", "x86_64"}:
            names = ["bin-win-vulkan-x64.zip", "bin-win-cpu-x64.zip"]
        elif system == "linux" and arch in {"amd64", "x86_64"}:
            names = ["bin-ubuntu-vulkan-x64.tar.gz", "bin-ubuntu-x64.tar.gz"]
        elif system == "linux" and arch in {"arm64", "aarch64"}:
            names = ["bin-ubuntu-arm64.tar.gz"]
        elif system == "darwin" and arch in {"arm64", "aarch64"}:
            names = ["bin-macos-arm64.tar.gz"]
        elif system == "darwin":
            names = ["bin-macos-x64.tar.gz"]
        else:
            names = []
        for suffix in names:
            for asset in assets:
                if str(asset.get("name") or "").endswith(suffix):
                    return asset
        return None

    async def install_runtime(self, progress_cb: Callable[[dict], None] | None = None) -> Path:
        existing = self.find_binary()
        if existing:
            return existing

        def report(message: str, pct: int = 0, phase: str = "runtime"):
            if progress_cb:
                progress_cb({"phase": phase, "message": message, "pct": pct})

        report("Finding the current llama.cpp release...", 1)
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(_GITHUB_LATEST, headers=_api_headers())
            release = response.raise_for_status().json()
            asset = self._runtime_asset(list(release.get("assets") or []))
            if not asset:
                raise RuntimeError("No compatible llama.cpp runtime was found for this computer")
            url = str(asset.get("browser_download_url") or "")
            name = str(asset.get("name") or "llama-runtime")
            total = int(asset.get("size") or 0)
            runtime = _runtime_dir()
            runtime.mkdir(parents=True, exist_ok=True)
            archive = runtime / f".{name}.part"
            report(f"Downloading {name}...", 3)
            async with client.stream("GET", url, headers=_api_headers(), timeout=600) as response:
                response.raise_for_status()
                received = 0
                with archive.open("wb") as handle:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        handle.write(chunk)
                        received += len(chunk)
                        if total:
                            pct = min(80, 3 + int(received / total * 77))
                            report("Downloading llama.cpp runtime...", pct)

        report("Extracting llama.cpp runtime...", 85)
        try:
            if name.endswith(".zip"):
                with zipfile.ZipFile(archive) as bundle:
                    root = _runtime_dir().resolve()
                    for info in bundle.infolist():
                        target = (_runtime_dir() / info.filename).resolve()
                        if target != root and root not in target.parents:
                            raise RuntimeError("Unsafe path in llama.cpp archive")
                    bundle.extractall(_runtime_dir())
            elif name.endswith((".tar.gz", ".tgz")):
                with tarfile.open(archive) as bundle:
                    bundle.extractall(_runtime_dir(), filter="data")
            else:
                raise RuntimeError("Unsupported llama.cpp release archive")
        finally:
            archive.unlink(missing_ok=True)
        binary = self.find_binary()
        if not binary:
            raise RuntimeError("llama.cpp runtime did not contain llama-server")
        if platform.system().lower() != "windows":
            binary.chmod(binary.stat().st_mode | 0o111)
        report("llama.cpp runtime is ready.", 100, "done")
        return binary

    async def repository_files(self, repo_id: str) -> list[dict]:
        repo_id = repo_id.strip().strip("/")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
            raise ValueError("Use a Hugging Face repository ID such as owner/model-GGUF")
        url = (
            f"{_HF_API}/{quote(repo_id, safe='/')}/tree/main"
            "?recursive=true&expand=false"
        )
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            headers = _api_headers(hugging_face=True)
            response = await client.get(url, headers=headers)
            if response.status_code == 401 and "Authorization" in headers:
                response = await client.get(url, headers=_api_headers())
            response.raise_for_status()
            siblings = list(response.json() or [])[:_MAX_REPO_FILES]
        files = []
        for item in siblings:
            name = str(item.get("path") or "")
            if name.lower().endswith(".gguf") and not name.startswith("."):
                files.append({"filename": name, "size": int(item.get("size") or 0)})
        files = sorted(files, key=lambda item: item["filename"].lower())
        for item in files:
            match = re.match(
                r"^(.*)-(\d{5})-of-(\d{5})\.gguf$",
                item["filename"],
                flags=re.IGNORECASE,
            )
            if not match:
                item["selectable"] = True
                continue
            prefix, part, total = match.groups()
            pattern = re.compile(
                rf"^{re.escape(prefix)}-\d{{5}}-of-{total}\.gguf$",
                re.IGNORECASE,
            )
            item["selectable"] = int(part) == 1
            item["shards"] = int(total)
            if item["selectable"]:
                item["size"] = sum(
                    int(candidate.get("size") or 0)
                    for candidate in files
                    if pattern.match(candidate["filename"])
                )
        return files

    @staticmethod
    def _selected_shards(filename: str, available: list[dict]) -> list[dict]:
        match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", filename, flags=re.IGNORECASE)
        if not match:
            return [item for item in available if item["filename"] == filename]
        prefix, _, total = match.groups()
        pattern = re.compile(rf"^{re.escape(prefix)}-\d{{5}}-of-{total}\.gguf$", re.IGNORECASE)
        return [item for item in available if pattern.match(item["filename"])]

    async def download_model(
        self,
        repo_id: str,
        filename: str,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> dict:
        available = await self.repository_files(repo_id)
        selected = self._selected_shards(filename, available)
        if not selected:
            raise ValueError("Selected GGUF file was not found in that repository")
        primary_filename = selected[0]["filename"]
        model_id = _safe_model_id(repo_id, primary_filename)
        model_dir = _models_dir() / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        total = sum(int(item.get("size") or 0) for item in selected)
        received_all = 0

        async with httpx.AsyncClient(timeout=900, follow_redirects=True) as client:
            for index, item in enumerate(selected, 1):
                remote_name = item["filename"]
                target = model_dir / Path(remote_name).name
                part = target.with_suffix(target.suffix + ".part")
                url = (
                    f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/main/"
                    f"{quote(remote_name, safe='/')}?download=true"
                )
                if progress_cb:
                    progress_cb({
                        "phase": "model",
                        "message": (
                            f"Downloading file {index}/{len(selected)}: "
                            f"{target.name}"
                        ),
                    })
                headers = _api_headers(hugging_face=True)

                async def stream_file(request_headers: dict[str, str]) -> None:
                    nonlocal received_all
                    async with client.stream("GET", url, headers=request_headers) as response:
                        response.raise_for_status()
                        with part.open("wb") as handle:
                            async for chunk in response.aiter_bytes(1024 * 1024):
                                handle.write(chunk)
                                received_all += len(chunk)
                                if progress_cb and total:
                                    progress_cb({
                                        "phase": "model",
                                        "message": (
                                            f"Downloading {received_all / 1024**3:.2f} / "
                                            f"{total / 1024**3:.2f} GB"
                                        ),
                                        "pct": int(received_all / total * 100),
                                    })

                try:
                    await stream_file(headers)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 401 or "Authorization" not in headers:
                        raise
                    received_all -= part.stat().st_size if part.exists() else 0
                    part.unlink(missing_ok=True)
                    await stream_file(_api_headers())
                part.replace(target)

        entry = {
            "id": model_id,
            "repo_id": repo_id,
            "filename": Path(primary_filename).name,
            "path": str(model_dir / Path(primary_filename).name),
            "size": sum(path.stat().st_size for path in model_dir.glob("*.gguf")),
            "source": "huggingface",
            "managed": True,
        }
        manifest = self._load_manifest()
        manifest[model_id] = entry
        self._save_manifest(manifest)
        if progress_cb:
            progress_cb({"phase": "done", "message": f"{entry['filename']} is ready.", "pct": 100})
        return entry

    def _folder_files(self, folder: str) -> list[dict]:
        """List selectable GGUF models in a folder without registering them."""
        root = Path(folder).expanduser()
        try:
            root = root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("The selected model folder does not exist") from exc
        if not root.is_dir():
            raise ValueError("Select a folder that contains GGUF models")

        candidates = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".gguf"),
            key=lambda path: str(path).lower(),
        )[:_MAX_REPO_FILES]
        manifest = self._load_manifest()
        known_paths = {
            str(Path(str(entry.get("path") or "")).resolve()).lower()
            for entry in manifest.values()
            if entry.get("path")
        }
        files: list[dict] = []
        for path in candidates:
            match = re.match(
                r"^(.*)-(\d{5})-of-(\d{5})\.gguf$",
                path.name,
                flags=re.IGNORECASE,
            )
            if match and int(match.group(2)) != 1:
                continue

            size = path.stat().st_size
            shard_count = 1
            if match:
                prefix, _, total = match.groups()
                pattern = re.compile(
                    rf"^{re.escape(prefix)}-\d{{5}}-of-{total}\.gguf$",
                    re.IGNORECASE,
                )
                shards = [
                    candidate
                    for candidate in candidates
                    if candidate.parent == path.parent and pattern.match(candidate.name)
                ]
                size = sum(candidate.stat().st_size for candidate in shards)
                shard_count = len(shards)

            relative = path.relative_to(root).as_posix()
            files.append({
                "relative_path": relative,
                "filename": path.name,
                "path": str(path),
                "size": size,
                "shards": shard_count,
                "added": str(path.resolve()).lower() in known_paths,
            })
        return files

    def local_folder_files(self) -> list[dict]:
        if not _load_local_settings().get("models_dir"):
            return []
        return self._folder_files(str(_models_dir()))

    def configure_models_folder(self, folder: str) -> list[dict]:
        """Set the user's GGUF library and return models available for selection."""
        selected = Path(folder).expanduser()
        try:
            selected = selected.resolve(strict=True)
        except OSError as exc:
            raise ValueError("The selected model folder does not exist") from exc
        if not selected.is_dir():
            raise ValueError("Select a folder for local GGUF models")

        settings_data = _load_local_settings()
        settings_data["models_dir"] = str(selected)
        _save_local_settings(settings_data)

        manifest = self._load_manifest()
        manifest = {
            model_id: entry
            for model_id, entry in manifest.items()
            if bool(entry.get("managed", True))
        }
        self._save_manifest(manifest)
        return self._folder_files(str(selected))

    def add_local_model(self, relative_path: str) -> dict:
        """Register one user-selected GGUF model from the configured folder."""
        if not _load_local_settings().get("models_dir"):
            raise ValueError("Choose a models folder first")
        candidates = {
            item["relative_path"]: item
            for item in self.local_folder_files()
        }
        selected = candidates.get(relative_path)
        if not selected:
            raise ValueError("The selected GGUF model was not found in the models folder")

        root = _models_dir().resolve()
        path = Path(selected["path"]).resolve()
        if root not in path.parents:
            raise ValueError("The selected model is outside the configured models folder")
        model_id = _safe_model_id(f"local:{root}", selected["relative_path"])
        entry = {
            "id": model_id,
            "repo_id": f"local:{root}",
            "filename": selected["filename"],
            "path": str(path),
            "size": selected["size"],
            "source": "local-folder",
            "source_folder": str(root),
            "managed": False,
            "shards": selected["shards"],
        }
        manifest = self._load_manifest()
        manifest[model_id] = entry
        self._save_manifest(manifest)
        return entry

    def _load_manifest(self) -> dict[str, dict]:
        try:
            return json.loads(_manifest_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, data: dict[str, dict]) -> None:
        _manifest_path().parent.mkdir(parents=True, exist_ok=True)
        _manifest_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_models(self) -> list[dict]:
        result = []
        dirty = False
        manifest = self._load_manifest()
        for model_id, entry in list(manifest.items()):
            path = Path(str(entry.get("path") or ""))
            if not path.exists():
                manifest.pop(model_id, None)
                dirty = True
                continue
            size = int(entry.get("size") or path.stat().st_size)
            result.append({
                **entry,
                "name": f"{_MODEL_PREFIX}{model_id}",
                "size_gb": round(size / 1024**3, 2),
            })
        if dirty:
            self._save_manifest(manifest)
        return sorted(result, key=lambda item: str(item.get("filename") or "").lower())

    def delete_model(self, model_id: str) -> bool:
        model_id = model_id.removeprefix(_MODEL_PREFIX).strip()
        manifest = self._load_manifest()
        entry = manifest.get(model_id)
        if not entry:
            return False
        if self._active_model == model_id:
            self.stop()
        if bool(entry.get("managed", True)):
            model_dir = Path(str(entry["path"])).parent
            root = _models_dir().resolve()
            resolved = model_dir.resolve()
            if resolved == root or root not in resolved.parents:
                raise ValueError("Refusing to delete a model outside the llama.cpp library")
            shutil.rmtree(resolved)
        manifest.pop(model_id, None)
        self._save_manifest(manifest)
        return True

    def _healthy(self) -> bool:
        try:
            return httpx.get(f"{_base_url()}/models", timeout=2).status_code == 200
        except Exception:
            return False

    def start_model(self, model_id: str) -> bool:
        model_id = model_id.removeprefix(_MODEL_PREFIX).strip()
        with self._lock:
            if self._active_model == model_id and self._healthy():
                return True
            manifest = self._load_manifest()
            entry = manifest.get(model_id)
            if not entry or not Path(str(entry.get("path") or "")).exists():
                raise ValueError("Local GGUF model is not installed")
            binary = self.find_binary()
            if not binary:
                raise RuntimeError(
                    "llama.cpp runtime is not installed. Install it in Settings first."
                )
            self.stop()
            log_path = _root() / "llama-server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("a", encoding="utf-8")
            port = int(_base_url().rsplit(":", 1)[-1].split("/", 1)[0])
            args = [
                str(binary), "--model", str(entry["path"]), "--alias", model_id,
                "--host", "127.0.0.1", "--port", str(port), "--jinja",
                "--ctx-size", "32768", "--n-gpu-layers", "99", "--parallel", "1",
            ]
            self._process = subprocess.Popen(
                args,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if platform.system().lower() == "windows"
                    else 0
                ),
            )
            self._active_model = model_id
        for _ in range(240):
            if self._healthy():
                return True
            if self._process and self._process.poll() is not None:
                break
            time.sleep(0.5)
        self.stop()
        log_path = _root() / "llama-server.log"
        raise RuntimeError(f"llama-server could not load the model. See {log_path}")

    def stop(self) -> None:
        with self._lock:
            if self._process:
                self._process.terminate()
                try:
                    self._process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
            self._active_model = ""

    def status(self) -> dict:
        local_settings = _load_local_settings()
        return {
            "runtime_installed": self.find_binary() is not None,
            "runtime_path": str(self.find_binary()) if self.find_binary() else None,
            "running": self._healthy(),
            "active_model": f"{_MODEL_PREFIX}{self._active_model}" if self._active_model else None,
            "models": self.list_models(),
            "models_dir": str(_models_dir()),
            "models_dir_configured": bool(local_settings.get("models_dir")),
            "local_files": self.local_folder_files(),
        }


llama_cpp_service = LlamaCppService()
