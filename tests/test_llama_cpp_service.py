from __future__ import annotations

import json
from pathlib import Path

import pytest

from remy.core import llama_cpp_service as module


def test_model_id_is_stable_and_safe():
    first = module._safe_model_id("owner/Model-GGUF", "Q4_K_M/model.gguf")
    second = module._safe_model_id("owner/Model-GGUF", "Q4_K_M/model.gguf")
    assert first == second
    assert first.startswith("model-")
    assert "/" not in first


def test_hugging_face_token_is_not_sent_to_github(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert "Authorization" not in module._api_headers()
    assert module._api_headers(hugging_face=True)["Authorization"] == "Bearer secret"


def test_sharded_gguf_download_selects_all_matching_parts():
    files = [
        {"filename": "model-Q4-00001-of-00003.gguf", "size": 1},
        {"filename": "model-Q4-00002-of-00003.gguf", "size": 1},
        {"filename": "model-Q4-00003-of-00003.gguf", "size": 1},
        {"filename": "model-Q8.gguf", "size": 2},
    ]
    selected = module.LlamaCppService._selected_shards(files[0]["filename"], files)
    assert [item["filename"] for item in selected] == [item["filename"] for item in files[:3]]


def test_sharded_selection_uses_first_part_as_primary():
    files = [
        {"filename": "model-Q4-00001-of-00002.gguf", "size": 1},
        {"filename": "model-Q4-00002-of-00002.gguf", "size": 1},
    ]
    selected = module.LlamaCppService._selected_shards(files[1]["filename"], files)
    assert selected[0]["filename"].endswith("00001-of-00002.gguf")


def test_manifest_lists_only_existing_models(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "_root", lambda: tmp_path)
    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)
    model = model_dir / "model.gguf"
    model.write_bytes(b"gguf")
    (tmp_path / "models.json").write_text(json.dumps({
        "test-model": {
            "id": "test-model", "repo_id": "owner/repo", "filename": "model.gguf",
            "path": str(model), "size": 4,
        },
        "missing": {
            "id": "missing", "repo_id": "owner/repo", "filename": "missing.gguf",
            "path": str(tmp_path / "missing.gguf"), "size": 4,
        },
    }), encoding="utf-8")

    models = module.LlamaCppService().list_models()

    assert len(models) == 1
    assert models[0]["name"] == "llamacpp:test-model"
    assert "missing" not in json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))


def test_delete_is_scoped_to_managed_model_library(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "_root", lambda: tmp_path)
    outside = tmp_path.parent / "outside.gguf"
    manifest = {"bad": {"id": "bad", "path": str(outside), "filename": outside.name}}
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "models.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        module.LlamaCppService().delete_model("bad")


def test_existing_local_models_are_linked_without_copying(monkeypatch, tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    first = library / "model-Q4_K_M.gguf"
    first.write_bytes(b"gguf-one")
    (library / "split-00001-of-00002.gguf").write_bytes(b"part-one")
    (library / "split-00002-of-00002.gguf").write_bytes(b"part-two")
    data_root = tmp_path / "remy-data"
    monkeypatch.setattr(module, "_root", lambda: data_root)

    service = module.LlamaCppService()
    discovered = service.configure_models_folder(str(library))

    assert len(discovered) == 2
    assert service.list_models() == []
    assert not (data_root / "models").exists()

    registered = service.add_local_model(discovered[0]["relative_path"])
    assert registered["managed"] is False
    assert Path(registered["path"]).parent == library
    assert service.delete_model(registered["id"]) is True
    assert first.exists()


def test_user_can_choose_models_storage_folder(monkeypatch, tmp_path):
    data_root = tmp_path / "remy-data"
    chosen = tmp_path / "chosen-models"
    chosen.mkdir()
    model = chosen / "chosen.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(module, "_root", lambda: data_root)

    service = module.LlamaCppService()
    discovered = service.configure_models_folder(str(chosen))

    assert module._models_dir() == chosen.resolve()
    assert discovered[0]["path"] == str(model.resolve())
    assert service.status()["models_dir_configured"] is True


def test_settings_expose_hugging_face_gguf_flow():
    settings_js = Path("src/remy/web/static/js/settings.js").read_text(encoding="utf-8")
    assert "Local Models (llama.cpp)" in settings_js
    assert "Browse GGUF models" in settings_js
    assert "/api/llamacpp/repository" in settings_js
    assert "/api/llamacpp/models/download" in settings_js
    assert "Choose models folder" in settings_js
    assert "/api/llamacpp/local/select-folder" in settings_js
    assert "Add to chat models" in settings_js
    assert "Use in chat" in settings_js
    assert "/api/llamacpp/local/model" in settings_js
    assert "Enter folder path manually" not in settings_js
    assert "models-changed" in settings_js

    chat_js = Path("src/remy/web/static/js/chat.js").read_text(encoding="utf-8")
    assert 'fetch("/api/models")' in chat_js
    assert 'document.addEventListener("models-changed"' in chat_js
