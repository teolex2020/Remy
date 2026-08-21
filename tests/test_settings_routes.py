import pytest

from remy.core import llama_cpp_service, model_registry
from remy.web.routes import settings_routes


@pytest.mark.asyncio
async def test_openrouter_secret_test_returns_success_without_echoing_key(monkeypatch):
    import httpx

    captured = {}

    class _Response:
        status_code = 200

    class _HttpClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, headers=None, **_kwargs):
            captured["url"] = url
            captured["headers"] = headers or {}
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _HttpClient)

    result = await settings_routes._test_openrouter_key("sk-test-secret")

    assert result == {"ok": True, "message": "OpenRouter key works."}
    assert captured["headers"]["Authorization"] == "Bearer sk-test-secret"
    assert "sk-test-secret" not in str(result)


@pytest.mark.asyncio
async def test_available_models_include_only_registered_and_local_models(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "list_registered_models",
        lambda: [
            {"name": "moonshotai/kimi-k3", "provider": "openrouter", "has_key": True},
            {"name": "unconfigured/model", "provider": "openrouter", "has_key": False},
        ],
    )
    monkeypatch.setattr(
        llama_cpp_service.llama_cpp_service,
        "list_models",
        lambda: [
            {
                "name": "llamacpp:gemma-local",
                "filename": "gemma-local.gguf",
                "size_gb": 4.23,
            }
        ],
    )

    result = await settings_routes.list_available_models()

    assert [item["name"] for item in result["models"]] == [
        "moonshotai/kimi-k3",
        "llamacpp:gemma-local",
    ]
    assert all("llama-3.3-70b-instruct:free" not in item["name"] for item in result["models"])
