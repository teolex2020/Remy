from remy.core import model_registry


def test_registered_nvidia_provider_wins_over_slash_heuristic(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "load_registry",
        lambda: {
            "deepseek-ai/deepseek-v4-flash": {
                "provider": "nvidia",
                "api_key": "nvapi-test",
            }
        },
    )

    assert model_registry.detect_provider("deepseek-ai/deepseek-v4-flash") == "deepseek"
    assert model_registry.get_provider_for_model("deepseek-ai/deepseek-v4-flash") == "nvidia"


def test_unregistered_slash_model_remains_openrouter(monkeypatch):
    monkeypatch.setattr(model_registry, "load_registry", lambda: {})

    assert model_registry.get_provider_for_model("anthropic/claude-sonnet-4-5") == "openrouter"
