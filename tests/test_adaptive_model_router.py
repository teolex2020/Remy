from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_simple_structural_task_prefers_cheapest_registered_model(monkeypatch):
    from remy.config.settings import settings
    from remy.core import adaptive_model_router as router

    monkeypatch.setattr(settings, "SUMMARY_MODEL", "default-model")
    monkeypatch.setattr(settings, "FALLBACK_MODELS", ["strong-model"])
    monkeypatch.setattr(settings, "MODEL_ROUTER_ENABLED", True)
    monkeypatch.setattr(
        router,
        "_model_price",
        lambda model: {"default-model": 5.0, "strong-model": 9.0, "cheap-model": 0.1}.get(model, 0.0),
    )
    monkeypatch.setattr(
        "remy.core.model_registry.list_registered_models",
        lambda: [{"name": "cheap-model", "provider": "openrouter", "has_key": True}],
    )

    routing = router.build_adaptive_model_routing(
        messages=[HumanMessage(content="Ok.")],
        channel="desktop",
        base_routing={},
    )

    assert routing["preferred_model"] == "cheap-model"
    assert routing["routing_source"] == "structural_low_cost_prior"
    assert routing["complexity_bucket"] == "simple"


def test_demanding_structural_task_prefers_highest_price_registered_model(monkeypatch):
    from remy.config.settings import settings
    from remy.core import adaptive_model_router as router

    text = "\n".join(
        [
            " ".join(f"unit{i}_{j}" for j in range(18))
            for i in range(7)
        ]
    )
    monkeypatch.setattr(settings, "SUMMARY_MODEL", "default-model")
    monkeypatch.setattr(settings, "FALLBACK_MODELS", ["cheap-model"])
    monkeypatch.setattr(settings, "MODEL_ROUTER_ENABLED", True)
    monkeypatch.setattr(
        router,
        "_model_price",
        lambda model: {"default-model": 1.0, "cheap-model": 0.1, "strong-model": 12.0}.get(model, 0.0),
    )
    monkeypatch.setattr(
        "remy.core.model_registry.list_registered_models",
        lambda: [{"name": "strong-model", "provider": "anthropic", "has_key": True}],
    )

    routing = router.build_adaptive_model_routing(
        messages=[HumanMessage(content=text)],
        channel="desktop",
        base_routing={},
    )

    assert routing["preferred_model"] == "strong-model"
    assert routing["routing_source"] == "structural_high_capability_prior"
    assert routing["complexity_bucket"] == "demanding"


def test_consequence_memory_preference_overrides_structural_price_prior(monkeypatch):
    from remy.config.settings import settings
    from remy.core import adaptive_model_router as router

    monkeypatch.setattr(settings, "SUMMARY_MODEL", "default-model")
    monkeypatch.setattr(settings, "FALLBACK_MODELS", ["cheap-model", "memory-good"])
    monkeypatch.setattr(settings, "MODEL_ROUTER_ENABLED", True)
    monkeypatch.setattr(router, "_model_price", lambda model: 0.0 if model == "cheap-model" else 10.0)
    monkeypatch.setattr("remy.core.model_registry.list_registered_models", lambda: [])

    routing = router.build_adaptive_model_routing(
        messages=[HumanMessage(content="Ok.")],
        channel="desktop",
        base_routing={"preferred_model": "memory-good", "avoid_models": ()},
    )

    assert routing["preferred_model"] == "memory-good"
    assert routing["routing_source"] == "consequence_memory"


def test_consequence_memory_avoid_blocks_structural_price_prior(monkeypatch):
    from remy.config.settings import settings
    from remy.core import adaptive_model_router as router

    monkeypatch.setattr(settings, "SUMMARY_MODEL", "default-model")
    monkeypatch.setattr(settings, "FALLBACK_MODELS", ["cheap-model"])
    monkeypatch.setattr(settings, "MODEL_ROUTER_ENABLED", True)
    monkeypatch.setattr(router, "_model_price", lambda model: 0.0 if model == "cheap-model" else 1.0)
    monkeypatch.setattr("remy.core.model_registry.list_registered_models", lambda: [])

    routing = router.build_adaptive_model_routing(
        messages=[HumanMessage(content="Ok.")],
        channel="desktop",
        base_routing={"avoid_models": ("cheap-model",)},
    )

    assert routing["preferred_model"] == "default-model"
    assert routing["avoid_models"] == ("cheap-model",)
    assert routing["routing_source"] == "structural_low_cost_prior"


def test_active_tool_context_increases_complexity_without_keywords():
    from remy.core.adaptive_model_router import estimate_task_complexity

    complexity = estimate_task_complexity(
        [
            HumanMessage(content="Do it."),
            AIMessage(content="", tool_calls=[{"id": "t1", "name": "x", "args": {}}]),
            ToolMessage(content="tool output", tool_call_id="t1"),
        ],
        channel="desktop",
    )

    assert complexity.score >= 2
    assert "active_tool_context" in complexity.reasons
