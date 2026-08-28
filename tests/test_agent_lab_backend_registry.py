import pytest

from remy.core.agent_lab_backend_registry import (
    AUTOMATIC_BACKEND,
    AgentLabBackendCapabilities,
    AgentLabBackendDescriptor,
    AgentLabBackendRegistration,
    AgentLabBackendRegistry,
    get_agent_lab_backend_registry,
    public_backend_receipt,
)


class ExampleBackend:
    mode = "example_isolate"

    def launch(self, request):
        return request

    def teardown(self, handle):
        return None


def test_default_registry_has_stable_order_and_unified_preflight():
    registry = get_agent_lab_backend_registry()
    assert registry.modes()[:2] == ["bounded_process", "container_required"]
    bounded = registry.preflight("bounded_process")
    assert bounded["available"] is True
    assert bounded["engine"] == "python-process"
    assert bounded["preparation_supported"] is False


def test_registry_accepts_new_backend_without_central_branching():
    registry = AgentLabBackendRegistry()
    registry.register(AgentLabBackendRegistration(
        descriptor=AgentLabBackendDescriptor(
            mode="example_isolate",
            label="Example isolate",
            description="Test backend",
            security_tier="test",
        ),
        factory=lambda **_kwargs: ExampleBackend(),
        probe=lambda **_kwargs: {"available": True, "engine": "example"},
    ))

    assert registry.modes() == ["example_isolate"]
    assert registry.preflight("example_isolate")["available"] is True
    assert registry.create("example_isolate").mode == "example_isolate"


def test_registry_rejects_duplicate_and_inconsistent_preparation_contracts():
    registry = AgentLabBackendRegistry()
    registration = AgentLabBackendRegistration(
        descriptor=AgentLabBackendDescriptor(
            mode="example_isolate",
            label="Example",
            description="Test",
            security_tier="test",
        ),
        factory=lambda **_kwargs: ExampleBackend(),
        probe=lambda **_kwargs: {"available": True},
    )
    registry.register(registration)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registration)
    with pytest.raises(ValueError, match="preparation contract"):
        registry.register(AgentLabBackendRegistration(
            descriptor=AgentLabBackendDescriptor(
                mode="broken_prepare",
                label="Broken",
                description="Broken",
                security_tier="test",
                preparation_supported=True,
            ),
            factory=lambda **_kwargs: ExampleBackend(),
            probe=lambda **_kwargs: {"available": False},
        ))


def test_public_receipt_removes_host_executable_and_environment():
    receipt = public_backend_receipt({
        "mode": "example_isolate",
        "available": True,
        "executable": "C:/private/runtime.exe",
        "environment": {"SECRET": "hidden"},
        "diagnostics": {"command": ["private.exe"], "engine": "example"},
    })
    assert receipt == {
        "mode": "example_isolate",
        "available": True,
        "diagnostics": {"engine": "example"},
    }


def test_preflight_failure_isolated_to_backend_and_fails_closed():
    registry = AgentLabBackendRegistry()
    registry.register(AgentLabBackendRegistration(
        descriptor=AgentLabBackendDescriptor(
            mode="example_isolate",
            label="Example",
            description="Test",
            security_tier="test",
        ),
        factory=lambda **_kwargs: ExampleBackend(),
        probe=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("probe exploded")),
    ))

    receipt = registry.preflight("example_isolate")
    assert receipt["available"] is False
    assert receipt["reason_code"] == "probe_failed"
    assert "probe exploded" in receipt["reason"]


def selection_registry():
    registry = AgentLabBackendRegistry()
    for mode, rank, available in (
        ("guarded", 10, True),
        ("hardened", 20, True),
    ):
        registry.register(AgentLabBackendRegistration(
            descriptor=AgentLabBackendDescriptor(
                mode=mode,
                label=mode.title(),
                description="Test",
                security_tier=mode,
                capabilities=AgentLabBackendCapabilities(isolation_rank=rank),
            ),
            factory=lambda **_kwargs: ExampleBackend(),
            probe=lambda available=available, **_kwargs: {"available": available},
        ))
    return registry


def test_automatic_selection_uses_least_privileged_sufficient_backend():
    registry = selection_registry()
    ordinary = registry.select(AUTOMATIC_BACKEND, {"minimum_isolation_rank": 10})
    hardened = registry.select(AUTOMATIC_BACKEND, {"minimum_isolation_rank": 20})

    assert ordinary["resolved_mode"] == "guarded"
    assert hardened["resolved_mode"] == "hardened"
    assert ordinary["selected_reason"] == "least_privileged_available_match"
    assert len(ordinary["candidates"]) == 2


def test_explicit_selection_never_falls_back_or_weakens_requirement():
    registry = selection_registry()
    with pytest.raises(ValueError, match="cannot start.*below required"):
        registry.select("guarded", {"minimum_isolation_rank": 20})


def test_selection_fails_closed_for_network_gpu_and_invalid_boolean_requirements():
    registry = selection_registry()
    with pytest.raises(ValueError, match="No registered.*network access"):
        registry.select(AUTOMATIC_BACKEND, {"network_required": True})
    with pytest.raises(ValueError, match="must be a boolean"):
        registry.select(AUTOMATIC_BACKEND, {"gpu_required": "false"})
