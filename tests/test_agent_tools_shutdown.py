from contextlib import nullcontext


def test_close_brain_marks_closed_on_expected_late_close_error(monkeypatch):
    import remy.core.agent_tools as agent_tools

    class FakeBrain:
        def close(self):
            raise RuntimeError("The system cannot find the file specified. (os error 2)")

    monkeypatch.setattr(agent_tools, "_brain_instance", FakeBrain(), raising=False)
    monkeypatch.setattr(agent_tools, "_brain_initialized", True, raising=False)
    monkeypatch.setattr(agent_tools, "brain_lock", nullcontext(), raising=False)
    monkeypatch.setattr(agent_tools, "_brain_closed", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_shutdown_started", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_close_error", "", raising=False)

    agent_tools.close_brain()

    assert agent_tools._brain_shutdown_started is True
    assert agent_tools._brain_closed is True
    assert "os error 2" in agent_tools._brain_close_error.lower()


def test_brain_runtime_allows_access_tracks_shutdown_flags(monkeypatch):
    import remy.core.agent_tools as agent_tools

    monkeypatch.setattr(agent_tools, "_brain_shutdown_started", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_closed", False, raising=False)
    assert agent_tools.brain_runtime_allows_access() is True

    monkeypatch.setattr(agent_tools, "_brain_shutdown_started", True, raising=False)
    assert agent_tools.brain_runtime_allows_access() is False

    monkeypatch.setattr(agent_tools, "_brain_shutdown_started", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_closed", True, raising=False)
    assert agent_tools.brain_runtime_allows_access() is False


def test_brain_initializes_lazily_on_first_use(monkeypatch):
    import remy.core.agent_tools as agent_tools

    calls = []

    class FakeBrain:
        def count(self):
            calls.append("count")
            return 7

    monkeypatch.setattr(agent_tools, "_brain_instance", None, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_initialized", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_closed", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_shutdown_started", False, raising=False)
    monkeypatch.setattr(agent_tools, "_init_brain", lambda: calls.append("init") or FakeBrain(), raising=False)
    monkeypatch.setattr(agent_tools, "_maybe_recover_brain_from_history", lambda: calls.append("recover"), raising=False)

    assert agent_tools.brain_is_initialized() is False

    assert agent_tools.brain.count() == 7
    assert calls == ["init", "recover", "count"]
    assert agent_tools.brain_is_initialized() is True


def test_close_brain_does_not_force_lazy_initialization(monkeypatch):
    import remy.core.agent_tools as agent_tools

    monkeypatch.setattr(agent_tools, "_brain_instance", None, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_initialized", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_closed", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_shutdown_started", False, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_close_error", "", raising=False)

    agent_tools.close_brain()

    assert agent_tools._brain_shutdown_started is True
    assert agent_tools._brain_closed is True
