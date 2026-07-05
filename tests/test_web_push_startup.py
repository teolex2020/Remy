def test_load_subscription_defers_until_brain_is_initialized(monkeypatch):
    import remy.web.push as push

    monkeypatch.setattr("remy.core.agent_tools.brain_is_initialized", lambda: False)

    push._subscription = None
    push.load_subscription()

    assert push.get_subscription() is None
