import asyncio


def test_shutdown_cleanup_stops_scheduler_before_closing_brain(monkeypatch):
    import remy.web.api as api

    order = []

    class FakeManager:
        async def close_session(self):
            order.append("session")

    class FakeScheduler:
        async def stop(self):
            order.append("scheduler")

    monkeypatch.setattr(api, "get_session_manager", lambda: FakeManager())
    monkeypatch.setattr(api, "_scheduler", FakeScheduler(), raising=False)
    monkeypatch.setattr("remy.core.agent_tools.close_brain", lambda: order.append("brain"))

    asyncio.run(api.shutdown_cleanup())

    assert order == ["session", "scheduler", "brain"]
