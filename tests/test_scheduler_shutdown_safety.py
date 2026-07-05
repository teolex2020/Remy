import asyncio


def test_full_maintenance_skips_when_brain_shutdown_started(monkeypatch):
    import remy.web.scheduler as scheduler_mod

    scheduler = scheduler_mod.Scheduler()
    monkeypatch.setattr(scheduler_mod, "brain_runtime_allows_access", lambda: False)

    called = []
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *args, **kwargs: called.append("to_thread"),
    )

    asyncio.run(scheduler._run_full_maintenance())

    assert called == []
