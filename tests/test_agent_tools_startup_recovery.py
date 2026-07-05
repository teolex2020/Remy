from contextlib import nullcontext


def test_persist_startup_recovery_artifact_stores_record_and_emits_alert(monkeypatch):
    import remy.core.agent_tools as agent_tools

    stored = []
    emitted = []

    class FakeBrain:
        def store(self, content, level=None, tags=None, metadata=None):
            stored.append(
                {
                    "content": content,
                    "level": level,
                    "tags": list(tags or []),
                    "metadata": metadata or {},
                }
            )
            return type("Record", (), {"id": "startup-artifact-1"})()

    monkeypatch.setattr(agent_tools, "brain", FakeBrain(), raising=False)
    monkeypatch.setattr(agent_tools, "brain_lock", nullcontext(), raising=False)
    monkeypatch.setattr(agent_tools, "_brain_quarantined_at_startup", True, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_quarantine_reason", "failed to fill whole buffer", raising=False)
    monkeypatch.setattr(agent_tools, "_brain_quarantine_path", "data/brain_incompatible_20260331_223805", raising=False)
    monkeypatch.setattr(agent_tools, "_brain_backup_path", "data/brain_backup_20260331_223805.json", raising=False)
    monkeypatch.setattr(agent_tools, "_brain_recovery_stats", {}, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_startup_artifact_id", "", raising=False)
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    artifact_id = agent_tools._persist_startup_recovery_artifact(
        {
            "status": "history_replayed",
            "files": 45,
            "entries": 1177,
            "tool_calls_replayed": 129,
            "tool_calls_skipped": 17,
            "existing_records_before": 0,
        }
    )

    assert artifact_id == "startup-artifact-1"
    assert stored
    assert "incident_snapshot" in stored[0]["tags"]
    assert "startup_incident" in stored[0]["tags"]
    assert stored[0]["metadata"]["failure_code"] == "memory_recovery_applied"
    assert stored[0]["metadata"]["recovery"]["tool_calls_replayed"] == 129
    assert emitted
    assert emitted[0]["event_type"] == "operator_alert"
    assert emitted[0]["event_data"]["source"] == "startup"
    assert emitted[0]["event_data"]["artifact_ids"] == ["startup-artifact-1"]
    assert emitted[0]["event_data"]["verification_status"] == "recovered_after_quarantine"


def test_persist_startup_recovery_artifact_is_idempotent(monkeypatch):
    import remy.core.agent_tools as agent_tools

    calls = []

    class FakeBrain:
        def store(self, content, level=None, tags=None, metadata=None):
            calls.append(content)
            return type("Record", (), {"id": "startup-artifact-1"})()

    monkeypatch.setattr(agent_tools, "brain", FakeBrain(), raising=False)
    monkeypatch.setattr(agent_tools, "brain_lock", nullcontext(), raising=False)
    monkeypatch.setattr(agent_tools, "_brain_quarantined_at_startup", True, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_quarantine_reason", "failed to fill whole buffer", raising=False)
    monkeypatch.setattr(agent_tools, "_brain_quarantine_path", "data/brain_incompatible_20260331_223805", raising=False)
    monkeypatch.setattr(agent_tools, "_brain_backup_path", "data/brain_backup_20260331_223805.json", raising=False)
    monkeypatch.setattr(agent_tools, "_brain_recovery_stats", {}, raising=False)
    monkeypatch.setattr(agent_tools, "_brain_startup_artifact_id", "", raising=False)
    monkeypatch.setattr("remy.core.notification_router.notify", lambda *args, **kwargs: None)

    first = agent_tools._persist_startup_recovery_artifact({"status": "history_replayed"})
    second = agent_tools._persist_startup_recovery_artifact({"status": "history_replayed"})

    assert first == "startup-artifact-1"
    assert second == "startup-artifact-1"
    assert len(calls) == 1
