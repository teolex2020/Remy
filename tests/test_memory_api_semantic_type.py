from remy.core_v3.memory.memory_api import AuraMemoryBackend, MemoryClass
from remy.core_v3.memory.record_models import failure_record, goal_record, finding_record


class _DummyStored:
    def __init__(self, record_id="r1"):
        self.id = record_id


class _DummyAura:
    def __init__(self):
        self.store_calls = []
        self.search_results = []

    def store(self, content, tags=None, metadata=None, semantic_type=None):
        self.store_calls.append(
            {
                "content": content,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "semantic_type": semantic_type,
            }
        )
        return _DummyStored()

    def search(self, query=None, tags=None, limit=20, level=None):
        return self.search_results


class _DummyAuraWithConsequence(_DummyAura):
    def __init__(self):
        super().__init__()
        self.consequence_calls = []

    def capture_consequence(
        self,
        *,
        situation,
        action,
        consequence,
        trust=0,
        scope=None,
        provenance=None,
        links=None,
        namespace=None,
    ):
        self.consequence_calls.append(
            {
                "situation": situation,
                "action": action,
                "consequence": consequence,
                "trust": trust,
                "scope": list(scope or []),
                "provenance": list(provenance or []),
                "links": dict(links or {}),
                "namespace": namespace,
            }
        )
        return {"record_id": "cu1"}


class _DummyRecord:
    def __init__(self, *, content, tags, metadata, level, record_id="rec1"):
        self.id = record_id
        self.content = content
        self.tags = tags
        self.metadata = metadata
        self.level = level


def test_v3_memory_store_persists_semantic_type():
    aura = _DummyAura()
    backend = AuraMemoryBackend(aura)

    record_id = backend.store(
        "User prefers tea over coffee",
        tags=["preference"],
        metadata={},
        memory_class=MemoryClass.STRATEGIC,
    )

    assert record_id == "r1"
    call = aura.store_calls[0]
    assert call["semantic_type"] == "preference"
    assert call["metadata"]["semantic_type"] == "preference"
    assert call["metadata"]["memory_class"] == "strategic"


def test_v3_memory_capture_consequence_uses_sdk_when_available():
    aura = _DummyAuraWithConsequence()
    backend = AuraMemoryBackend(aura)

    record_id = backend.capture_consequence(
        situation="goal needs verified answer",
        action="researcher.execute",
        consequence="SUPPORTS",
        trust=1,
        scope=["goal:g1"],
        provenance=["test"],
        links={"goal": "g1"},
        namespace="remy",
    )

    assert record_id == "cu1"
    assert aura.consequence_calls[0]["consequence"] == "SUPPORTS"
    assert aura.consequence_calls[0]["scope"] == ["goal:g1"]
    assert aura.store_calls == []


def test_v3_memory_capture_consequence_resolves_runtime_links_to_aura_records():
    aura = _DummyAuraWithConsequence()
    aura.search_results = [
        _DummyRecord(
            content="Goal outcome",
            tags=["outcome", "goal_outcome"],
            metadata={"goal_id": "g1", "mission_id": "m1"},
            level="Decisions",
            record_id="outcome-rec-1",
        )
    ]
    backend = AuraMemoryBackend(aura)

    backend.capture_consequence(
        situation="goal needs verified answer",
        action="complete",
        consequence="SUPPORTS",
        trust=1,
        links={"goal": "g1", "mission": "m1", "step": "s1"},
        namespace="remy",
    )

    links = aura.consequence_calls[0]["links"]
    assert links["goal"] == "g1"
    assert links["mission"] == "m1"
    assert links["goal_record"] == "outcome-rec-1"
    assert links["mission_record"] == "outcome-rec-1"
    assert "step_record" not in links


def test_v3_memory_capture_consequence_falls_back_to_structured_outcome_record():
    aura = _DummyAura()
    backend = AuraMemoryBackend(aura)

    record_id = backend.capture_consequence(
        situation="goal failed",
        action="executor.run",
        consequence="REFUTES",
        trust=-1,
        scope=["goal:g2"],
        provenance=["test"],
        namespace="remy",
    )

    assert record_id == "r1"
    call = aura.store_calls[0]
    assert "consequence-unit" in call["tags"]
    assert "consequence-refute" in call["tags"]
    assert call["metadata"]["kind"] == "consequence_unit"
    assert call["metadata"]["cu_consequence"] == "REFUTES"
    assert call["metadata"]["memory_class"] == "outcome"


def test_v3_memory_wrap_record_exposes_semantic_type():
    aura = _DummyAura()
    aura.search_results = [
        _DummyRecord(
            content="Scheduled vendor payment for tomorrow",
            tags=["task"],
            metadata={"created_at": "123"},
            level="Decisions",
        )
    ]
    backend = AuraMemoryBackend(aura)

    results = backend.recall("vendor payment", memory_class=MemoryClass.TASK)

    assert len(results) == 1
    assert results[0].semantic_type == "decision"
    assert results[0].metadata["semantic_type"] == "decision"
    assert results[0].memory_class == MemoryClass.TASK


def test_v3_record_models_provide_semantic_type_defaults():
    _, _, goal_meta, _ = goal_record("Finalize launch plan", mission_id="m1")
    _, _, finding_meta, _ = finding_record("Competitor pricing starts at $29", mission_id="m1")
    _, _, failure_meta, _ = failure_record("DISPROVED: signup flow is stable", mission_id="m1")

    assert goal_meta["semantic_type"] == "decision"
    assert finding_meta["semantic_type"] == "fact"
    assert failure_meta["semantic_type"] == "contradiction"


def test_cycle_recorder_stores_cycle_as_consequence_unit(monkeypatch):
    from remy.core_v3.execution.cycle_recorder import CycleRecord, CycleRecorder
    import remy.core_v3.memory.memory_api as memory_api

    class _FakeMemory:
        def __init__(self):
            self.calls = []

        def capture_consequence(self, **kwargs):
            self.calls.append(kwargs)
            return "cu-cycle"

    fake = _FakeMemory()
    monkeypatch.setattr(memory_api, "get_memory", lambda: fake)

    recorder = CycleRecorder()
    recorder._write_v2 = lambda rec: None
    recorder.record(
        CycleRecord(
            cycle_num=7,
            mission_id="m1",
            goal_id="g1",
            step_id="s1",
            goal_description="Verify claim before answering",
            specialist="researcher",
            status="success",
            verdict="success",
            reason="verified",
            decision="complete",
        )
    )

    assert fake.calls[0]["situation"] == "Verify claim before answering"
    assert fake.calls[0]["action"] == "complete"
    assert fake.calls[0]["consequence"] == "SUPPORTS"
    assert fake.calls[0]["trust"] == 1
    assert "remy:cycle_recorder" in fake.calls[0]["provenance"]
