"""Integration coverage for AuraSDK 1.57 temporal memory wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from remy.core.agent_tools import Level, _AuraCompat
from remy.core.temporal_memory import parse_memory_timestamp, temporal_store_kwargs
from remy.core.tool_dispatch import _temporal_memory_tool
from remy.core_v3.memory.memory_api import AuraMemoryBackend, MemoryClass


def test_timestamp_parser_accepts_iso_and_rejects_inverted_interval():
    assert parse_memory_timestamp("1970-01-01T00:03:20Z", field_name="at") == 200.0
    with pytest.raises(ValueError, match="earlier"):
        temporal_store_kwargs({"valid_from": "200", "valid_until": "100"})


def test_compat_temporal_version_chain_explain_and_capsule(tmp_path):
    memory = _AuraCompat(str(tmp_path / "brain"))
    old = memory.store(
        "Office is Kyiv",
        level=Level.DOMAIN,
        tags=["location"],
        valid_from=100.0,
    )
    new = memory.supersede(
        old.id,
        "Office is Lviv",
        level=Level.DOMAIN,
        tags=["location"],
        effective_at=200.0,
    )

    before = memory.recall_as_of("Office", 150.0, top_k=5)
    after = memory.recall_as_of("Office", 250.0, top_k=5)
    assert before[0]["id"] == old.id
    assert before[0]["valid_until"] == 200.0
    assert after[0]["id"] == new.id
    assert after[0]["valid_from"] == 200.0

    explanation = memory.explain_recall("Office", top_k=5)
    assert explanation["trace_id"].startswith("mem_")
    assert explanation["decision_summary"]["rejection_counts"]["expired"] >= 1

    capsule = memory.build_context_capsule(
        "office location", token_budget=500, valid_at=150.0
    )
    assert capsule["entries"][0]["record_id"] == old.id


def test_v3_backend_preserves_temporal_fields(tmp_path):
    backend = AuraMemoryBackend(_AuraCompat(str(tmp_path / "v3-brain")))
    old_id = backend.store(
        "Plan is alpha",
        tags=["plan"],
        memory_class=MemoryClass.STRATEGIC,
        valid_from=100.0,
    )
    new_id = backend.supersede(
        old_id,
        "Plan is beta",
        effective_at=200.0,
        tags=["plan"],
        memory_class=MemoryClass.STRATEGIC,
    )

    old = backend.recall_as_of("Plan", 150.0, limit=5)[0]
    current = backend.recall_as_of("Plan", 250.0, limit=5)[0]
    assert old.id == old_id
    assert old.valid_until == 200.0
    assert current.id == new_id
    assert current.valid_from == 200.0


def test_agent_temporal_tools_return_bounded_auditable_payloads(tmp_path, monkeypatch):
    memory = _AuraCompat(str(tmp_path / "tool-brain"))
    old = memory.store("Status is pending", level=Level.DOMAIN, valid_from=100.0)
    fake_bt = SimpleNamespace(brain=memory, clear_recall_cache=lambda *_: None)
    monkeypatch.setattr("remy.core.tool_dispatch._get_bt", lambda: fake_bt)

    superseded = json.loads(
        _temporal_memory_tool(
            "supersede_memory",
            {
                "record_id": old.id,
                "new_content": "Status is complete",
                "effective_at": "1970-01-01T00:03:20Z",
            },
            channel="desktop",
        )
    )
    assert superseded["superseded"] is True
    assert superseded["old_valid_until"] == 200.0

    recalled = json.loads(
        _temporal_memory_tool(
            "recall_memory_as_of",
            {"query": "Status", "timestamp": "150", "top_k": 5},
            channel="desktop",
        )
    )
    assert recalled["activated_records"] is False
    assert recalled["results"][0]["id"] == old.id

    explained = json.loads(
        _temporal_memory_tool(
            "explain_memory_recall",
            {"query": "Status", "top_k": 5},
            channel="desktop",
        )
    )
    assert explained["explanation"]["trace_id"].startswith("mem_")

    capsule = json.loads(
        _temporal_memory_tool(
            "build_memory_context",
            {"purpose": "status", "valid_at": "150", "token_budget": 500},
            channel="desktop",
        )
    )
    assert capsule["context_capsule"]["entries"][0]["record_id"] == old.id
