"""
Tool-output compression for `aura_cognitive_ops`.

The dispatcher must:
  - default to brief mode for list payloads (top-K + aggregate)
  - allow opt-out via params={'full': true}
  - hard-cap any response at 60 KB regardless of mode

Reason for these guards: an unbounded `list_records` dump previously blew past
Gemini's 1M-token context window in a single tool turn, killing the agent loop
before any downstream brain step (mouth, governance) could fire.
"""

from __future__ import annotations

import json

import pytest

from remy.core.tool_dispatch import (
    _AURA_OP_BRIEF_TOP_K,
    _AURA_OP_PAYLOAD_BYTE_CAP,
    _aura_op_brief_aggregate,
    _aura_op_briefify_record,
    _aura_op_compress_if_huge,
)


def _record(i: int, conn_count: int = 50) -> dict:
    return {
        "id": f"rec_{i:04d}",
        "content": f"Test content for record {i} " + ("x" * 200),
        "tags": [f"tag_{i % 10}"],
        "strength": 0.5 + (i % 50) / 100,
        "activation_count": i % 100,
        "connections": {f"rec_{(i + j) % 500:04d}": 0.5 for j in range(conn_count)},
    }


def test_brief_mode_compresses_large_list_to_top_k():
    big = [_record(i) for i in range(500)]
    raw_bytes = len(json.dumps(big).encode("utf-8"))
    assert raw_bytes > 100_000, "fixture should be huge to make the test meaningful"

    result, meta = _aura_op_compress_if_huge(big, "list_records", brief=True)

    assert meta["briefed"] is True
    assert result["mode"] == "brief"
    assert len(result["top_k"]) == _AURA_OP_BRIEF_TOP_K
    out_bytes = len(json.dumps(result).encode("utf-8"))
    assert out_bytes < raw_bytes / 10, (
        f"brief mode should reduce by >10x, got {raw_bytes} -> {out_bytes}"
    )


def test_brief_mode_top_k_is_sorted_by_activation_desc():
    big = [_record(i) for i in range(300)]
    result, _ = _aura_op_compress_if_huge(big, "list_records", brief=True)
    activations = [r["activation_count"] for r in result["top_k"]]
    assert activations == sorted(activations, reverse=True)


def test_brief_mode_drops_connections_map_but_keeps_count():
    big = [_record(i, conn_count=42) for i in range(100)]
    result, _ = _aura_op_compress_if_huge(big, "list_records", brief=True)
    for r in result["top_k"]:
        assert "connections" not in r
        assert r["connections_count"] == 42


def test_brief_mode_aggregate_describes_full_set():
    big = [_record(i) for i in range(50)]
    result, _ = _aura_op_compress_if_huge(big, "list_records", brief=True)
    agg = result["aggregate"]
    assert agg["total"] == 50
    assert agg["activation_max"] <= 99
    assert agg["connections_mean"] == 50.0


def test_small_list_passes_through_unchanged_in_brief_mode():
    small = [_record(i) for i in range(_AURA_OP_BRIEF_TOP_K)]
    result, meta = _aura_op_compress_if_huge(small, "list_records", brief=True)
    assert isinstance(result, list)
    assert len(result) == _AURA_OP_BRIEF_TOP_K
    assert meta == {}


def test_full_mode_oversize_list_triggers_byte_cap_truncation():
    big = [_record(i) for i in range(500)]
    result, meta = _aura_op_compress_if_huge(big, "list_records", brief=False)

    assert meta["truncated"] is True
    assert meta["original_bytes"] > _AURA_OP_PAYLOAD_BYTE_CAP
    assert result["mode"] == "truncated"
    assert len(result["sample"]) == 5
    out_bytes = len(json.dumps(result).encode("utf-8"))
    assert out_bytes < _AURA_OP_PAYLOAD_BYTE_CAP


def test_full_mode_small_list_passes_through_unchanged():
    small = [_record(i, conn_count=2) for i in range(3)]
    result, meta = _aura_op_compress_if_huge(small, "list_records", brief=False)
    assert isinstance(result, list)
    assert len(result) == 3
    assert meta == {}


def test_oversize_dict_payload_is_truncated_to_key_listing():
    big = {f"key_{i}": _record(i) for i in range(500)}
    result, meta = _aura_op_compress_if_huge(big, "get_family_graph", brief=False)
    assert meta["truncated"] is True
    assert result["mode"] == "truncated"
    assert result["key_count"] == 500
    assert len(result["keys"]) <= 50


def test_scalar_payload_passes_through():
    for v in (None, 42, 3.14, True, "hello"):
        result, meta = _aura_op_compress_if_huge(v, "count", brief=True)
        assert result == v
        assert meta == {}


def test_briefify_truncates_long_content_field():
    rec = {"id": "x", "content": "a" * 1000}
    out = _aura_op_briefify_record(rec)
    assert len(out["content"]) <= 201  # 200 chars + ellipsis


def test_briefify_handles_non_dict():
    assert _aura_op_briefify_record("just a string") == "just a string"
    assert _aura_op_briefify_record(None) is None


def test_aggregate_handles_empty_list():
    assert _aura_op_brief_aggregate([]) == {"total": 0}


def test_aggregate_handles_list_of_non_dicts():
    agg = _aura_op_brief_aggregate(["a", "b", "c"])
    assert agg["total"] == 3
