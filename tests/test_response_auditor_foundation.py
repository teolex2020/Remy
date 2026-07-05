"""Foundation tests for the response-auditor pipeline.

Scope of this file:
  - ClaimSpan dataclass sanity
  - introspection_cache TTL + multi-session isolation
  - evidence_resolver single-seam behavior for each EvidenceRequirement

Detectors and full orchestrator are covered in separate test files once added.
"""

from __future__ import annotations

import time

import pytest

from remy.core import introspection_cache
from remy.core.retrieval.claim_spans import (
    AuditReport,
    ClaimSpan,
    EvidenceRequirement,
)
from remy.core.retrieval.evidence_resolver import (
    TurnContext,
    find_supporting_evidence,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    introspection_cache.reset()
    yield
    introspection_cache.reset()


# ---------- ClaimSpan ----------

class TestClaimSpan:
    def test_basic_fields(self):
        c = ClaimSpan(
            text="2410.16270",
            span=(10, 20),
            claim_type="arxiv_id",
            requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
            detector="external_ids",
            entity_hint="2410.16270",
        )
        assert c.claim_type == "arxiv_id"
        assert c.entity_hint == "2410.16270"
        assert c.numeric_value is None

    def test_audit_report_violations_filter(self):
        c = ClaimSpan(
            text="x", span=(0, 1), claim_type="entitlement",
            requires_evidence=EvidenceRequirement.TOOL_CALL_ANY, detector="ent",
        )
        from remy.core.retrieval.claim_spans import AuditAction
        r = AuditReport(
            response_text="x",
            actions=[
                AuditAction(mode="pass", claim=c),
                AuditAction(mode="warn", claim=c, reason="no evidence"),
            ],
        )
        assert r.has_violations is True
        assert len(r.violations) == 1
        assert r.violations[0].mode == "warn"


# ---------- introspection_cache ----------

class TestIntrospectionCache:
    def test_stamp_and_get_fresh(self):
        introspection_cache.stamp("s1", "status", {"count": 877})
        e = introspection_cache.get_fresh("s1")
        assert e is not None
        assert e.op == "status"
        assert e.result["count"] == 877

    def test_ttl_expiry(self):
        introspection_cache.stamp("s1", "status", {"x": 1})
        # Force the TTL to expire by using ttl_sec=0.
        time.sleep(0.01)
        assert introspection_cache.get_fresh("s1", ttl_sec=0.001) is None
        # With generous TTL it's still there.
        assert introspection_cache.get_fresh("s1", ttl_sec=60) is not None

    def test_op_filter(self):
        introspection_cache.stamp("s1", "status", {"a": 1})
        introspection_cache.stamp("s1", "get_contradiction_review_queue", {"b": 2})
        e = introspection_cache.get_fresh("s1", op="status")
        assert e is not None and e.op == "status"
        # Most recent wins when op=None.
        e2 = introspection_cache.get_fresh("s1")
        assert e2.op == "get_contradiction_review_queue"

    def test_multi_session_isolation(self):
        introspection_cache.stamp("sess-A", "status", {"A": 1})
        assert introspection_cache.get_fresh("sess-B") is None
        assert introspection_cache.get_fresh("sess-A") is not None

    def test_empty_session_id_is_noop(self):
        introspection_cache.stamp(None, "status", {"x": 1})
        introspection_cache.stamp("", "status", {"x": 1})
        assert introspection_cache.get_fresh(None) is None
        assert introspection_cache.get_fresh("") is None


# ---------- evidence_resolver ----------

def _make_turn(session_id: str = "s1", calls: list[dict] | None = None) -> TurnContext:
    return TurnContext(session_log=list(calls or []), session_id=session_id)


def _call(tool: str, result: str, args: dict | None = None) -> dict:
    return {
        "type": "tool_call",
        "tool": tool,
        "args": args or {},
        "result_full": result,
    }


class TestResolverFreshIntrospection:
    def test_returns_none_when_cache_empty(self):
        claim = ClaimSpan(
            text="38.8%", span=(0, 5), claim_type="live_metric",
            requires_evidence=EvidenceRequirement.FRESH_INTROSPECTION,
            detector="live_telemetry", numeric_value=38.8,
        )
        assert find_supporting_evidence(_make_turn(), claim) is None

    def test_returns_match_when_stamped(self):
        introspection_cache.stamp("s1", "status", {"stability": 0.388})
        claim = ClaimSpan(
            text="38.8%", span=(0, 5), claim_type="live_metric",
            requires_evidence=EvidenceRequirement.FRESH_INTROSPECTION,
            detector="live_telemetry",
        )
        m = find_supporting_evidence(_make_turn(), claim)
        assert m is not None
        assert m.source == "introspection_cache"
        assert "aura_cognitive_ops" in (m.tool_name or "")


class TestResolverToolCallWithId:
    def test_matches_entity_hint_in_result(self):
        calls = [_call("extract_content", "Paper 2410.16270 discusses reflection")]
        claim = ClaimSpan(
            text="2410.16270", span=(0, 10), claim_type="arxiv_id",
            requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
            detector="external_ids", entity_hint="2410.16270",
        )
        m = find_supporting_evidence(_make_turn(calls=calls), claim)
        assert m is not None
        assert m.tool_name == "extract_content"

    def test_no_match_when_id_absent(self):
        calls = [_call("web_search", "unrelated results about quantum stuff")]
        claim = ClaimSpan(
            text="2410.16270", span=(0, 10), claim_type="arxiv_id",
            requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
            detector="external_ids", entity_hint="2410.16270",
        )
        assert find_supporting_evidence(_make_turn(calls=calls), claim) is None

    def test_ignores_wrong_tool(self):
        # recall is not in the arxiv backing set
        calls = [_call("recall", "I remember 2410.16270 from yesterday")]
        claim = ClaimSpan(
            text="2410.16270", span=(0, 10), claim_type="arxiv_id",
            requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
            detector="external_ids", entity_hint="2410.16270",
        )
        assert find_supporting_evidence(_make_turn(calls=calls), claim) is None


class TestResolverToolCallInTurn:
    def test_any_matching_tool_call_resolves(self):
        calls = [_call("aura_cognitive_ops", '{"total_records": 877}')]
        claim = ClaimSpan(
            text="877 records", span=(0, 11), claim_type="record_count",
            requires_evidence=EvidenceRequirement.TOOL_CALL_IN_TURN,
            detector="live_telemetry", numeric_value=877,
        )
        m = find_supporting_evidence(_make_turn(calls=calls), claim)
        assert m is not None
        assert m.tool_name == "aura_cognitive_ops"

    def test_empty_log_yields_none(self):
        claim = ClaimSpan(
            text="877 records", span=(0, 11), claim_type="record_count",
            requires_evidence=EvidenceRequirement.TOOL_CALL_IN_TURN,
            detector="live_telemetry",
        )
        assert find_supporting_evidence(_make_turn(calls=[]), claim) is None


class TestResolverTurnContextHelpers:
    def test_filter_by_tool_name(self):
        turn = _make_turn(calls=[
            _call("web_search", "r1"),
            _call("recall", "r2"),
            _call("extract_content", "r3"),
        ])
        found = turn.tool_calls(frozenset({"web_search", "extract_content"}))
        names = [c["tool"] for c in found]
        assert names == ["web_search", "extract_content"]

    def test_ignores_non_tool_entries(self):
        turn = TurnContext(session_log=[
            {"type": "user_message", "content": "hi"},
            _call("web_search", "r"),
        ])
        assert len(turn.tool_calls()) == 1
