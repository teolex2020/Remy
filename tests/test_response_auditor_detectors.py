"""Tests for response-auditor detectors and orchestrator.

The canonical fixture is a reduced version of a real conversation where the
agent emitted multiple fabricated claims in a single turn — arXiv IDs without
fetch, internal metrics without aura_cognitive_ops, first-person discovery
claims, and a record count. That fixture should produce multiple violations
in warn mode and a cleanly rewritten response in block mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from remy.core import introspection_cache
from remy.core.retrieval.claim_spans import EvidenceRequirement
from remy.core.retrieval.detectors import (
    detect_all,
    detect_entitlement,
    detect_external_ids,
    detect_live_telemetry,
)
from remy.core.retrieval.evidence_resolver import TurnContext
from remy.core.retrieval.response_auditor import audit_response


# ---------- fixtures ----------

REAL_AGENT_RESPONSE = (
    "Сьогодні я виділила 5 ключових робіт з arXiv. "
    "Reflection-Bench (2410.16270) — про саморефлексію. "
    "Cognitive Agency Surrender (2603.21735) — про втрату агентності. "
    "У мене зараз 976 записів у пам'яті. "
    "Стабільність знань становить 38.8%. "
    "Мій стрес 0.26, а цікавість 0.67. "
    "70 переконань конфліктують між собою."
)


@pytest.fixture(autouse=True)
def _reset_cache():
    introspection_cache.reset()
    yield
    introspection_cache.reset()


def _tool_call(tool: str, result: str, args: dict | None = None) -> dict:
    return {
        "type": "tool_call",
        "tool": tool,
        "args": args or {},
        "result_full": result,
    }


# ---------- external_ids detector ----------

class TestExternalIds:
    def test_detects_arxiv_ids(self):
        spans = detect_external_ids(REAL_AGENT_RESPONSE)
        arxiv = [s for s in spans if s.claim_type == "arxiv_id"]
        ids = sorted(s.entity_hint for s in arxiv)
        assert ids == ["2410.16270", "2603.21735"]

    def test_detects_doi(self):
        spans = detect_external_ids("See 10.1145/3442188.3445922 for details")
        assert any(s.claim_type == "doi" for s in spans)

    def test_detects_url(self):
        spans = detect_external_ids("visit https://arxiv.org/abs/2410.16270 now")
        urls = [s for s in spans if s.claim_type == "url_authoritative"]
        assert urls and urls[0].entity_hint.startswith("https://arxiv.org")

    def test_no_false_positive_on_plain_text(self):
        spans = detect_external_ids("Просто звичайний текст без айдішників.")
        assert spans == []

    def test_requires_evidence_with_id(self):
        spans = detect_external_ids("paper 2410.16270 is good")
        assert spans[0].requires_evidence is EvidenceRequirement.TOOL_CALL_WITH_ID


# ---------- live_telemetry detector ----------

class TestLiveTelemetry:
    def test_detects_stability_percent(self):
        spans = detect_live_telemetry("Стабільність 38.8%")
        live = [s for s in spans if s.claim_type == "live_metric"]
        assert live
        assert live[0].numeric_value == 38.8
        assert live[0].requires_evidence is EvidenceRequirement.FRESH_INTROSPECTION

    def test_detects_temperature_decimal(self):
        spans = detect_live_telemetry("Температура зараз 0.149")
        live = [s for s in spans if s.claim_type == "live_metric"]
        assert live and live[0].numeric_value == 0.149

    def test_detects_record_count(self):
        spans = detect_live_telemetry("У мене 976 записів у пам'яті")
        rec = [s for s in spans if s.claim_type == "record_count"]
        assert rec and rec[0].numeric_value == 976

    def test_detects_belief_count(self):
        spans = detect_live_telemetry("70 переконань конфліктують")
        beliefs = [s for s in spans if s.claim_type == "belief_count"]
        assert beliefs and beliefs[0].numeric_value == 70

    def test_skips_memory_grounded_world_fact(self):
        # "65% людей" is a fact about the world, not live telemetry.
        spans = detect_live_telemetry("65% людей в Україні стресують постійно")
        assert [s for s in spans if s.claim_type == "live_metric"] == []

    def test_skips_plain_number_without_hint(self):
        spans = detect_live_telemetry("Було 5 яблук на столі")
        assert spans == []

    def test_real_fixture_finds_all_metrics(self):
        spans = detect_live_telemetry(REAL_AGENT_RESPONSE)
        types = sorted(s.claim_type for s in spans)
        # Expect: record_count (976), live_metric (38.8, 0.26, 0.67), belief_count (70)
        assert "record_count" in types
        assert "belief_count" in types
        assert types.count("live_metric") >= 2


# ---------- entitlement detector ----------

class TestEntitlement:
    def test_detects_i_found_works(self):
        spans = detect_entitlement("я виділила 5 ключових робіт")
        assert spans
        assert spans[0].claim_type == "entitlement"
        assert spans[0].requires_evidence is EvidenceRequirement.TOOL_CALL_ANY

    def test_skips_feelings(self):
        spans = detect_entitlement("я почуваюся бадьоро сьогодні")
        assert spans == []

    def test_skips_non_factual_analysis(self):
        # "я виявила проблему" — not a factual lookup, no target from list
        spans = detect_entitlement("я виявила проблему у своїй поведінці")
        assert spans == []

    def test_detects_found_paper(self):
        spans = detect_entitlement("я знайшла 3 статті про агентність")
        assert spans


# ---------- full orchestrator, warn mode ----------

class TestAuditWarnMode:
    def test_fabricated_response_yields_multiple_violations(self):
        report = audit_response(
            REAL_AGENT_RESPONSE,
            turn=TurnContext(session_log=[], session_id="s1"),
            mode="warn",
            log_path=Path("d:/tmp/test_audit_ignore.jsonl"),
        )
        assert report.has_violations
        types = sorted({v.claim.claim_type for v in report.violations})
        # Expect at least: arxiv_id, live_metric, record_count, belief_count, entitlement
        assert "arxiv_id" in types
        assert "live_metric" in types
        assert "record_count" in types
        assert "belief_count" in types
        assert "entitlement" in types

    def test_arxiv_claim_passes_when_extract_content_backs_it(self):
        turn = TurnContext(
            session_log=[
                _tool_call("extract_content", "Paper 2410.16270 abstract: Reflection...")
            ],
            session_id="s1",
        )
        text = "стаття 2410.16270 релевантна"
        report = audit_response(text, turn=turn, mode="warn",
                                log_path=Path("d:/tmp/test_audit_ignore.jsonl"))
        arxiv_actions = [a for a in report.actions if a.claim.claim_type == "arxiv_id"]
        assert arxiv_actions and all(a.mode == "pass" for a in arxiv_actions)

    def test_live_metric_passes_when_introspection_fresh(self):
        introspection_cache.stamp("s1", "status", {"stability": 0.388})
        text = "стабільність 38.8% зараз"
        report = audit_response(
            text,
            turn=TurnContext(session_id="s1"),
            mode="warn",
            log_path=Path("d:/tmp/test_audit_ignore.jsonl"),
        )
        live = [a for a in report.actions if a.claim.claim_type == "live_metric"]
        assert live and all(a.mode == "pass" for a in live)

    def test_off_mode_short_circuits(self):
        report = audit_response(REAL_AGENT_RESPONSE, mode="off")
        assert report.actions == []
        assert report.has_violations is False


# ---------- block mode rewrite ----------

class TestAuditBlockMode:
    def test_arxiv_redacted(self, tmp_path):
        text = "paper 2410.16270 is relevant"
        report = audit_response(
            text,
            turn=TurnContext(session_id="s1"),
            mode="block",
            log_path=tmp_path / "audit.jsonl",
        )
        assert report.rewritten_text is not None
        assert "2410.16270" not in report.rewritten_text
        assert "[потребує перевірки]" in report.rewritten_text

    def test_live_metric_redacted(self, tmp_path):
        text = "Стабільність 38.8% зараз низька"
        report = audit_response(
            text,
            turn=TurnContext(session_id="s1"),
            mode="block",
            log_path=tmp_path / "audit.jsonl",
        )
        assert "38.8%" not in (report.rewritten_text or "")

    def test_entitlement_downgraded(self, tmp_path):
        text = "я виділила 5 робіт"
        report = audit_response(
            text,
            turn=TurnContext(session_id="s1"),
            mode="block",
            log_path=tmp_path / "audit.jsonl",
        )
        rewritten = report.rewritten_text or ""
        assert "я виділила" not in rewritten
        assert "можливо" in rewritten or "припускаю" in rewritten


# ---------- audit log ----------

class TestAuditLog:
    def test_violation_writes_jsonl_record(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        audit_response(
            REAL_AGENT_RESPONSE,
            turn=TurnContext(session_id="sess-X"),
            mode="warn",
            log_path=log,
            turn_id="turn-42",
        )
        assert log.exists()
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["session_id"] == "sess-X"
        assert rec["turn_id"] == "turn-42"
        assert rec["response_hash"].startswith("sha256:")
        assert len(rec["violations"]) >= 5

    def test_no_violations_no_log_written(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        audit_response(
            "Звичайне повідомлення без жодних цифр чи IDшників.",
            turn=TurnContext(session_id="s1"),
            mode="warn",
            log_path=log,
        )
        assert not log.exists()


# ---------- detect_all composition ----------

class TestDetectAll:
    def test_combines_all_detectors(self):
        spans = detect_all(REAL_AGENT_RESPONSE)
        detectors = {s.detector for s in spans}
        assert "external_ids" in detectors
        assert "live_telemetry" in detectors
        assert "entitlement" in detectors
