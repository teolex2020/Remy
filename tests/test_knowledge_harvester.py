"""
Tests for Knowledge Harvester Mode 1 (Skeleton Build).

Uses mock LLM callers — no real API calls.
"""

import json
import pytest

from tools.authoring.knowledge_harvester import (
    DomainSkeleton,
    HarvestBudget,
    HarvestReport,
    harvest_skeleton,
    _build_interrogation_plan,
    _parse_extraction_response,
    _raw_to_pack_record,
    _filter_candidates,
)


# ── Fixtures ──


def _make_skeleton(**kwargs):
    defaults = {
        "domain": "test_domain",
        "intended_role": "test_analyst",
        "concepts": ["concept_a", "concept_b"],
        "procedures": ["procedure_x"],
        "risks": ["risk_1"],
        "constraints": ["constraint_alpha"],
        "tags_prefix": ["domain:test"],
    }
    defaults.update(kwargs)
    return DomainSkeleton(**defaults)


def _mock_llm_caller(responses: list[str]):
    """Returns a callable that yields pre-defined responses in order."""
    idx = [0]

    def caller(prompt: str) -> str:
        if idx[0] < len(responses):
            result = responses[idx[0]]
            idx[0] += 1
            return result
        return "[]"

    return caller


_SAMPLE_EXTRACTION_JSON = json.dumps([
    {
        "content": "When lateral movement crosses network segments, containment must isolate affected segments within 5 minutes.",
        "unit_type": "Rule",
        "tags": ["threat:lateral-movement", "response:containment"],
        "semantic_type": "fact",
        "confidence_note": "high",
    },
    {
        "content": "Firewall rule changes during active incidents require dual approval to prevent accidental lockout.",
        "unit_type": "Constraint",
        "tags": ["incident:response", "policy:dual-approval"],
        "semantic_type": "decision",
        "confidence_note": "medium",
    },
])


# ── Interrogation Plan ──


class TestInterrogationPlan:
    def test_plan_covers_all_skeleton_items(self):
        skeleton = _make_skeleton()
        plan = _build_interrogation_plan(skeleton)

        # 2 concepts × 2 patterns + 1 procedure × 2 + 1 risk × 2 + 1 constraint × 1 = 9
        assert len(plan) == 9

    def test_plan_empty_skeleton(self):
        skeleton = _make_skeleton(concepts=[], procedures=[], risks=[], constraints=[])
        plan = _build_interrogation_plan(skeleton)
        assert len(plan) == 0

    def test_plan_items_have_required_fields(self):
        skeleton = _make_skeleton()
        plan = _build_interrogation_plan(skeleton)

        for q in plan:
            assert "pattern" in q
            assert "item" in q
            assert "prompt" in q
            assert skeleton.domain in q["prompt"]
            assert q["item"] in q["prompt"]

    def test_plan_concept_generates_two_patterns(self):
        skeleton = _make_skeleton(concepts=["firewall_rules"], procedures=[], risks=[], constraints=[])
        plan = _build_interrogation_plan(skeleton)
        assert len(plan) == 2
        patterns = {q["pattern"] for q in plan}
        assert "concept_mechanisms" in patterns
        assert "concept_relationships" in patterns


# ── Extraction Parsing ──


class TestExtractionParsing:
    def test_parse_valid_json_array(self):
        records = _parse_extraction_response(_SAMPLE_EXTRACTION_JSON)
        assert len(records) == 2
        assert records[0]["unit_type"] == "Rule"

    def test_parse_json_with_markdown_fences(self):
        text = f"```json\n{_SAMPLE_EXTRACTION_JSON}\n```"
        records = _parse_extraction_response(text)
        assert len(records) == 2

    def test_parse_json_embedded_in_text(self):
        text = f"Here are the records:\n{_SAMPLE_EXTRACTION_JSON}\nEnd of records."
        records = _parse_extraction_response(text)
        assert len(records) == 2

    def test_parse_invalid_json_returns_empty(self):
        records = _parse_extraction_response("this is not json at all")
        assert records == []

    def test_parse_empty_array(self):
        records = _parse_extraction_response("[]")
        assert records == []


# ── Raw to PackRecord ──


class TestRawToPackRecord:
    def test_valid_record(self):
        skeleton = _make_skeleton()
        raw = {
            "content": "Containment must isolate affected segments within 5 minutes.",
            "unit_type": "Rule",
            "tags": ["response:containment"],
            "semantic_type": "fact",
            "confidence_note": "high",
        }
        rec = _raw_to_pack_record(raw, skeleton, "concept_a", "concept_mechanisms", "test-model")
        assert rec is not None
        assert rec["source_authority"] == "Inferred"
        assert rec["confidence"] == 0.80  # high
        assert "domain:test" in rec["tags"]  # prefix tag
        assert rec["namespace"] == "default"
        assert any("harvester:skeleton:" in p for p in rec["provenance_refs"])

    def test_empty_content_rejected(self):
        skeleton = _make_skeleton()
        raw = {"content": "", "unit_type": "Fact"}
        assert _raw_to_pack_record(raw, skeleton, "x", "y", "m") is None

    def test_short_content_rejected(self):
        skeleton = _make_skeleton()
        raw = {"content": "Too short", "unit_type": "Fact"}
        assert _raw_to_pack_record(raw, skeleton, "x", "y", "m") is None

    def test_invalid_unit_type_defaults_to_fact(self):
        skeleton = _make_skeleton()
        raw = {"content": "A valid content sentence for testing purposes.", "unit_type": "InvalidType"}
        rec = _raw_to_pack_record(raw, skeleton, "x", "y", "m")
        assert rec is not None
        assert rec["unit_type"] == "Fact"

    def test_confidence_mapping(self):
        skeleton = _make_skeleton()
        for note, expected in [("high", 0.80), ("medium", 0.60), ("low", 0.40), ("unknown", 0.60)]:
            raw = {
                "content": "A valid content sentence for testing the confidence mapping.",
                "unit_type": "Fact",
                "confidence_note": note,
            }
            rec = _raw_to_pack_record(raw, skeleton, "x", "y", "m")
            assert rec["confidence"] == expected, f"Failed for {note}"


# ── Filtering ──


class TestFiltering:
    def test_empty_candidates(self):
        accepted, stats = _filter_candidates([])
        assert accepted == []
        assert stats["total"] == 0

    def test_deduplication(self):
        candidates = [
            {"content": "Same content here.", "tags": ["a"]},
            {"content": "Same content here.", "tags": ["b"]},
            {"content": "Different content here.", "tags": ["c"]},
        ]
        accepted, stats = _filter_candidates(candidates)
        assert len(accepted) == 2
        assert stats["duplicates_removed"] == 1

    def test_all_unique_pass_without_brain(self):
        candidates = [
            {"content": f"Unique content number {i} for testing." , "tags": []}
            for i in range(5)
        ]
        accepted, stats = _filter_candidates(candidates)
        assert len(accepted) == 5
        assert stats["accepted"] == 5


# ── End-to-End Harvest ──


class TestHarvestSkeleton:
    def test_basic_harvest(self):
        """End-to-end harvest with mock LLM returning valid extraction."""
        skeleton = _make_skeleton(
            concepts=["access_control"],
            procedures=[],
            risks=[],
            constraints=[],
        )

        # 2 questions (concept_mechanisms + concept_relationships)
        # Each needs: interrogation response + extraction response = 4 calls
        responses = [
            "Access control involves authentication and authorization...",
            _SAMPLE_EXTRACTION_JSON,
            "Access control interacts with identity management...",
            _SAMPLE_EXTRACTION_JSON,
        ]

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            budget=HarvestBudget(max_llm_calls=10),
            llm_caller=_mock_llm_caller(responses),
            llm_model_name="mock-model",
        )

        assert report.domain == "test_domain"
        assert report.llm_calls_used == 4
        assert report.candidate_records > 0
        assert report.accepted_records > 0
        assert pack["base_id"] == "test_domain-harvested"
        assert pack["version"] == "0.1.0"
        assert len(pack["records"]) == report.accepted_records
        assert pack["metadata"]["source_mode"] == "harvester_skeleton_v1"

        # All records should be Inferred
        for rec in pack["records"]:
            assert rec["source_authority"] == "Inferred"

    def test_budget_enforced(self):
        """Budget exhaustion stops harvest early."""
        skeleton = _make_skeleton(
            concepts=["a", "b", "c", "d", "e"],
            procedures=["p1", "p2", "p3"],
            risks=["r1", "r2"],
            constraints=["c1"],
        )

        # Only allow 4 LLM calls (2 questions worth)
        responses = [
            "Some knowledge...",
            _SAMPLE_EXTRACTION_JSON,
            "More knowledge...",
            _SAMPLE_EXTRACTION_JSON,
        ]

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            budget=HarvestBudget(max_llm_calls=4),
            llm_caller=_mock_llm_caller(responses),
        )

        assert report.llm_calls_used <= 4
        assert report.budget_exhausted is True

    def test_max_accepted_records_cap(self):
        """Max accepted records cap is enforced."""
        skeleton = _make_skeleton(concepts=["a"], procedures=[], risks=[], constraints=[])

        # Return many records
        many_records = json.dumps([
            {
                "content": f"Knowledge record number {i} with enough text to pass.",
                "unit_type": "Fact",
                "tags": ["test"],
                "semantic_type": "fact",
                "confidence_note": "medium",
            }
            for i in range(20)
        ])

        responses = [
            "Knowledge text...",
            many_records,
            "More knowledge...",
            many_records,
        ]

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            budget=HarvestBudget(max_accepted_records=5),
            llm_caller=_mock_llm_caller(responses),
        )

        assert len(pack["records"]) <= 5

    def test_llm_failure_recorded_in_errors(self):
        """LLM failures are recorded, not fatal."""
        skeleton = _make_skeleton(concepts=["a"], procedures=[], risks=[], constraints=[])

        call_count = [0]
        def failing_caller(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("LLM unavailable")
            return _SAMPLE_EXTRACTION_JSON

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            llm_caller=failing_caller,
        )

        assert len(report.errors) >= 1
        assert "LLM unavailable" in report.errors[0]

    def test_empty_skeleton_produces_empty_pack(self):
        """Empty skeleton = no questions = empty pack."""
        skeleton = _make_skeleton(concepts=[], procedures=[], risks=[], constraints=[])

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            llm_caller=_mock_llm_caller([]),
        )

        assert report.llm_calls_used == 0
        assert report.candidate_records == 0
        assert len(pack["records"]) == 0

    def test_provenance_tracking(self):
        """All records have harvester provenance."""
        skeleton = _make_skeleton(concepts=["firewall"], procedures=[], risks=[], constraints=[])

        responses = [
            "Firewall knowledge...",
            _SAMPLE_EXTRACTION_JSON,
            "Firewall interactions...",
            _SAMPLE_EXTRACTION_JSON,
        ]

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            llm_caller=_mock_llm_caller(responses),
            llm_model_name="test-model-v1",
        )

        for rec in pack["records"]:
            assert len(rec["provenance_refs"]) > 0
            ref = rec["provenance_refs"][0]
            assert ref.startswith("harvester:skeleton:")
            assert "test_domain" in ref
            assert "test-model-v1" in ref


# ── Round-Robin Fairness ──


class TestRoundRobinFairness:
    def test_all_items_get_at_least_one_pass(self):
        """With enough budget, every skeleton item gets interrogated."""
        skeleton = _make_skeleton(
            concepts=["alpha", "beta", "gamma"],
            procedures=["proc_1"],
            risks=["risk_1"],
            constraints=["con_1"],
        )
        # 6 items × varying questions = 11 total questions × 2 calls each = 22 LLM calls
        # Use unique content per call to avoid dedup collapsing everything

        call_idx = [0]
        items_interrogated: list[str] = []

        def tracking_caller(prompt: str) -> str:
            call_idx[0] += 1
            # Track which items get interrogated (odd calls are interrogation)
            if call_idx[0] % 2 == 1:
                for item in ["alpha", "beta", "gamma", "proc_1", "risk_1", "con_1"]:
                    if f"'{item}'" in prompt:
                        items_interrogated.append(item)
                        break
                return f"Knowledge about topic {call_idx[0]}..."
            else:
                # Extraction: return unique content per call
                return json.dumps([{
                    "content": f"Unique record {call_idx[0]} from extraction with enough words.",
                    "unit_type": "Fact",
                    "tags": ["test"],
                    "semantic_type": "fact",
                    "confidence_note": "high",
                }])

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            budget=HarvestBudget(max_llm_calls=30, max_candidate_records=500),
            llm_caller=tracking_caller,
            llm_model_name="mock",
        )

        # All 6 skeleton items should have been interrogated
        assert set(items_interrogated) == {"alpha", "beta", "gamma", "proc_1", "risk_1", "con_1"}

    def test_per_item_cap_prevents_domination(self):
        """A prolific item cannot consume the entire candidate budget."""
        skeleton = _make_skeleton(
            concepts=["prolific", "quiet"],
            procedures=[],
            risks=[],
            constraints=[],
        )

        # "prolific" returns 20 records per extraction, "quiet" returns 2
        prolific_extraction = json.dumps([
            {
                "content": f"Prolific knowledge record number {i} with sufficient text.",
                "unit_type": "Fact",
                "tags": ["test"],
                "semantic_type": "fact",
                "confidence_note": "high",
            }
            for i in range(20)
        ])
        quiet_extraction = json.dumps([
            {
                "content": f"Quiet knowledge record {i} with enough content here.",
                "unit_type": "Rule",
                "tags": ["test"],
                "semantic_type": "fact",
                "confidence_note": "high",
            }
            for i in range(2)
        ])

        # Round-robin pass 0: prolific_mechanisms, quiet_mechanisms
        # Round-robin pass 1: prolific_relationships, quiet_relationships
        responses = [
            # Pass 0: prolific concept_mechanisms
            "Prolific knowledge...", prolific_extraction,
            # Pass 0: quiet concept_mechanisms
            "Quiet knowledge...", quiet_extraction,
            # Pass 1: prolific concept_relationships
            "More prolific...", prolific_extraction,
            # Pass 1: quiet concept_relationships
            "More quiet...", quiet_extraction,
        ]

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            budget=HarvestBudget(
                max_llm_calls=20,
                max_candidate_records=100,
                per_item_candidate_cap=15,
            ),
            llm_caller=_mock_llm_caller(responses),
        )

        # Count records per item
        item_counts: dict[str, int] = {}
        for rec in pack["records"]:
            for ref in rec.get("provenance_refs", []):
                parts = ref.split(":")
                if len(parts) >= 5:
                    item = parts[3]
                    item_counts[item] = item_counts.get(item, 0) + 1

        # Prolific should be capped, quiet should have its records
        assert item_counts.get("prolific", 0) <= 15
        assert item_counts.get("quiet", 0) >= 2

    def test_round_robin_order(self):
        """First pass visits all items before second pass starts."""
        skeleton = _make_skeleton(
            concepts=["A", "B"],
            procedures=[],
            risks=[],
            constraints=[],
        )
        # 2 concepts × 2 patterns = 4 questions
        # Round-robin: pass 0 → A:mechanisms, B:mechanisms
        #              pass 1 → A:relationships, B:relationships

        interrogation_log: list[str] = []
        call_idx = [0]

        def logging_caller(prompt: str) -> str:
            call_idx[0] += 1
            # Interrogation calls (odd) contain the item name in quotes
            for item in ["A", "B"]:
                if f"'{item}'" in prompt:
                    interrogation_log.append(item)
                    return f"Knowledge about {item} call {call_idx[0]}..."
            # Extraction calls — return unique content
            return json.dumps([{
                "content": f"Unique extraction record number {call_idx[0]} here.",
                "unit_type": "Fact",
                "tags": ["test"],
                "semantic_type": "fact",
                "confidence_note": "high",
            }])

        pack, report = harvest_skeleton(
            skeleton=skeleton,
            budget=HarvestBudget(max_llm_calls=20, max_candidate_records=500),
            llm_caller=logging_caller,
        )

        # Should have 4 interrogations: A, B, A, B (round-robin)
        assert len(interrogation_log) == 4
        # First two should be A and B (not A, A) — coverage before depth
        assert set(interrogation_log[:2]) == {"A", "B"}


# ── Budget Validation ──


class TestBudgetValidation:
    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            HarvestBudget(max_llm_calls=0)

        with pytest.raises(ValueError):
            HarvestBudget(max_candidate_records=-1)
