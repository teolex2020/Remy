"""
Knowledge Harvester — Mode 1: Skeleton Build

Bounded offline authoring tool for generating Specialist Base Packs.
This is NOT a runtime daemon. It is a one-time pack authoring workflow.

Architecture:
    Operator provides DomainSkeleton (concepts, procedures, risks, constraints)
    → Interrogator formulates narrow LLM questions
    → Extractor parses LLM responses → PackRecord dicts
    → Gate filters via Aura's filter_pack_records()
    → Accepted records assembled into a BasePack JSON

All extracted records get source_authority="Inferred" — they came from
LLM synthesis, not official sources. Full provenance is tracked.

Budget enforcement is hard from day 1:
    - max_llm_calls: hard cap on total LLM invocations
    - max_candidate_records: hard cap on records before filtering
    - max_accepted_records: hard cap on final pack size
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("knowledge_harvester")


# ── Domain Skeleton ──────────────────────────────────────────────────────────


@dataclass
class DomainSkeleton:
    """Operator-provided domain structure for skeleton-based harvesting."""

    domain: str
    """Domain identifier (e.g. 'incident_response', 'sensor_diagnostics')."""

    intended_role: str = ""
    """Target role or persona (e.g. 'security_operations_analyst')."""

    concepts: list[str] = field(default_factory=list)
    """Core domain concepts (5-15 recommended)."""

    procedures: list[str] = field(default_factory=list)
    """Core procedures (3-10 recommended)."""

    risks: list[str] = field(default_factory=list)
    """Core risks (5-10 recommended)."""

    constraints: list[str] = field(default_factory=list)
    """Core constraints (3-8 recommended)."""

    namespace: str = "default"
    """Isolation namespace for the generated records."""

    tags_prefix: list[str] = field(default_factory=list)
    """Tags to add to every generated record."""


# ── Budget ───────────────────────────────────────────────────────────────────


@dataclass
class HarvestBudget:
    """Hard budget caps for a single harvest session."""

    max_llm_calls: int = 100
    """Maximum LLM invocations (interrogation + extraction combined)."""

    max_candidate_records: int = 300
    """Maximum candidate records before filtering."""

    max_accepted_records: int = 200
    """Maximum records in the final pack."""

    per_item_candidate_cap: int = 0
    """Max candidates per skeleton item. 0 = auto (total / num_items)."""

    def __post_init__(self):
        if self.max_llm_calls < 1:
            raise ValueError("max_llm_calls must be >= 1")
        if self.max_candidate_records < 1:
            raise ValueError("max_candidate_records must be >= 1")


# ── Harvest Report ───────────────────────────────────────────────────────────


@dataclass
class HarvestReport:
    """Report from a skeleton harvest session."""

    domain: str = ""
    llm_calls_used: int = 0
    llm_calls_budget: int = 0
    candidate_records: int = 0
    accepted_records: int = 0
    acceptance_rate: float = 0.0
    budget_exhausted: bool = False
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ── Interrogator ─────────────────────────────────────────────────────────────


# Interrogation patterns: narrow, mechanism-focused questions per skeleton item.
_INTERROGATION_TEMPLATES = {
    "concept_mechanisms": (
        "What are the key causal mechanisms and internal dynamics of '{item}' "
        "in the domain of {domain}? Focus on WHY things happen, not just WHAT happens. "
        "Include exception cases and boundary conditions."
    ),
    "concept_relationships": (
        "How does '{item}' interact with other core concepts in {domain}? "
        "What dependencies, conflicts, or reinforcing loops exist? "
        "Focus on relationships that affect operational decisions."
    ),
    "procedure_exceptions": (
        "What exception cases, edge conditions, or failure modes can break "
        "the procedure '{item}' in {domain}? "
        "What should a practitioner watch for that standard documentation misses?"
    ),
    "procedure_constraints": (
        "What constraints apply when executing '{item}' in {domain}? "
        "Consider timing constraints, resource constraints, safety constraints, "
        "and conditions where the procedure should NOT be followed."
    ),
    "risk_mechanisms": (
        "What causal mechanisms drive the risk '{item}' in {domain}? "
        "What early indicators precede this risk? "
        "What conditions make this risk more or less likely?"
    ),
    "risk_mitigations": (
        "What mitigation strategies exist for the risk '{item}' in {domain}? "
        "What are the tradeoffs of each strategy? "
        "Which mitigations can fail and under what conditions?"
    ),
    "constraint_interactions": (
        "When the constraint '{item}' applies in {domain}, "
        "what other constraints does it interact with? "
        "What decisions become harder or impossible under this constraint? "
        "What workarounds exist and what are their risks?"
    ),
}


def _build_interrogation_plan(skeleton: DomainSkeleton) -> list[dict[str, str]]:
    """Build a list of interrogation questions from the skeleton.

    Each question is a dict with 'pattern', 'item', 'prompt'.
    """
    plan: list[dict[str, str]] = []
    domain = skeleton.domain

    for concept in skeleton.concepts:
        for pattern_key in ("concept_mechanisms", "concept_relationships"):
            plan.append({
                "pattern": pattern_key,
                "item": concept,
                "prompt": _INTERROGATION_TEMPLATES[pattern_key].format(
                    item=concept, domain=domain,
                ),
            })

    for proc in skeleton.procedures:
        for pattern_key in ("procedure_exceptions", "procedure_constraints"):
            plan.append({
                "pattern": pattern_key,
                "item": proc,
                "prompt": _INTERROGATION_TEMPLATES[pattern_key].format(
                    item=proc, domain=domain,
                ),
            })

    for risk in skeleton.risks:
        for pattern_key in ("risk_mechanisms", "risk_mitigations"):
            plan.append({
                "pattern": pattern_key,
                "item": risk,
                "prompt": _INTERROGATION_TEMPLATES[pattern_key].format(
                    item=risk, domain=domain,
                ),
            })

    for constraint in skeleton.constraints:
        plan.append({
            "pattern": "constraint_interactions",
            "item": constraint,
            "prompt": _INTERROGATION_TEMPLATES["constraint_interactions"].format(
                item=constraint, domain=domain,
            ),
        })

    return plan


# ── Extractor ────────────────────────────────────────────────────────────────


_EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge extraction engine. Given a knowledge text, extract structured cognitive records.

For each distinct piece of knowledge, output a JSON array of objects with these fields:
- "content": the knowledge statement (1-3 sentences, precise, actionable)
- "unit_type": one of ["Rule", "Procedure", "Constraint", "Exception", "Caution", "RiskSignal", "Fact", "Relationship"]
- "tags": 2-4 domain tags (lowercase, colon-separated hierarchy like "threat:lateral-movement")
- "semantic_type": one of ["fact", "decision", "preference", "contradiction"]
- "confidence_note": "high", "medium", or "low"

Rules:
- Prefer Rule/Exception/Constraint/Caution over Fact/Definition
- Skip generic statements that any practitioner would already know
- Focus on mechanisms ("because"), conditions ("when/unless"), and exceptions ("except when")
- Each record should be independently useful — not a sentence fragment
- Output ONLY the JSON array, no surrounding text"""


def _build_extraction_prompt(llm_response_text: str, domain: str) -> str:
    """Build the extraction prompt for parsing an LLM response into records."""
    return (
        f"{_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"Domain: {domain}\n\n"
        f"Knowledge text to extract from:\n"
        f"---\n{llm_response_text}\n---"
    )


def _parse_extraction_response(text: str | list | Any) -> list[dict[str, Any]]:
    """Parse the extractor's JSON response into record dicts.

    Handles common LLM formatting issues (markdown fences, trailing text).
    Also handles cases where the LLM returns already-parsed structured data.
    """
    # If already a list (structured output), return directly
    if isinstance(text, list):
        return [r for r in text if isinstance(r, dict)]

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        records = json.loads(text)
        if isinstance(records, list):
            return records
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            records = json.loads(text[start : end + 1])
            if isinstance(records, list):
                return records
        except json.JSONDecodeError:
            pass

    return []


_CONFIDENCE_MAP = {"high": 0.80, "medium": 0.60, "low": 0.40}


def _raw_to_pack_record(
    raw: dict[str, Any],
    skeleton: DomainSkeleton,
    source_item: str,
    pattern: str,
    llm_model: str,
) -> dict[str, Any] | None:
    """Convert a raw extracted record dict into a PackRecord-compatible dict.

    Returns None if the record is malformed.
    """
    content = raw.get("content", "").strip()
    if not content or len(content) < 10:
        return None

    unit_type = raw.get("unit_type", "Fact")
    valid_types = {
        "Rule", "Procedure", "Constraint", "Exception",
        "Caution", "RiskSignal", "Fact", "Relationship",
    }
    if unit_type not in valid_types:
        unit_type = "Fact"

    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    # Add skeleton prefix tags
    tags = list(skeleton.tags_prefix) + [t for t in tags if t not in skeleton.tags_prefix]

    semantic_type = raw.get("semantic_type", "fact")
    if semantic_type not in ("fact", "decision", "preference", "contradiction"):
        semantic_type = "fact"

    confidence_note = raw.get("confidence_note", "medium")
    confidence = _CONFIDENCE_MAP.get(confidence_note, 0.60)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return {
        "content": content,
        "unit_type": unit_type,
        "tags": tags,
        "namespace": skeleton.namespace,
        "semantic_type": semantic_type,
        "source_count": 1,
        "source_authority": "Inferred",
        "confidence": confidence,
        "provenance_refs": [
            f"harvester:skeleton:{skeleton.domain}:{source_item}:{pattern}:{llm_model}:{timestamp}"
        ],
    }


# ── Gate (filtering) ─────────────────────────────────────────────────────────


def _filter_candidates(
    candidates: list[dict[str, Any]],
    brain: Any | None = None,
    min_value_score: float = 0.30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter candidate records through Aura's quality gate.

    Returns (accepted_records, filter_stats).
    If brain is available, uses filter_pack_records() for surprise scoring.
    Otherwise falls back to basic deduplication.
    """
    if not candidates:
        return [], {"total": 0, "accepted": 0, "rejected": 0}

    # Deduplicate by content (exact match)
    seen_content: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in candidates:
        key = rec["content"].lower().strip()
        if key not in seen_content:
            seen_content.add(key)
            unique.append(rec)

    duplicates_removed = len(candidates) - len(unique)

    accepted = unique  # If no brain, accept all unique records
    rejected_by_gate = 0

    if brain is not None:
        try:
            from aura import BasePack, filter_pack_records

            # Build a temporary BasePack for scoring
            temp_pack_data = {
                "base_id": "harvester-temp",
                "version": "0.0.1",
                "domain": candidates[0].get("tags", ["unknown"])[0] if candidates else "unknown",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "records": unique,
            }
            temp_pack = BasePack.from_json(json.dumps(temp_pack_data))
            report = filter_pack_records(temp_pack, brain._aura if hasattr(brain, "_aura") else brain)

            # Use scored records — accept those above threshold
            accepted = []
            for scored in report.scored_records:
                if scored.value_score >= min_value_score:
                    # Find the original record by content match
                    for rec in unique:
                        if rec["content"] == scored.content:
                            accepted.append(rec)
                            break
                else:
                    rejected_by_gate += 1

        except (ImportError, Exception) as e:
            logger.warning("filter_pack_records() not available: %s — accepting all unique records", e)

    stats = {
        "total": len(candidates),
        "duplicates_removed": duplicates_removed,
        "unique": len(unique),
        "accepted": len(accepted),
        "rejected_by_gate": rejected_by_gate,
    }

    return accepted, stats


# ── Main Workflow ────────────────────────────────────────────────────────────


def harvest_skeleton(
    skeleton: DomainSkeleton,
    budget: HarvestBudget | None = None,
    brain: Any | None = None,
    llm_caller: Any | None = None,
    llm_model_name: str = "unknown",
    min_value_score: float = 0.30,
) -> tuple[dict[str, Any], HarvestReport]:
    """Run a Mode 1 Skeleton Build harvest.

    Args:
        skeleton: Domain structure to interrogate about.
        budget: Hard budget caps (defaults to HarvestBudget()).
        brain: Optional Aura brain instance for surprise scoring.
        llm_caller: Callable(prompt: str) -> str. If None, uses remy's call_llm.
        llm_model_name: Model name for provenance tracking.
        min_value_score: Minimum value score for acceptance (default 0.30).

    Returns:
        (base_pack_dict, harvest_report)
        base_pack_dict is a BasePack-compatible dict ready for JSON serialization.
    """
    if budget is None:
        budget = HarvestBudget()

    report = HarvestReport(
        domain=skeleton.domain,
        llm_calls_budget=budget.max_llm_calls,
    )

    t_start = time.time()

    # Resolve LLM caller
    if llm_caller is None:
        try:
            from remy.core.llm import call_llm
            llm_caller = lambda prompt: call_llm(prompt, purpose="harvester").content
        except ImportError:
            raise RuntimeError(
                "No llm_caller provided and remy.core.llm not available. "
                "Pass a callable(prompt: str) -> str."
            )

    # Step 1: Build interrogation plan
    plan = _build_interrogation_plan(skeleton)
    logger.info(
        "Harvest plan: %d questions for domain '%s' (budget: %d calls)",
        len(plan), skeleton.domain, budget.max_llm_calls,
    )

    # Step 2: Interrogate + Extract with round-robin fairness
    #
    # Coverage before depth: we iterate skeleton items in round-robin,
    # ensuring every item gets at least one interrogation pass before
    # any item gets a second. Per-item candidate cap prevents early
    # items from consuming the entire budget.
    all_candidates: list[dict[str, Any]] = []

    # Group questions by skeleton item (preserving question order within item)
    from collections import OrderedDict
    item_questions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for q in plan:
        item_questions.setdefault(q["item"], []).append(q)

    num_items = len(item_questions)
    per_item_cap = budget.per_item_candidate_cap
    if per_item_cap <= 0 and num_items > 0:
        # Auto: distribute candidate budget evenly across items.
        # Leave no headroom — strict fairness so every item gets a share.
        per_item_cap = max(10, budget.max_candidate_records // num_items)
    item_candidate_counts: dict[str, int] = {item: 0 for item in item_questions}

    logger.info(
        "Round-robin harvest: %d items, per-item cap: %d candidates",
        num_items, per_item_cap,
    )

    # Round-robin: iterate by pass number, then by item
    max_passes = max((len(qs) for qs in item_questions.values()), default=0)

    for pass_idx in range(max_passes):
        if report.llm_calls_used >= budget.max_llm_calls:
            report.budget_exhausted = True
            logger.warning("LLM call budget exhausted (%d/%d)", report.llm_calls_used, budget.max_llm_calls)
            break

        if len(all_candidates) >= budget.max_candidate_records:
            report.budget_exhausted = True
            logger.warning("Candidate record budget exhausted (%d/%d)", len(all_candidates), budget.max_candidate_records)
            break

        for item_name, questions in item_questions.items():
            if pass_idx >= len(questions):
                continue  # This item has no more questions

            if report.llm_calls_used >= budget.max_llm_calls:
                report.budget_exhausted = True
                break

            if len(all_candidates) >= budget.max_candidate_records:
                report.budget_exhausted = True
                break

            # Per-item cap: skip if this item already has enough candidates
            if item_candidate_counts[item_name] >= per_item_cap:
                logger.info("Item '%s' reached per-item cap (%d), skipping", item_name, per_item_cap)
                continue

            question = questions[pass_idx]

            # Interrogation call
            try:
                response_text = llm_caller(question["prompt"])
                report.llm_calls_used += 1
            except Exception as e:
                report.errors.append(f"Interrogation failed for '{question['item']}': {e}")
                logger.warning("Interrogation failed: %s", e)
                continue

            if report.llm_calls_used >= budget.max_llm_calls:
                report.budget_exhausted = True
                break

            # Extraction call
            if not isinstance(response_text, str):
                response_text = json.dumps(response_text, ensure_ascii=False) if isinstance(response_text, (list, dict)) else str(response_text)
            extraction_prompt = _build_extraction_prompt(response_text, skeleton.domain)
            try:
                extraction_text = llm_caller(extraction_prompt)
                report.llm_calls_used += 1
            except Exception as e:
                report.errors.append(f"Extraction failed for '{question['item']}': {e}")
                logger.warning("Extraction failed: %s", e)
                continue

            # Parse extracted records, respecting per-item cap
            raw_records = _parse_extraction_response(extraction_text)
            for raw in raw_records:
                if item_candidate_counts[item_name] >= per_item_cap:
                    break
                if len(all_candidates) >= budget.max_candidate_records:
                    break

                pack_rec = _raw_to_pack_record(
                    raw, skeleton, question["item"], question["pattern"], llm_model_name,
                )
                if pack_rec is not None:
                    all_candidates.append(pack_rec)
                    item_candidate_counts[item_name] += 1

    report.candidate_records = len(all_candidates)
    logger.info("Harvested %d candidate records in %d LLM calls", len(all_candidates), report.llm_calls_used)

    # Step 3: Filter through gate
    accepted, filter_stats = _filter_candidates(all_candidates, brain, min_value_score)

    # Apply max_accepted cap with per-item fairness.
    # Simple truncation would cut items that appear later in the list
    # (risks, constraints). Instead, distribute the cap evenly across items,
    # then fill remaining slots with the strongest leftover records.
    if len(accepted) > budget.max_accepted_records:
        # Group accepted by source item
        item_buckets: dict[str, list[dict[str, Any]]] = {}
        for rec in accepted:
            item_key = "unknown"
            for ref in rec.get("provenance_refs", []):
                parts = ref.split(":")
                if len(parts) >= 5:
                    item_key = parts[3]
                    break
            item_buckets.setdefault(item_key, []).append(rec)

        n_items = max(len(item_buckets), 1)
        per_item_accept = budget.max_accepted_records // n_items

        fair_accepted: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []

        for item_key, recs in item_buckets.items():
            fair_accepted.extend(recs[:per_item_accept])
            overflow.extend(recs[per_item_accept:])

        # Fill remaining slots from overflow (order preserved)
        remaining = budget.max_accepted_records - len(fair_accepted)
        if remaining > 0:
            fair_accepted.extend(overflow[:remaining])

        accepted = fair_accepted[:budget.max_accepted_records]

    report.accepted_records = len(accepted)
    report.acceptance_rate = (
        len(accepted) / len(all_candidates) if all_candidates else 0.0
    )

    # Step 4: Assemble BasePack dict
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_pack = {
        "base_id": f"{skeleton.domain}-harvested",
        "version": "0.1.0",
        "domain": skeleton.domain,
        "intended_role": skeleton.intended_role,
        "created_at": timestamp,
        "updated_at": timestamp,
        "changelog": (
            f"Auto-harvested from domain skeleton. "
            f"{report.accepted_records} records from {report.candidate_records} candidates. "
            f"LLM calls: {report.llm_calls_used}. "
            f"Acceptance rate: {report.acceptance_rate:.0%}."
        ),
        "records": accepted,
        "initial_concepts": [],
        "initial_cautions": [],
        "canonical_cases": [],
        "metadata": {
            "source_mode": "harvester_skeleton_v1",
            "record_count": len(accepted),
            "concept_count": 0,
            "caution_count": 0,
            "case_count": 0,
            "harvest_report": {
                "llm_calls_used": report.llm_calls_used,
                "candidate_records": report.candidate_records,
                "accepted_records": report.accepted_records,
                "acceptance_rate": round(report.acceptance_rate, 3),
                "filter_stats": filter_stats,
            },
        },
    }

    report.elapsed_seconds = time.time() - t_start

    logger.info(
        "Harvest complete: %d accepted / %d candidates (%.0f%%) in %.1fs, %d LLM calls",
        report.accepted_records,
        report.candidate_records,
        report.acceptance_rate * 100,
        report.elapsed_seconds,
        report.llm_calls_used,
    )

    return base_pack, report
