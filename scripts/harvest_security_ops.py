"""
First real domain harvest: Security Operations.

End-to-end validation of the Knowledge Harvester pipeline:
  DomainSkeleton → Interrogator → Extractor → Gate → BasePack JSON

Usage:
  cd <repo>/app
  python scripts/harvest_security_ops.py

Output:
  harvested_packs/security-ops-harvested.json
  harvested_packs/security-ops-harvest-report.json
"""

import json
import os
import sys
import logging

# Ensure project root and src are on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.authoring.knowledge_harvester import (
    DomainSkeleton,
    HarvestBudget,
    harvest_skeleton,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("harvest_security_ops")


# ── Domain Skeleton: Security Operations ─────────────────────────────────────

SECURITY_OPS_SKELETON = DomainSkeleton(
    domain="security_operations",
    intended_role="security_operations_analyst",
    namespace="default",
    tags_prefix=["domain:security-ops"],
    concepts=[
        "incident triage and classification",
        "lateral movement detection",
        "privilege escalation patterns",
        "data exfiltration indicators",
        "phishing campaign analysis",
        "endpoint detection and response",
        "threat intelligence integration",
    ],
    procedures=[
        "incident response initiation within SLA",
        "containment of compromised hosts",
        "forensic evidence preservation",
        "post-incident review process",
        "vulnerability disclosure handling",
    ],
    risks=[
        "alert fatigue leading to missed real threats",
        "insider threat from privileged accounts",
        "supply chain compromise via third-party tools",
        "ransomware propagation through network shares",
        "credential stuffing from leaked databases",
    ],
    constraints=[
        "compliance requirement for 72-hour breach notification",
        "evidence must be legally admissible",
        "patching windows limited to maintenance hours",
        "budget cap on external forensic engagement",
    ],
)

# ── Budget: conservative for first run ───────────────────────────────────────

BUDGET = HarvestBudget(
    max_llm_calls=100,        # 38 questions × 2 = 76 calls + headroom for retries
    max_candidate_records=1600,
    max_accepted_records=400,
    per_item_candidate_cap=76, # 1600/21 ≈ 76 — strict fairness, ~2 question pairs per item
)


# ── LLM Caller ──────────────────────────────────────────────────────────────

def make_llm_caller():
    """Build an LLM caller using remy's infrastructure."""
    from remy.core.llm import call_llm

    def caller(prompt: str):
        result = call_llm(prompt, purpose="harvester")
        content = result.content
        # content may be a string or already-parsed structured data
        return content

    return caller


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "harvested_packs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Security Operations Domain Harvest")
    logger.info("=" * 60)
    logger.info("Concepts:     %d", len(SECURITY_OPS_SKELETON.concepts))
    logger.info("Procedures:   %d", len(SECURITY_OPS_SKELETON.procedures))
    logger.info("Risks:        %d", len(SECURITY_OPS_SKELETON.risks))
    logger.info("Constraints:  %d", len(SECURITY_OPS_SKELETON.constraints))
    logger.info("Budget:       %d LLM calls, %d max candidates, %d max accepted",
                BUDGET.max_llm_calls, BUDGET.max_candidate_records, BUDGET.max_accepted_records)
    logger.info("-" * 60)

    try:
        llm_caller = make_llm_caller()
    except Exception as e:
        logger.error("Failed to initialize LLM caller: %s", e)
        logger.info("Falling back to dry-run mode (no LLM calls)")
        llm_caller = None

    if llm_caller is None:
        logger.info("DRY RUN — showing interrogation plan only")
        from tools.authoring.knowledge_harvester import _build_interrogation_plan
        plan = _build_interrogation_plan(SECURITY_OPS_SKELETON)
        logger.info("Interrogation plan: %d questions", len(plan))
        for i, q in enumerate(plan):
            logger.info("  [%02d] %s | %s", i + 1, q["pattern"], q["item"])
        return

    # Run harvest
    pack_dict, report = harvest_skeleton(
        skeleton=SECURITY_OPS_SKELETON,
        budget=BUDGET,
        llm_caller=llm_caller,
        llm_model_name="gemini",
        min_value_score=0.30,
    )

    # Save outputs
    pack_path = os.path.join(output_dir, "security-ops-harvested.json")
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(pack_dict, f, indent=2, ensure_ascii=False)
    logger.info("Pack saved: %s", pack_path)

    report_path = os.path.join(output_dir, "security-ops-harvest-report.json")
    report_dict = {
        "domain": report.domain,
        "llm_calls_used": report.llm_calls_used,
        "llm_calls_budget": report.llm_calls_budget,
        "candidate_records": report.candidate_records,
        "accepted_records": report.accepted_records,
        "acceptance_rate": round(report.acceptance_rate, 3),
        "budget_exhausted": report.budget_exhausted,
        "errors": report.errors,
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info("Report saved: %s", report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("HARVEST SUMMARY")
    logger.info("=" * 60)
    logger.info("LLM calls:       %d / %d", report.llm_calls_used, report.llm_calls_budget)
    logger.info("Candidates:      %d", report.candidate_records)
    logger.info("Accepted:        %d", report.accepted_records)
    logger.info("Acceptance rate: %.0f%%", report.acceptance_rate * 100)
    logger.info("Budget exhausted: %s", report.budget_exhausted)
    logger.info("Errors:          %d", len(report.errors))
    logger.info("Elapsed:         %.1fs", report.elapsed_seconds)

    if report.errors:
        logger.info("-" * 60)
        logger.info("ERRORS:")
        for err in report.errors:
            logger.info("  - %s", err)

    # Unit type distribution
    unit_types: dict[str, int] = {}
    for rec in pack_dict["records"]:
        ut = rec.get("unit_type", "Unknown")
        unit_types[ut] = unit_types.get(ut, 0) + 1

    logger.info("-" * 60)
    logger.info("UNIT TYPE DISTRIBUTION:")
    for ut, count in sorted(unit_types.items(), key=lambda x: -x[1]):
        logger.info("  %-15s %d", ut, count)

    # Confidence distribution
    high = sum(1 for r in pack_dict["records"] if r.get("confidence", 0) >= 0.75)
    medium = sum(1 for r in pack_dict["records"] if 0.50 <= r.get("confidence", 0) < 0.75)
    low = sum(1 for r in pack_dict["records"] if r.get("confidence", 0) < 0.50)
    logger.info("-" * 60)
    logger.info("CONFIDENCE DISTRIBUTION:")
    logger.info("  High (>=0.75):    %d", high)
    logger.info("  Medium (0.50-0.75): %d", medium)
    logger.info("  Low (<0.50):      %d", low)

    logger.info("=" * 60)

    # Success criteria check
    if report.acceptance_rate >= 0.60:
        logger.info("SUCCESS: Acceptance rate >= 60%% — pack is viable")
    else:
        logger.warning("WARNING: Acceptance rate < 60%% — review filter thresholds")


if __name__ == "__main__":
    main()
