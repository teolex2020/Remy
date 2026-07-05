"""Scenario-based live validation pack for autonomous job families.

Unlike deterministic benchmarks, this pack stores editable validation scenarios
and produces a memory-aware readiness report for real-world runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from remy.config.settings import settings


@dataclass
class LiveValidationScenario:
    name: str
    category: str
    goal_template: str
    goal: str
    target_url: str = ""
    action: str = "click"
    expected_artifact: str = ""
    notes: str = ""


def _scenario_path() -> Path:
    return settings.DATA_DIR / "autonomy_live_validation_scenarios.json"


def _report_path() -> Path:
    return settings.DATA_DIR / "autonomy_live_validation_report.json"


def _default_scenarios() -> list[LiveValidationScenario]:
    return [
        LiveValidationScenario(
            name="signup_generic",
            category="signup",
            goal_template="signup_operator",
            goal="Register a new account and verify that the final destination is an account/dashboard page.",
            target_url="https://app.example.com/signup",
            action="click",
            expected_artifact="verified account/dashboard destination",
            notes="Replace with a real signup flow that matters for your operator tests.",
        ),
        LiveValidationScenario(
            name="publisher_generic",
            category="publisher",
            goal_template="publisher",
            goal="Publish one post and verify the final public post URL.",
            target_url="https://x.com/compose/post",
            action="click",
            expected_artifact="live public post URL",
            notes="Replace with a real publishing target used in your workflow.",
        ),
        LiveValidationScenario(
            name="market_research_memory_tools",
            category="research",
            goal_template="market_research",
            goal="Research AI memory tools, compare pricing/positioning, and generate a concrete report artifact.",
            target_url="",
            action="",
            expected_artifact="report URL or record_id",
            notes="Use this as a stable recurring research validation task.",
        ),
    ]


def _normalize_scenario(item: dict, fallback_name: str) -> dict:
    name = str(item.get("name") or fallback_name).strip() or fallback_name
    return {
        "name": name,
        "category": str(item.get("category") or "other").strip() or "other",
        "goal_template": str(item.get("goal_template") or "market_research").strip()
        or "market_research",
        "goal": str(item.get("goal") or "").strip(),
        "target_url": str(item.get("target_url") or "").strip(),
        "action": str(item.get("action") or "").strip(),
        "expected_artifact": str(item.get("expected_artifact") or "").strip(),
        "notes": str(item.get("notes") or "").strip(),
    }


def _deduplicate_scenario_names(scenarios: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    deduplicated: list[dict] = []
    for idx, item in enumerate(scenarios, start=1):
        scenario = dict(item)
        base_name = str(scenario.get("name") or f"scenario_{idx}").strip() or f"scenario_{idx}"
        count = seen.get(base_name, 0)
        unique_name = base_name if count == 0 else f"{base_name}_{count + 1}"
        seen[base_name] = count + 1
        scenario["name"] = unique_name
        deduplicated.append(scenario)
    return deduplicated


def load_live_validation_scenarios() -> list[dict]:
    """Load editable scenarios, seeding defaults on first use."""
    path = _scenario_path()
    if not path.exists():
        scenarios = [asdict(item) for item in _default_scenarios()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
        return scenarios
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    scenarios = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        scenarios.append(_normalize_scenario(item, fallback_name=f"scenario_{idx}"))
    return _deduplicate_scenario_names(scenarios)


def save_live_validation_scenarios(scenarios: list[dict]) -> None:
    normalized = _deduplicate_scenario_names(
        [
            _normalize_scenario(item, fallback_name=f"scenario_{idx}")
            for idx, item in enumerate(scenarios or [], start=1)
            if isinstance(item, dict)
        ]
    )
    path = _scenario_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _hard_blocker_counts(failure_hints: list[dict]) -> dict[str, int]:
    tracked = {
        "captcha",
        "email_verification",
        "phone_verification",
        "kyc_verification",
        "payment_block",
    }
    counts: dict[str, int] = {}
    for item in failure_hints:
        signature = str(item.get("signature") or "")
        if signature in tracked:
            counts[signature] = counts.get(signature, 0) + int(item.get("count", 0))
    return counts


def _build_template_checklist(scenario: dict, hints: dict) -> list[str]:
    template = scenario.get("goal_template")
    preferred = hints.get("preferred_selectors") or []
    checklist = []
    if template == "signup_operator":
        checklist.extend(
            [
                "Open the target signup flow and verify the current page URL/origin.",
                "Complete the form, then verify account/dashboard state after submit.",
                "If captcha/email/SMS/payment/KYC appears, mark blocked_external with evidence.",
            ]
        )
    elif template == "publisher":
        checklist.extend(
            [
                "Open the compose/publish flow and draft a single post.",
                "Submit once, then verify the final live/public URL.",
                "Do not treat draft/compose state as completion.",
            ]
        )
    elif template == "market_research":
        checklist.extend(
            [
                "Run focused research queries and store findings with source URLs.",
                "Generate a concrete report artifact rather than a text-only summary.",
                "Verify findings_count and report artifact before marking completion.",
            ]
        )
    if preferred:
        checklist.append(
            f"Prefer selectors such as {preferred[0]['selector']} when the DOM still matches."
        )
    return checklist


def _evaluate_scenario_readiness(scenario: dict) -> dict:
    hints = {
        "domain": None,
        "flow": "navigation",
        "failure_hints": [],
        "success_hints": [],
        "avoided_selectors": [],
        "preferred_selectors": [],
    }
    if scenario.get("target_url"):
        try:
            from remy.core.browser_failure_memory import get_browser_execution_hints

            hints = get_browser_execution_hints(
                url=str(scenario.get("target_url") or ""),
                text=str(scenario.get("goal") or ""),
                action=str(scenario.get("action") or ""),
                limit=3,
            )
        except Exception:
            pass

    success_count = sum(int(item.get("count", 0)) for item in hints.get("success_hints") or [])
    blocker_counts = _hard_blocker_counts(hints.get("failure_hints") or [])
    failure_count = sum(int(item.get("count", 0)) for item in hints.get("failure_hints") or [])

    if success_count > 0:
        status = "ready"
    elif blocker_counts:
        status = "risky"
    elif failure_count > 0:
        status = "unknown"
    else:
        status = "untrained"

    return {
        "name": scenario.get("name"),
        "category": scenario.get("category"),
        "goal_template": scenario.get("goal_template"),
        "goal": scenario.get("goal"),
        "target_url": scenario.get("target_url", ""),
        "expected_artifact": scenario.get("expected_artifact", ""),
        "status": status,
        "domain": hints.get("domain"),
        "flow": hints.get("flow"),
        "known_successes": hints.get("success_hints") or [],
        "known_failures": hints.get("failure_hints") or [],
        "preferred_selectors": hints.get("preferred_selectors") or [],
        "avoided_selectors": hints.get("avoided_selectors") or [],
        "hard_blockers": blocker_counts,
        "checklist": _build_template_checklist(scenario, hints),
        "notes": scenario.get("notes", ""),
    }


def run_live_validation_pack() -> dict:
    """Build a readiness report for scenario-based live validation runs."""
    scenarios = load_live_validation_scenarios()
    results = [_evaluate_scenario_readiness(scenario) for scenario in scenarios]

    summary = {
        "total": len(results),
        "ready": sum(1 for item in results if item["status"] == "ready"),
        "risky": sum(1 for item in results if item["status"] == "risky"),
        "unknown": sum(1 for item in results if item["status"] == "unknown"),
        "untrained": sum(1 for item in results if item["status"] == "untrained"),
    }

    category_summary: dict[str, dict[str, int]] = {}
    for category in sorted({item.get("category", "other") for item in results}):
        subset = [item for item in results if item.get("category", "other") == category]
        category_summary[category] = {
            "total": len(subset),
            "ready": sum(1 for item in subset if item["status"] == "ready"),
            "risky": sum(1 for item in subset if item["status"] == "risky"),
            "unknown": sum(1 for item in subset if item["status"] == "unknown"),
            "untrained": sum(1 for item in subset if item["status"] == "untrained"),
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "category_summary": category_summary,
        "results": results,
    }
    path = _report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_live_validation_report() -> dict | None:
    path = _report_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
