"""Load and summarize composable harness modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class HarnessModule:
    module_id: str
    label: str
    status: str
    responsibility: str
    artifacts: list[str]

    def to_summary(self) -> dict:
        return {
            "id": self.module_id,
            "label": self.label,
            "status": self.status,
            "responsibility": self.responsibility,
            "artifacts": list(self.artifacts),
        }


@dataclass(slots=True)
class HarnessModuleRegistry:
    version: str
    name: str
    modules: list[HarnessModule]
    source_path: str

    def to_summary(self) -> dict:
        active = [module for module in self.modules if module.status == "active"]
        return {
            "version": self.version,
            "name": self.name,
            "count": len(self.modules),
            "active_count": len(active),
            "modules": [module.to_summary() for module in self.modules],
            "source_path": self.source_path,
        }


def _modules_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "harness_modules.yaml"


@lru_cache(maxsize=1)
def load_harness_modules() -> HarnessModuleRegistry:
    path = _modules_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    modules = []
    for item in payload.get("modules", []) or []:
        if not isinstance(item, dict):
            continue
        modules.append(
            HarnessModule(
                module_id=str(item.get("id") or "").strip(),
                label=str(item.get("label") or item.get("id") or "").strip(),
                status=str(item.get("status") or "unknown").strip(),
                responsibility=str(item.get("responsibility") or "").strip(),
                artifacts=[str(x) for x in item.get("artifacts", []) or []],
            )
        )
    return HarnessModuleRegistry(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "harness-modules"),
        modules=modules,
        source_path=str(path),
    )


def get_harness_modules_summary() -> dict:
    return load_harness_modules().to_summary()


def derive_active_harness_module(*, runtime_snapshot: dict | None = None, memory_status: dict | None = None) -> dict:
    runtime_snapshot = runtime_snapshot or {}
    memory_status = memory_status or {}
    auto = runtime_snapshot.get("autonomy", runtime_snapshot) if isinstance(runtime_snapshot, dict) else {}
    current_role = str((auto.get("current_role") or "")).strip().lower()
    research_session = auto.get("research_session") if isinstance(auto.get("research_session"), dict) else {}
    current_goal = auto.get("current_goal") if isinstance(auto.get("current_goal"), dict) else {}
    current_mission = auto.get("current_mission") if isinstance(auto.get("current_mission"), dict) else {}
    current_task = auto.get("current_task") if isinstance(auto.get("current_task"), dict) else {}
    current_step = auto.get("current_step") if isinstance(auto.get("current_step"), dict) else {}
    approvals = auto.get("approval_queue") if isinstance(auto.get("approval_queue"), list) else []

    history_review = memory_status.get("history_review") if isinstance(memory_status.get("history_review"), dict) else {}
    corrections = memory_status.get("corrections") if isinstance(memory_status.get("corrections"), dict) else {}
    verification = memory_status.get("verification") if isinstance(memory_status.get("verification"), dict) else {}

    if research_session.get("topic") or "research" in current_role:
        return {
            "id": "research",
            "label": "Research module",
            "reason": research_session.get("topic") or current_role or "research flow is active",
        }

    if "verifier" in current_role or "reviewer" in current_role or int(verification.get("repair_required_count", 0) or 0) > 0:
        return {
            "id": "verification",
            "label": "Verification module",
            "reason": "verification or review pressure is active",
        }

    if (
        int(history_review.get("missing_candidates_count", 0) or 0) > 0
        or str((verification.get("last_reconstruction") or {}).get("status") or "").strip() != ""
    ):
        return {
            "id": "recovery",
            "label": "Recovery module",
            "reason": "memory reconstruction or replay review is active",
        }

    if (
        int(corrections.get("review_queue_count", 0) or 0) > 0
        or int(corrections.get("recently_corrected_count", 0) or 0) > 0
    ):
        return {
            "id": "correction",
            "label": "Correction module",
            "reason": "correction feedback or review queue is active",
        }

    if approvals or current_goal or current_mission or current_task or current_step or current_role:
        return {
            "id": "planning",
            "label": "Planning module",
            "reason": "the runtime is routing or executing the current plan",
        }

    return {
        "id": "",
        "label": "",
        "reason": "",
    }
