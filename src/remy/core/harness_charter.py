"""Load and summarize the shared runtime harness charter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class HarnessCharter:
    version: str
    name: str
    lifecycle_semantics: list[str]
    approval_semantics: list[str]
    retry_semantics: list[str]
    child_agent_semantics: list[str]
    shutdown_semantics: list[str]
    artifact_persistence_expectations: list[str]
    source_path: str

    def to_summary(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "lifecycle_semantics": list(self.lifecycle_semantics),
            "approval_semantics": list(self.approval_semantics),
            "retry_semantics": list(self.retry_semantics),
            "child_agent_semantics": list(self.child_agent_semantics),
            "shutdown_semantics": list(self.shutdown_semantics),
            "artifact_persistence_expectations": list(self.artifact_persistence_expectations),
            "source_path": self.source_path,
        }


def _charter_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "harness_charter.yaml"


@lru_cache(maxsize=1)
def load_harness_charter() -> HarnessCharter:
    path = _charter_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    return HarnessCharter(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "harness-charter"),
        lifecycle_semantics=[str(x) for x in payload.get("lifecycle_semantics", []) or []],
        approval_semantics=[str(x) for x in payload.get("approval_semantics", []) or []],
        retry_semantics=[str(x) for x in payload.get("retry_semantics", []) or []],
        child_agent_semantics=[str(x) for x in payload.get("child_agent_semantics", []) or []],
        shutdown_semantics=[str(x) for x in payload.get("shutdown_semantics", []) or []],
        artifact_persistence_expectations=[str(x) for x in payload.get("artifact_persistence_expectations", []) or []],
        source_path=str(path),
    )


def get_harness_charter_summary() -> dict:
    return load_harness_charter().to_summary()
