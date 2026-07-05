"""Load and summarize the active runtime contract artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class RuntimeContract:
    version: str
    name: str
    stages: list[str]
    transitions: list[str]
    stop_conditions: list[str]
    retry_rules: list[str]
    escalation_conditions: list[str]
    required_artifacts: list[str]
    failure_classes: list[str]
    source_path: str

    def to_summary(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "stages": list(self.stages),
            "failure_classes": list(self.failure_classes),
            "required_artifacts": list(self.required_artifacts),
            "source_path": self.source_path,
        }


def _contract_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "runtime_contract.yaml"


@lru_cache(maxsize=1)
def load_runtime_contract() -> RuntimeContract:
    path = _contract_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    return RuntimeContract(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "runtime-contract"),
        stages=[str(x) for x in payload.get("stages", [])],
        transitions=[str(x) for x in payload.get("transitions", [])],
        stop_conditions=[str(x) for x in payload.get("stop_conditions", [])],
        retry_rules=[str(x) for x in payload.get("retry_rules", [])],
        escalation_conditions=[str(x) for x in payload.get("escalation_conditions", [])],
        required_artifacts=[str(x) for x in payload.get("required_artifacts", [])],
        failure_classes=[str(x) for x in payload.get("failure_classes", [])],
        source_path=str(path),
    )


def get_runtime_contract_summary() -> dict:
    return load_runtime_contract().to_summary()
