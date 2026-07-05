"""Load and summarize explicit runtime state semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class StateClass:
    state_id: str
    label: str
    owner: str
    persistence: str
    lifecycle: str
    examples: list[str]
    recovery_source: str

    def to_summary(self) -> dict:
        return {
            "id": self.state_id,
            "label": self.label,
            "owner": self.owner,
            "persistence": self.persistence,
            "lifecycle": self.lifecycle,
            "examples": list(self.examples),
            "recovery_source": self.recovery_source,
        }


@dataclass(slots=True)
class StateSemanticsRegistry:
    version: str
    name: str
    states: list[StateClass]
    source_path: str

    def to_summary(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "count": len(self.states),
            "states": [state.to_summary() for state in self.states],
            "durable_count": sum(1 for state in self.states if state.persistence == "durable"),
            "ephemeral_count": sum(1 for state in self.states if state.persistence == "ephemeral"),
            "transient_count": sum(1 for state in self.states if state.persistence == "transient"),
            "source_path": self.source_path,
        }


def _semantics_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "state_semantics.yaml"


@lru_cache(maxsize=1)
def load_state_semantics() -> StateSemanticsRegistry:
    path = _semantics_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    states = []
    for item in payload.get("states", []) or []:
        if not isinstance(item, dict):
            continue
        states.append(
            StateClass(
                state_id=str(item.get("id") or "").strip(),
                label=str(item.get("label") or item.get("id") or "").strip(),
                owner=str(item.get("owner") or "").strip(),
                persistence=str(item.get("persistence") or "").strip(),
                lifecycle=str(item.get("lifecycle") or "").strip(),
                examples=[str(x) for x in item.get("examples", []) or []],
                recovery_source=str(item.get("recovery_source") or "").strip(),
            )
        )
    return StateSemanticsRegistry(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "state-semantics"),
        states=states,
        source_path=str(path),
    )


def get_state_semantics_summary() -> dict:
    return load_state_semantics().to_summary()
