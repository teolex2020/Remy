"""Load and summarize the deterministic script/adapter registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class RegistryEntry:
    entry_id: str
    label: str
    category: str
    path: str
    purpose: str
    critical: bool

    def to_summary(self) -> dict:
        return {
            "id": self.entry_id,
            "label": self.label,
            "category": self.category,
            "path": self.path,
            "purpose": self.purpose,
            "critical": self.critical,
        }


@dataclass(slots=True)
class ScriptAdapterRegistry:
    version: str
    name: str
    entries: list[RegistryEntry]
    source_path: str

    def to_summary(self) -> dict:
        categories: dict[str, int] = {}
        for item in self.entries:
            categories[item.category] = categories.get(item.category, 0) + 1
        return {
            "version": self.version,
            "name": self.name,
            "count": len(self.entries),
            "critical_count": sum(1 for item in self.entries if item.critical),
            "categories": categories,
            "entries": [item.to_summary() for item in self.entries],
            "source_path": self.source_path,
        }


def _registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "script_adapter_registry.yaml"


@lru_cache(maxsize=1)
def load_script_adapter_registry() -> ScriptAdapterRegistry:
    path = _registry_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    entries = []
    for item in payload.get("entries", []) or []:
        if not isinstance(item, dict):
            continue
        entries.append(
            RegistryEntry(
                entry_id=str(item.get("id") or "").strip(),
                label=str(item.get("label") or item.get("id") or "").strip(),
                category=str(item.get("category") or "").strip(),
                path=str(item.get("path") or "").strip(),
                purpose=str(item.get("purpose") or "").strip(),
                critical=bool(item.get("critical")),
            )
        )
    return ScriptAdapterRegistry(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "script-adapter-registry"),
        entries=entries,
        source_path=str(path),
    )


def get_script_adapter_registry_summary() -> dict:
    return load_script_adapter_registry().to_summary()
