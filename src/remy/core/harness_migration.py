"""Load and summarize harness diff and migration discipline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class HarnessArtifactVersion:
    artifact_id: str
    version: str

    def to_summary(self) -> dict:
        return {"id": self.artifact_id, "version": self.version}


@dataclass(slots=True)
class HarnessMigrationManifest:
    version: str
    name: str
    artifacts: list[HarnessArtifactVersion]
    runtime_behavior_changes: list[str]
    migration_notes: list[str]
    source_path: str

    def to_summary(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "artifact_count": len(self.artifacts),
            "artifacts": [artifact.to_summary() for artifact in self.artifacts],
            "runtime_behavior_changes": list(self.runtime_behavior_changes),
            "migration_notes": list(self.migration_notes),
            "source_path": self.source_path,
        }


def _manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "harness_migration_manifest.yaml"


@lru_cache(maxsize=1)
def load_harness_migration_manifest() -> HarnessMigrationManifest:
    path = _manifest_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    artifacts = []
    for item in payload.get("artifacts", []) or []:
        if not isinstance(item, dict):
            continue
        artifacts.append(
            HarnessArtifactVersion(
                artifact_id=str(item.get("id") or "").strip(),
                version=str(item.get("version") or "unknown").strip(),
            )
        )
    return HarnessMigrationManifest(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "harness-migration-manifest"),
        artifacts=artifacts,
        runtime_behavior_changes=[str(x) for x in payload.get("runtime_behavior_changes", []) or []],
        migration_notes=[str(x) for x in payload.get("migration_notes", []) or []],
        source_path=str(path),
    )


def get_harness_migration_summary() -> dict:
    return load_harness_migration_manifest().to_summary()
