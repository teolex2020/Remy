"""Load and summarize explicit runtime role contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class RoleContract:
    role_id: str
    label: str
    allowed_tools: list[str]
    required_output: str
    required_evidence: str
    verification_owner: str
    failure_owner: str
    escalation_path: list[str]

    def to_summary(self) -> dict:
        return {
            "id": self.role_id,
            "label": self.label,
            "allowed_tools": list(self.allowed_tools),
            "required_output": self.required_output,
            "required_evidence": self.required_evidence,
            "verification_owner": self.verification_owner,
            "failure_owner": self.failure_owner,
            "escalation_path": list(self.escalation_path),
        }


@dataclass(slots=True)
class RoleContractRegistry:
    version: str
    name: str
    roles: list[RoleContract]
    source_path: str

    def get(self, role_id: str) -> RoleContract | None:
        wanted = str(role_id or "").strip().lower()
        if not wanted:
            return None
        for role in self.roles:
            if role.role_id.strip().lower() == wanted:
                return role
        return None

    def to_summary(self, *, current_role: str = "") -> dict:
        current = self.get(current_role)
        return {
            "version": self.version,
            "name": self.name,
            "count": len(self.roles),
            "roles": [role.to_summary() for role in self.roles],
            "current_role": current.to_summary() if current else None,
            "source_path": self.source_path,
        }


def _contracts_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "role_contracts.yaml"


@lru_cache(maxsize=1)
def load_role_contracts() -> RoleContractRegistry:
    path = _contracts_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    roles = []
    for item in payload.get("roles", []) or []:
        if not isinstance(item, dict):
            continue
        roles.append(
            RoleContract(
                role_id=str(item.get("id") or "").strip(),
                label=str(item.get("label") or item.get("id") or "").strip(),
                allowed_tools=[str(x) for x in item.get("allowed_tools", []) or []],
                required_output=str(item.get("required_output") or "").strip(),
                required_evidence=str(item.get("required_evidence") or "").strip(),
                verification_owner=str(item.get("verification_owner") or "").strip(),
                failure_owner=str(item.get("failure_owner") or "").strip(),
                escalation_path=[str(x) for x in item.get("escalation_path", []) or []],
            )
        )
    return RoleContractRegistry(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "role-contracts"),
        roles=roles,
        source_path=str(path),
    )


def get_role_contracts_summary(*, current_role: str = "") -> dict:
    return load_role_contracts().to_summary(current_role=current_role)
