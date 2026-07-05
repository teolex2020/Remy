"""
Simple auth/session state store for integration plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuthRecord:
    plugin_id: str
    auth_mode: str
    configured: bool = False
    account_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AuthStore:
    def __init__(self):
        self._records: dict[str, AuthRecord] = {}

    def set(
        self,
        plugin_id: str,
        *,
        auth_mode: str,
        configured: bool,
        account_label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuthRecord:
        record = AuthRecord(
            plugin_id=plugin_id,
            auth_mode=auth_mode,
            configured=configured,
            account_label=account_label,
            metadata=metadata or {},
        )
        self._records[plugin_id] = record
        return record

    def get(self, plugin_id: str) -> AuthRecord | None:
        return self._records.get(plugin_id)

    def is_configured(self, plugin_id: str) -> bool:
        rec = self.get(plugin_id)
        return bool(rec and rec.configured)

