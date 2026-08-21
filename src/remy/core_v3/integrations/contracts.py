"""
Contracts for Remy v3 integrations.

Integration plugins are capability providers behind a single gateway. The
gateway handles governance, routing, audit, and budget decisions; plugins focus
on adapter behavior and execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .plugin_context import PluginContext


class AuthMode(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    APP_PASSWORD = "app_password"
    OAUTH = "oauth"
    BROWSER_SESSION = "browser_session"
    MANUAL_LOGIN = "manual_login"


class ExecutionMode(str, Enum):
    API = "api"
    PROTOCOL = "protocol"
    SDK = "sdk"
    BROWSER = "browser"
    HUMAN_ASSISTED = "human_assisted"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class IntegrationDecision(str, Enum):
    ALLOW = "allow"
    APPROVAL_REQUIRED = "approval_required"
    BLOCKED = "blocked"
    MANUAL_ASSIST_REQUIRED = "manual_assist_required"


@dataclass(frozen=True)
class PluginCapability:
    name: str
    description: str = ""


@dataclass
class PluginRequest:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    mode: ExecutionMode | None = None
    cost_estimate_usd: float = 0.0
    tools: list[str] = field(default_factory=list)


@dataclass
class PluginResult:
    ok: bool
    status: str = "success"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    requires_human: bool = False
    manual_reason: str = ""


class BaseIntegrationPlugin(ABC):
    plugin_id: str = "base"
    label: str = "Base Plugin"
    auth_mode: AuthMode = AuthMode.NONE
    default_mode: ExecutionMode = ExecutionMode.API
    risk_level: str = "low"
    capabilities: tuple[PluginCapability, ...] = ()

    def setup(self, ctx: PluginContext) -> Callable[[], None] | None:
        """Register plugin-owned resources and optionally return extra cleanup."""

        return None

    def teardown(self, ctx: PluginContext) -> None:
        """Final plugin cleanup after registered resources are disposed."""

        return None

    @abstractmethod
    def supports(self, action: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def estimate_cost(self, action: str, payload: dict[str, Any] | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        raise NotImplementedError

    def health_check(self) -> HealthStatus:
        return HealthStatus.HEALTHY

    def requires_manual_assist(self, request: PluginRequest) -> str | None:
        return None
