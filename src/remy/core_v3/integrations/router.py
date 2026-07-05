"""
Routing logic for integration execution.
"""

from __future__ import annotations

from .contracts import ExecutionMode, PluginRequest

STRICT_LOGIN_DOMAINS = frozenset({"google", "gmail", "apple", "meta", "facebook", "instagram"})


def classify_domain(site: str) -> str:
    lower = (site or "").lower()
    for marker in STRICT_LOGIN_DOMAINS:
        if marker in lower:
            return marker
    return ""


def choose_mode(plugin_default: ExecutionMode, request: PluginRequest) -> ExecutionMode:
    return request.mode or plugin_default


def requires_manual_login(site: str, action: str) -> bool:
    action = (action or "").lower()
    if "login" not in action and "signin" not in action:
        return False
    return bool(classify_domain(site))

