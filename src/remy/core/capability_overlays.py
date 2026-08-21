"""Validated runtime profiles layered over capability packs and skills."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Iterable


_ID_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,79}$")


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    id: str
    label: str
    source: str
    tools: tuple[str, ...] = ()
    prompt_sections: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    id: str
    label: str
    base: str = "channel_default"
    bundles: tuple[str, ...] = ()
    include_tools: tuple[str, ...] = ()
    deny_tools: tuple[str, ...] = ()
    tool_ceiling: tuple[str, ...] = ()
    prompt_sections: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedCapabilityOverlay:
    profile_id: str
    profile_label: str
    bundle_ids: tuple[str, ...]
    tool_names: frozenset[str]
    denied_tools: frozenset[str]
    excluded_tools: frozenset[str]
    unknown_tools: frozenset[str]
    unknown_bundles: tuple[str, ...]
    prompt_sections: tuple[str, ...]
    overlay_hash: str

    def to_summary(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "bundles": list(self.bundle_ids),
            "tools": sorted(self.tool_names),
            "tool_count": len(self.tool_names),
            "denied_tools": sorted(self.denied_tools),
            "excluded_tools": sorted(self.excluded_tools),
            "unknown_tools": sorted(self.unknown_tools),
            "unknown_bundles": list(self.unknown_bundles),
            "prompt_sections": list(self.prompt_sections),
            "overlay_hash": self.overlay_hash,
        }


_READ_ONLY_CEILING = (
    "recall",
    "search",
    "recall_memory_as_of",
    "explain_memory_recall",
    "build_memory_context",
    "web_search",
    "extract_content",
    "http_get",
    "get_current_datetime",
    "browse_page",
    "browser_close",
    "read_file",
    "list_directory",
    "fs_read",
    "fs_search",
    "search_transcript_history",
    "review_history_memory_gaps",
    "list_child_sessions",
    "get_child_report",
    "list_ptc_tools",
    "validate_ptc_program",
    "run_ptc_program",
    "list_available_tools",
    "list_available_skills",
    "list_capability_profiles",
    "activate_capability_profile",
    "enable_skill",
    "enable_tools",
)


_BUILTIN_PROFILES = (
    CapabilityProfile(
        id="standard",
        label="Standard",
        base="channel_default",
        description="Preserves current channel defaults and progressive loading.",
    ),
    CapabilityProfile(
        id="research",
        label="Grounded Research",
        base="core",
        bundles=("pack:market_research", "skill:deep_research"),
        prompt_sections=(
            "Research profile: gather evidence before synthesis and preserve source URLs.",
        ),
        description=(
            "Grounded research with the existing market-research pack and deep-research skill."
        ),
    ),
    CapabilityProfile(
        id="read_only",
        label="Read-only",
        base="core",
        tool_ceiling=_READ_ONLY_CEILING,
        prompt_sections=(
            "READ-ONLY PROFILE: do not mutate files, memory, accounts, browser "
            "forms, or external systems.",
        ),
        description=(
            "Inspection and research only; a hard tool ceiling blocks mutating capabilities."
        ),
    ),
    CapabilityProfile(
        id="operator",
        label="Operator",
        base="all",
        prompt_sections=(
            "Operator profile expands tool visibility but never bypasses approvals, "
            "provenance, or policy guards.",
        ),
        description="Full tool visibility with the normal ToolPipeline safety path intact.",
    ),
)


class CapabilityOverlayRegistry:
    def __init__(self):
        self._bundles: dict[str, CapabilityBundle] = {}
        self._profiles: dict[str, CapabilityProfile] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_id(value: str, kind: str) -> str:
        normalized = str(value or "").strip().lower()
        if not _ID_RE.fullmatch(normalized):
            raise ValueError(f"Invalid {kind} id: {value!r}")
        return normalized

    def register_bundle(self, bundle: CapabilityBundle) -> None:
        bundle_id = self._validate_id(bundle.id, "bundle")
        if not str(bundle.label or "").strip():
            raise ValueError("bundle label is required")
        with self._lock:
            if bundle_id in self._bundles:
                raise ValueError(f"Bundle '{bundle_id}' is already registered")
            self._bundles[bundle_id] = bundle

    def register_profile(self, profile: CapabilityProfile) -> None:
        profile_id = self._validate_id(profile.id, "profile")
        if profile.base not in {"channel_default", "core", "all", "none"}:
            raise ValueError(f"Invalid profile base: {profile.base}")
        with self._lock:
            if profile_id in self._profiles:
                raise ValueError(f"Profile '{profile_id}' is already registered")
            self._profiles[profile_id] = profile

    def get_bundle(self, bundle_id: str) -> CapabilityBundle | None:
        with self._lock:
            return self._bundles.get(str(bundle_id or "").strip().lower())

    def get_profile(self, profile_id: str) -> CapabilityProfile | None:
        with self._lock:
            return self._profiles.get(str(profile_id or "").strip().lower())

    def list_bundles(self) -> list[CapabilityBundle]:
        with self._lock:
            return list(self._bundles.values())

    def list_profiles(self) -> list[CapabilityProfile]:
        with self._lock:
            return list(self._profiles.values())

    def resolve(
        self,
        *,
        profile_id: str,
        channel: str,
        available_tools: Iterable[str],
        core_tools: Iterable[str],
        bundle_ids: Iterable[str] = (),
        session_tools: Iterable[str] = (),
    ) -> ResolvedCapabilityOverlay:
        available = {str(item) for item in available_tools}
        core = {str(item) for item in core_tools} & available
        requested_profile = str(profile_id or "standard").strip().lower()
        profile = self.get_profile(requested_profile)
        if profile is None:
            raise ValueError(f"Unknown capability profile: {requested_profile!r}")

        if profile.base == "all":
            selected = set(available)
        elif profile.base == "core":
            selected = set(core)
        elif profile.base == "none":
            selected = set()
        else:
            selected = set(available if channel in {"autonomous", "proactive"} else core)

        explicit_bundles = sorted({str(bundle_id) for bundle_id in bundle_ids})
        ordered_bundles = list(dict.fromkeys((*profile.bundles, *explicit_bundles)))
        prompt_sections = list(profile.prompt_sections)
        unknown_bundles: list[str] = []
        requested_tools = set(profile.include_tools) | {str(item) for item in session_tools}
        for bundle_id in ordered_bundles:
            bundle = self.get_bundle(bundle_id)
            if bundle is None:
                unknown_bundles.append(str(bundle_id))
                continue
            requested_tools.update(bundle.tools)
            prompt_sections.extend(bundle.prompt_sections)
        selected.update(requested_tools)

        unknown_tools = selected - available
        selected &= available
        before_constraints = set(selected)
        if profile.tool_ceiling:
            selected &= set(profile.tool_ceiling)
        denied = set(profile.deny_tools) & available
        selected -= denied
        excluded = before_constraints - selected

        payload = {
            "profile": profile.id,
            "bundles": ordered_bundles,
            "tools": sorted(selected),
            "denied": sorted(denied),
            "prompts": prompt_sections,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ResolvedCapabilityOverlay(
            profile_id=profile.id,
            profile_label=profile.label,
            bundle_ids=tuple(
                bundle_id for bundle_id in ordered_bundles if self.get_bundle(bundle_id)
            ),
            tool_names=frozenset(selected),
            denied_tools=frozenset(denied),
            excluded_tools=frozenset(excluded),
            unknown_tools=frozenset(unknown_tools),
            unknown_bundles=tuple(unknown_bundles),
            prompt_sections=tuple(dict.fromkeys(section for section in prompt_sections if section)),
            overlay_hash=digest,
        )


def build_overlay_registry() -> CapabilityOverlayRegistry:
    from remy.core.capability_packs import get_all_packs
    from remy.core.skill_catalog import SKILL_CATALOG

    registry = CapabilityOverlayRegistry()
    for pack in get_all_packs(include_disabled=False).values():
        registry.register_bundle(
            CapabilityBundle(
                id=f"pack:{pack.id}",
                label=pack.label,
                source="capability_pack",
                tools=tuple(pack.tools),
                prompt_sections=tuple(pack.guardrails),
                metadata={
                    "worker": pack.worker,
                    "approval_mode": pack.approval_mode,
                    "step_budget": pack.step_budget,
                    "timeout_sec": pack.timeout_sec,
                },
            )
        )
    for name, skill in SKILL_CATALOG.items():
        registry.register_bundle(
            CapabilityBundle(
                id=f"skill:{name}",
                label=name.replace("_", " ").title(),
                source="skill_catalog",
                tools=tuple(str(tool) for tool in skill.get("tools", ())),
                prompt_sections=(str(skill.get("description") or ""),),
            )
        )
    for profile in _BUILTIN_PROFILES:
        registry.register_profile(profile)
    return registry


_registry: CapabilityOverlayRegistry | None = None
_registry_lock = threading.Lock()


def get_overlay_registry() -> CapabilityOverlayRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = build_overlay_registry()
        return _registry


def refresh_overlay_registry() -> CapabilityOverlayRegistry:
    global _registry
    with _registry_lock:
        _registry = build_overlay_registry()
        return _registry


def list_capability_profiles() -> list[dict]:
    return [
        {
            "id": profile.id,
            "label": profile.label,
            "description": profile.description,
            "base": profile.base,
            "bundles": list(profile.bundles),
            "hard_tool_ceiling": bool(profile.tool_ceiling),
        }
        for profile in get_overlay_registry().list_profiles()
    ]


def format_overlay_prompt(overlay: ResolvedCapabilityOverlay) -> str:
    if overlay.profile_id == "standard" and not overlay.bundle_ids:
        return ""
    lines = [
        "=== ACTIVE CAPABILITY OVERLAY ===",
        f"Profile: {overlay.profile_label} ({overlay.profile_id})",
        f"Bundles: {', '.join(overlay.bundle_ids) or 'none'}",
        "This overlay controls tool visibility only. It cannot override policy, "
        "approvals, provenance, or permissions.",
    ]
    if overlay.prompt_sections:
        lines.append("Profile and bundle instructions:")
        lines.extend(f"- {section}" for section in overlay.prompt_sections)
    if overlay.excluded_tools:
        lines.append(
            "Capabilities excluded by the active profile: "
            + ", ".join(sorted(overlay.excluded_tools))
        )
    return "\n".join(lines)
