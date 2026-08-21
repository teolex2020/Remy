"""Contract tests for capability profiles and bundle overlays."""

import pytest

from remy.core.capability_overlays import (
    CapabilityBundle,
    CapabilityOverlayRegistry,
    CapabilityProfile,
    format_overlay_prompt,
    get_overlay_registry,
    list_capability_profiles,
)
from remy.core.brain_tools import BRAIN_TOOLS, CORE_TOOL_NAMES


def _available() -> set[str]:
    return {tool.name for tool in BRAIN_TOOLS}


def test_standard_profile_preserves_channel_defaults_and_session_tools():
    registry = get_overlay_registry()
    extended = next(iter(_available() - CORE_TOOL_NAMES))

    desktop = registry.resolve(
        profile_id="standard",
        channel="desktop",
        available_tools=_available(),
        core_tools=CORE_TOOL_NAMES,
        session_tools={extended},
    )
    autonomous = registry.resolve(
        profile_id="standard",
        channel="autonomous",
        available_tools=_available(),
        core_tools=CORE_TOOL_NAMES,
    )

    assert desktop.tool_names == frozenset(CORE_TOOL_NAMES | {extended})
    assert autonomous.tool_names == frozenset(_available())


def test_research_profile_composes_existing_pack_and_skill():
    overlay = get_overlay_registry().resolve(
        profile_id="research",
        channel="desktop",
        available_tools=_available(),
        core_tools=CORE_TOOL_NAMES,
    )

    assert overlay.bundle_ids == ("pack:market_research", "skill:deep_research")
    assert {"web_search", "extract_content", "start_research"} <= overlay.tool_names
    assert "get_research_status" in overlay.unknown_tools
    assert "Research profile" in format_overlay_prompt(overlay)


def test_read_only_ceiling_wins_over_session_and_skill_additions():
    overlay = get_overlay_registry().resolve(
        profile_id="read_only",
        channel="autonomous",
        available_tools=_available(),
        core_tools=CORE_TOOL_NAMES,
        bundle_ids={"skill:project_work"},
        session_tools={"write_file", "store", "browser_act"},
    )

    assert {"write_file", "store", "browser_act"}.isdisjoint(overlay.tool_names)
    assert {"write_file", "store", "browser_act"} <= overlay.excluded_tools
    assert {
        "read_file",
        "list_directory",
        "list_child_sessions",
        "get_child_report",
    } <= overlay.tool_names
    assert {
        "follow_up_child_session",
        "interrupt_child_session",
        "resume_child_session",
    }.isdisjoint(overlay.tool_names)
    prompt = format_overlay_prompt(overlay)
    assert "READ-ONLY PROFILE" in prompt
    assert "cannot override policy" in prompt


def test_deny_is_last_and_overlay_hash_is_deterministic():
    registry = CapabilityOverlayRegistry()
    registry.register_bundle(
        CapabilityBundle(id="bundle:test", label="Test", source="test", tools=("b", "c"))
    )
    registry.register_bundle(
        CapabilityBundle(id="bundle:extra", label="Extra", source="test", tools=("d",))
    )
    registry.register_profile(
        CapabilityProfile(
            id="profile_test",
            label="Test",
            base="core",
            bundles=("bundle:test",),
            include_tools=("d",),
            deny_tools=("b",),
        )
    )

    first = registry.resolve(
        profile_id="profile_test",
        channel="desktop",
        available_tools={"a", "b", "c", "d"},
        core_tools={"a", "b"},
        bundle_ids=["bundle:extra"],
        session_tools={"c", "d"},
    )
    second = registry.resolve(
        profile_id="profile_test",
        channel="desktop",
        available_tools={"d", "c", "b", "a"},
        core_tools={"b", "a"},
        bundle_ids={"bundle:extra"},
        session_tools={"d", "c"},
    )

    assert first.tool_names == frozenset({"a", "c", "d"})
    assert "b" in first.denied_tools
    assert first.overlay_hash == second.overlay_hash


def test_registry_validates_ids_and_duplicates():
    registry = CapabilityOverlayRegistry()
    with pytest.raises(ValueError, match="Invalid bundle id"):
        registry.register_bundle(CapabilityBundle(id="Bad ID", label="Bad", source="test"))

    profile = CapabilityProfile(id="test_profile", label="Test")
    registry.register_profile(profile)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_profile(profile)
    with pytest.raises(ValueError, match="Unknown capability profile"):
        registry.resolve(
            profile_id="missing",
            channel="desktop",
            available_tools={"recall"},
            core_tools={"recall"},
        )


def test_profile_catalog_exposes_builtins_and_hard_ceiling():
    profiles = {item["id"]: item for item in list_capability_profiles()}

    assert {"standard", "research", "read_only", "operator"} <= profiles.keys()
    assert profiles["read_only"]["hard_tool_ceiling"] is True
    assert profiles["standard"]["hard_tool_ceiling"] is False
