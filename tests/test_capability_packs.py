from remy.core.capability_packs import (
    PUBLISHER,
    format_publisher_playbook_for_prompt,
    get_publisher_playbook,
    infer_publisher_channel,
    infer_publisher_mode,
)


def test_infer_publisher_mode_article():
    goal = {"description": "Draft a Dev.to article comparing AuraSDK and Mem0"}
    assert infer_publisher_mode(goal) == "article"


def test_infer_publisher_mode_comment():
    goal = {"task_action": "Draft a Reddit comment reply for this thread"}
    assert infer_publisher_mode(goal) == "comment"


def test_infer_publisher_channel_from_url():
    goal = {"target_url": "https://dev.to/new"}
    assert infer_publisher_channel(goal) == "devto"


def test_get_publisher_playbook_returns_rules():
    goal = {
        "description": "Draft a Twitter post about memory agents",
        "target_url": "https://x.com/compose/post",
    }
    playbook = get_publisher_playbook(goal)
    assert playbook["mode"] == "post"
    assert playbook["channel"] == "x"
    assert playbook["rules"]


def test_format_publisher_playbook_for_prompt():
    goal = {
        "description": "Write a Dev.to article about AuraSDK",
        "target_url": "https://dev.to/new",
    }
    prompt = format_publisher_playbook_for_prompt(goal)
    assert "PUBLISHER MODE" in prompt
    assert "Mode: article" in prompt
    assert "Channel: devto" in prompt
    assert "structure" in prompt.lower() or "technical" in prompt.lower()


def test_publisher_pack_exposes_modes_and_channels():
    assert "comment" in PUBLISHER.modes
    assert "devto" in PUBLISHER.channels
