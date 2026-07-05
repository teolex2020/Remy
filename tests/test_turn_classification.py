from remy.core.turn_classification import TurnClass, classify_turn


def test_classifies_zero_tool_cycle_as_idle():
    assert classify_turn([]) == TurnClass.IDLE


def test_classifies_productive_tool_with_priority():
    log = [
        {"type": "tool_call", "tool": "tool_status"},
        {"type": "tool_call", "tool": "web_search"},
    ]

    assert classify_turn(log) == TurnClass.PRODUCTIVE


def test_classifies_maintenance_tools():
    log = [{"type": "tool_call", "tool": "tool_status"}]

    assert classify_turn(log) == TurnClass.MAINTENANCE


def test_unknown_tools_are_idle():
    log = [{"type": "tool_call", "tool": "unknown_internal_probe"}]

    assert classify_turn(log) == TurnClass.IDLE
