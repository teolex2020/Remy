from remy.core.loop_detection import (
    LoopDetectionState,
    detect_loop,
    extract_fingerprint,
    format_loop_warning_for_prompt,
)


def test_exact_repeated_cycle_warns_on_second_run():
    state = LoopDetectionState()
    session_log = [
        {
            "type": "tool_call",
            "tool": "tool_status",
            "args": {"scope": "health"},
        }
    ]

    first = detect_loop(state, extract_fingerprint(session_log, cycle_num=1))
    second = detect_loop(state, extract_fingerprint(session_log, cycle_num=2))

    assert first["level"] == "none"
    assert second["level"] == "warning"
    assert second["repetition_count"] == 2
    assert second["sha256_match"] is True
    assert second["repeated_tools"] == ("tool_status",)


def test_format_loop_warning_for_prompt():
    assert format_loop_warning_for_prompt({"level": "none"}) == ""

    warning = format_loop_warning_for_prompt(
        {
            "level": "warning",
            "repetition_count": 3,
            "repeated_tools": ("recall",),
        }
    )

    assert "LOOP DETECTION: WARNING" in warning
    assert "recall" in warning
    assert "3 consecutive cycles" in warning
