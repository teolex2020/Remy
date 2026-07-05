from remy.core.runtime_event_contract import build_runtime_event


def test_build_runtime_event_produces_typed_envelope():
    event = build_runtime_event(
        "operator_alert",
        event_domain="operator",
        level="warning",
        payload={"message": "Gateway degraded"},
        legacy_fields={"message": "Gateway degraded"},
        timestamp=123.0,
    )

    assert event["schema"] == "remy.runtime.event"
    assert event["schema_version"] == 1
    assert event["type"] == "operator_alert"
    assert event["event_name"] == "operator_alert"
    assert event["event_domain"] == "operator"
    assert event["payload"]["message"] == "Gateway degraded"
    assert event["message"] == "Gateway degraded"
    assert event["timestamp"] == 123.0
