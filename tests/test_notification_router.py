"""Tests for notification router web-runtime suppression and alert routing."""

import importlib
from unittest.mock import MagicMock, patch


def test_should_notify_telegram_suppressed_when_web_runtime_enabled():
    from remy.core.notification_router import set_web_runtime_enabled, should_notify_telegram

    try:
        set_web_runtime_enabled(True)
        with patch("remy.core.notification_router._telegram_configured", return_value=True):
            with patch("remy.core.notification_router.is_user_active_on_web", return_value=False):
                assert should_notify_telegram() is False
    finally:
        set_web_runtime_enabled(False)


def test_should_notify_telegram_when_web_runtime_disabled():
    from remy.core.notification_router import set_web_runtime_enabled, should_notify_telegram

    set_web_runtime_enabled(False)
    with patch("remy.core.notification_router._telegram_configured", return_value=True), \
         patch("remy.core.notification_router.is_web_runtime_available", return_value=False), \
         patch("remy.core.notification_router.is_user_active_on_web", return_value=False):
        assert should_notify_telegram() is True


def test_operator_alert_info_suppressed_for_telegram_by_default():
    from remy.core.notification_router import should_send_telegram_for_event

    fake_settings = MagicMock()
    fake_settings.TELEGRAM_OPERATOR_ALERT_MIN_LEVEL = "warning"

    with patch("remy.core.notification_router.should_notify_telegram", return_value=True), \
         patch("remy.config.settings.settings", fake_settings):
        assert should_send_telegram_for_event("operator_alert", "info") is False


def test_operator_alert_warning_sent_to_telegram():
    from remy.core.notification_router import should_send_telegram_for_event

    fake_settings = MagicMock()
    fake_settings.TELEGRAM_OPERATOR_ALERT_MIN_LEVEL = "warning"

    with patch("remy.core.notification_router.should_notify_telegram", return_value=True), \
         patch("remy.config.settings.settings", fake_settings):
        assert should_send_telegram_for_event("operator_alert", "warning") is True


def test_non_operator_notification_not_filtered_by_operator_threshold():
    from remy.core.notification_router import should_send_telegram_for_event

    fake_settings = MagicMock()
    fake_settings.TELEGRAM_OPERATOR_ALERT_MIN_LEVEL = "critical"

    with patch("remy.core.notification_router.should_notify_telegram", return_value=True), \
         patch("remy.config.settings.settings", fake_settings):
        assert should_send_telegram_for_event("background.report", "info") is True


def test_recent_notifications_keeps_newest_first():
    from remy.core.notification_router import get_recent_notifications, notify

    notify("first", level="info", event_type="operator_alert")
    notify("second", level="warning", event_type="operator_alert")

    items = get_recent_notifications(event_type="operator_alert", limit=2)
    assert len(items) >= 2
    assert items[0]["message"] == "second"
    assert items[1]["message"] == "first"


def test_acknowledge_notification_marks_item():
    from remy.core.notification_router import acknowledge_notification, get_recent_notifications, notify

    notify("ack me", level="warning", event_type="operator_alert")
    item = get_recent_notifications(event_type="operator_alert", limit=1)[0]

    assert acknowledge_notification(item["id"]) is True

    updated = get_recent_notifications(event_type="operator_alert", limit=1)[0]
    assert updated["acknowledged"] is True


def test_operator_alerts_are_coalesced_when_same_incident_repeats():
    import importlib

    nr = importlib.import_module("remy.core.notification_router")
    with patch.object(nr, "NOTIFICATION_STORE_FILE", MagicMock()), \
         patch.object(nr, "_NOTIFICATIONS_LOADED", True):
        nr._RECENT_NOTIFICATIONS.clear()
        nr.notify(
            "Gateway degraded",
            level="warning",
            event_type="operator_alert",
            event_data={
                "gateway_health": "degraded",
                "health_level": "YELLOW",
                "action_target": "open_memory_verification",
                "artifact_ids": ["report-1"],
                "failure_code": "verification_failed",
                "verification_status": "repair_required",
                "requested": 1,
                "applied": 0,
                "skipped": 1,
            },
        )
        nr.notify(
            "Gateway degraded",
            level="warning",
            event_type="operator_alert",
            event_data={
                "gateway_health": "degraded",
                "health_level": "YELLOW",
                "action_target": "open_missing_memory_review",
                "artifact_ids": ["report-2"],
                "failure_code": "validation_error",
                "verification_status": "repair_required",
                "requested": 2,
                "applied": 1,
                "skipped": 1,
            },
        )

        items = nr.get_recent_notifications(event_type="operator_alert", limit=5)

    matches = [item for item in items if item["message"] == "Gateway degraded"]
    assert len(matches) == 1
    assert matches[0]["repeat_count"] == 2
    assert matches[0]["action_target"] == "open_missing_memory_review"
    assert matches[0]["artifact_ids"] == ["report-2"]
    assert matches[0]["failure_code"] == "validation_error"
    assert matches[0]["requested"] == 2
    assert matches[0]["applied"] == 1
    assert matches[0]["skipped"] == 1


def test_resolve_marks_matching_incident_and_keeps_recovery_event():
    from remy.core.notification_router import get_recent_notifications, notify

    notify(
        "Gateway degraded",
        level="warning",
        event_type="operator_alert",
        event_data={
            "dedupe_key": "gateway:degraded",
            "gateway_health": "degraded",
            "health_level": "YELLOW",
        },
    )
    notify(
        "Gateway recovered",
        level="info",
        event_type="operator_alert",
        event_data={
            "dedupe_key": "recovery:GREEN:ok",
            "resolved": True,
            "resolves": ["gateway:degraded"],
            "gateway_health": "ok",
            "health_level": "GREEN",
        },
    )

    items = get_recent_notifications(event_type="operator_alert", limit=5)
    recovered = next(item for item in items if item["message"] == "Gateway recovered")
    degraded = next(item for item in items if item["message"] == "Gateway degraded")
    assert recovered["resolved"] is True
    assert degraded["resolved"] is True
    assert degraded["dedupe_key"] == "gateway:degraded"


def test_notifications_persist_to_disk(tmp_path):
    nr = importlib.import_module("remy.core.notification_router")

    store = tmp_path / "operator_alerts.json"
    with patch.object(nr, "NOTIFICATION_STORE_FILE", store), \
         patch.object(nr, "_NOTIFICATIONS_LOADED", False):
        nr._RECENT_NOTIFICATIONS.clear()
        nr.notify("persist me", level="warning", event_type="operator_alert")
        nr._RECENT_NOTIFICATIONS.clear()
        nr._NOTIFICATIONS_LOADED = False

        items = nr.get_recent_notifications(event_type="operator_alert", limit=5)

    assert len(items) == 1
    assert items[0]["message"] == "persist me"
