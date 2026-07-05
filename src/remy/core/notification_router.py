"""
Notification Router — presence-aware notification delivery.

Routes notifications to the right channel based on user presence:
- User active on web → event_bus only (web shows it)
- User NOT on web + Telegram configured → Telegram
- No Telegram → event_bus only

Centralizes the "where to send?" logic that was previously scattered
across autonomy.py, survival.py, and guidance_queue.py.
"""

import logging
import json
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger("NotificationRouter")

# How recently the user must have interacted with the web UI
# to be considered "active" (seconds).
WEB_PRESENCE_THRESHOLD_SEC = 300  # 5 minutes
_WEB_RUNTIME_ENABLED = False
_LEVEL_RANK = {"info": 1, "warning": 2, "critical": 3}
_RECENT_NOTIFICATIONS = deque(maxlen=50)
_NOTIFICATIONS_LOADED = False

try:
    from remy.config.settings import settings as _settings

    NOTIFICATION_STORE_FILE = _settings.DATA_DIR / "operator_alerts.json"
except Exception:
    NOTIFICATION_STORE_FILE = Path("data") / "operator_alerts.json"


def set_web_runtime_enabled(enabled: bool):
    """Mark whether the web runtime is currently available.

    This is used as a strong suppression signal for Telegram when the web UI is
    the primary active interface, even before a web session becomes active.
    """
    global _WEB_RUNTIME_ENABLED
    _WEB_RUNTIME_ENABLED = bool(enabled)


def is_web_runtime_enabled() -> bool:
    return _WEB_RUNTIME_ENABLED


# ============== PRESENCE DETECTION ==============


def is_user_active_on_web(threshold_sec: float = WEB_PRESENCE_THRESHOLD_SEC) -> bool:
    """Check if user is currently active in the web interface.

    Reads WebSession.last_activity from the web session manager.
    Returns False safely if web module is not loaded (headless/CLI mode).
    """
    try:
        # The session manager is instantiated by the web app — import the
        # singleton from the routes module where it's created.
        from remy.web.routes._helpers import _get_api
        from remy.web.session import WebSessionManager

        api = _get_api()
        mgr: WebSessionManager = api.get_session_manager()
        if mgr.session is None:
            return False
        age = time.time() - mgr.session.last_activity
        return age < threshold_sec
    except Exception:
        # Web module not loaded, or no session yet — not active
        return False


def is_web_runtime_available() -> bool:
    """True if the web runtime/session manager is loaded, even without recent activity."""
    if _WEB_RUNTIME_ENABLED:
        return True
    try:
        from remy.web.routes._helpers import _get_api

        api = _get_api()
        return api.get_session_manager() is not None
    except Exception:
        return False


def _telegram_configured() -> bool:
    """Check if Telegram bot token and chat ID are configured."""
    try:
        from remy.config.settings import settings

        return bool(settings.TELEGRAM_BOT_TOKEN and settings.PROACTIVE_CHAT_ID)
    except Exception:
        return False


def should_notify_telegram() -> bool:
    """True if Telegram is configured AND user is NOT active on web."""
    if not _telegram_configured():
        return False
    try:
        from remy.config.settings import settings

        if settings.TELEGRAM_SUPPRESS_WHEN_WEB_ENABLED and is_web_runtime_available():
            return False
    except Exception:
        pass
    return not is_user_active_on_web()


def should_send_telegram_for_event(event_type: str, level: str = "info") -> bool:
    """Apply per-event Telegram routing policy after presence checks pass."""
    if not should_notify_telegram():
        return False

    if event_type != "operator_alert":
        return True

    try:
        from remy.config.settings import settings

        min_level = (settings.TELEGRAM_OPERATOR_ALERT_MIN_LEVEL or "warning").lower()
    except Exception:
        min_level = "warning"

    current_rank = _LEVEL_RANK.get((level or "info").lower(), _LEVEL_RANK["info"])
    min_rank = _LEVEL_RANK.get(min_level, _LEVEL_RANK["warning"])
    return current_rank >= min_rank


def get_recent_notifications(*, event_type: str | None = None, limit: int = 10) -> list[dict]:
    """Return recent notifications, newest first."""
    _ensure_notifications_loaded()
    items = list(_RECENT_NOTIFICATIONS)
    if event_type:
        items = [item for item in items if item.get("type") == event_type]
    return list(reversed(items[-limit:]))


def acknowledge_notification(notification_id: str) -> bool:
    """Mark a recent notification as acknowledged."""
    _ensure_notifications_loaded()
    if not notification_id:
        return False
    for item in _RECENT_NOTIFICATIONS:
        if item.get("id") == notification_id:
            item["acknowledged"] = True
            _save_notifications()
            return True
    return False


def _resolve_recent_notifications(dedupe_keys: list[str] | None) -> int:
    """Mark matching recent notifications as resolved."""
    _ensure_notifications_loaded()
    if not dedupe_keys:
        return 0
    resolved = 0
    wanted = {str(key).strip() for key in dedupe_keys if str(key).strip()}
    if not wanted:
        return 0
    now = time.time()
    for item in _RECENT_NOTIFICATIONS:
        if item.get("dedupe_key") not in wanted:
            continue
        if item.get("resolved"):
            continue
        item["resolved"] = True
        item["resolved_at"] = now
        resolved += 1
    return resolved


def _operator_alert_dedupe_key(data: dict) -> str:
    """Build a stable coalescing key for operator alerts."""
    explicit = str(data.get("dedupe_key", "")).strip()
    if explicit:
        return explicit
    gateway_health = str(data.get("gateway_health", "")).strip().lower()
    health_level = str(data.get("health_level", "")).strip().upper()
    level = str(data.get("level", "info")).strip().lower()
    message = str(data.get("message", "")).strip()
    return "|".join([level, gateway_health, health_level, message])


def _store_recent_notification(data: dict) -> dict:
    """Insert or coalesce a recent notification entry."""
    _ensure_notifications_loaded()
    _resolve_recent_notifications(data.get("resolves"))

    if data.get("type") == "operator_alert":
        dedupe_key = _operator_alert_dedupe_key(data)
        for item in _RECENT_NOTIFICATIONS:
            if item.get("type") != "operator_alert":
                continue
            if item.get("dedupe_key") != dedupe_key:
                continue
            if item.get("acknowledged"):
                continue
            if item.get("resolved"):
                continue
            item["timestamp"] = data["timestamp"]
            item["message"] = data["message"]
            item["level"] = data["level"]
            item["gateway_health"] = data.get("gateway_health", item.get("gateway_health", ""))
            item["health_level"] = data.get("health_level", item.get("health_level", ""))
            item["source"] = data.get("source", item.get("source", ""))
            item["scenario_id"] = data.get("scenario_id", item.get("scenario_id", ""))
            item["action_target"] = data.get("action_target", item.get("action_target", ""))
            item["artifact_ids"] = list(data.get("artifact_ids") or item.get("artifact_ids", []) or [])
            item["failure_code"] = data.get("failure_code", item.get("failure_code", ""))
            item["verification_status"] = data.get("verification_status", item.get("verification_status", ""))
            item["verification_reason"] = data.get("verification_reason", item.get("verification_reason", ""))
            item["eval_status"] = data.get("eval_status", item.get("eval_status", ""))
            item["requested"] = data.get("requested", item.get("requested"))
            item["applied"] = data.get("applied", item.get("applied"))
            item["skipped"] = data.get("skipped", item.get("skipped"))
            item["repeat_count"] = int(item.get("repeat_count", 1)) + 1
            _RECENT_NOTIFICATIONS.remove(item)
            _RECENT_NOTIFICATIONS.append(item)
            _save_notifications()
            return dict(item)
        data["dedupe_key"] = dedupe_key
        data["repeat_count"] = int(data.get("repeat_count", 1) or 1)

    _RECENT_NOTIFICATIONS.append(dict(data))
    _save_notifications()
    return dict(data)


def _load_notifications() -> None:
    """Load persisted recent notifications from disk."""
    global _NOTIFICATIONS_LOADED
    _RECENT_NOTIFICATIONS.clear()
    try:
        if NOTIFICATION_STORE_FILE.exists():
            data = json.loads(NOTIFICATION_STORE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data[-_RECENT_NOTIFICATIONS.maxlen:]:
                    if isinstance(item, dict):
                        _RECENT_NOTIFICATIONS.append(item)
    except Exception as e:
        logger.debug("Could not load operator alerts: %s", e)
    _NOTIFICATIONS_LOADED = True


def _save_notifications() -> None:
    """Persist recent notifications to disk."""
    try:
        from remy.core.file_utils import atomic_write

        NOTIFICATION_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            NOTIFICATION_STORE_FILE,
            json.dumps(list(_RECENT_NOTIFICATIONS), ensure_ascii=False, indent=2) + "\n",
        )
    except Exception as e:
        logger.debug("Could not save operator alerts: %s", e)


def _ensure_notifications_loaded() -> None:
    global _NOTIFICATIONS_LOADED
    if not _NOTIFICATIONS_LOADED:
        _load_notifications()


# ============== SENDING ==============


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    """Send a message to Telegram. Returns True on success."""
    try:
        import httpx

        from remy.config.settings import settings

        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = httpx.post(
            url,
            json={
                "chat_id": settings.PROACTIVE_CHAT_ID,
                "text": message,
                "parse_mode": parse_mode,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.debug("Telegram send failed: %s", e)
        return False


def notify(
    message: str,
    *,
    level: str = "info",
    event_type: str = "notification",
    event_data: dict | None = None,
    parse_mode: str = "Markdown",
):
    """Route a notification to the appropriate channel.

    Args:
        message: The notification text.
        level: "info", "warning", "critical" — used for event metadata.
        event_type: Event bus event name (e.g., "autonomous.action").
        event_data: Extra data for the event bus emission.
        parse_mode: Telegram parse mode.
    """
    from remy.core.event_bus import event_bus
    from remy.core.runtime_event_contract import build_runtime_event

    # Always emit to event_bus — web will pick it up if open
    data = build_runtime_event(
        event_type,
        event_domain="operator" if event_type == "operator_alert" else "runtime",
        level=level,
        payload={"message": message},
        legacy_fields={
            "id": uuid4().hex[:12],
            "message": message,
            "acknowledged": False,
            "resolved": False,
            "resolved_at": None,
        },
    )
    if event_data:
        data.update(event_data)
        if isinstance(data.get("payload"), dict):
            data["payload"].update(event_data)
    stored = _store_recent_notification(data)
    event_bus.emit(event_type, stored)

    # Send to Telegram only if user is NOT on web
    if should_send_telegram_for_event(event_type, level):
        send_telegram(message, parse_mode=parse_mode)
