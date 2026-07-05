"""
Web Push Notifications — VAPID key management, subscription storage, push dispatch.

Single-user design: one subscription stored in memory + brain record for restart recovery.
"""

import json
import logging

from remy.config.settings import settings

logger = logging.getLogger("WebPush")

# In-memory subscription (single user)
_subscription: dict | None = None


def generate_vapid_keys() -> tuple[str, str]:
    """Generate VAPID key pair and save to runtime settings. Returns (public, private)."""
    from py_vapid import Vapid

    vapid = Vapid()
    vapid.generate_keys()

    public_key = vapid.public_key_urlsafe_base64()
    private_key = vapid.private_key_urlsafe_base64()

    from remy.config.settings import set_runtime_setting

    set_runtime_setting("VAPID_PUBLIC_KEY", public_key, target=settings)
    set_runtime_setting("VAPID_PRIVATE_KEY", private_key, target=settings)

    logger.info("Generated and saved VAPID keys to runtime settings")
    return public_key, private_key


def get_vapid_keys() -> tuple[str, str]:
    """Get VAPID keys, auto-generating if not configured."""
    if settings.VAPID_PUBLIC_KEY and settings.VAPID_PRIVATE_KEY:
        return settings.VAPID_PUBLIC_KEY, settings.VAPID_PRIVATE_KEY
    return generate_vapid_keys()


def save_subscription(sub_dict: dict):
    """Store push subscription in memory and brain."""
    global _subscription
    _subscription = sub_dict

    try:
        from remy.core.agent_tools import brain, brain_lock
        from remy.core.agent_tools import Level

        # Remove old subscription record if exists
        with brain_lock:
            existing = brain.search(query="", tags=["push-subscription"], limit=1)
            for r in existing:
                brain.delete(r.id)

            brain.store(
                content=json.dumps(sub_dict),
                level=Level.DOMAIN,
                tags=["push-subscription", "system"],
                deduplicate=False,
            )
        logger.info("Push subscription saved")
    except Exception as e:
        logger.warning(f"Failed to persist subscription to brain: {e}")


def load_subscription():
    """Restore subscription from brain on startup."""
    global _subscription
    try:
        from remy.core.agent_tools import brain, brain_is_initialized, brain_lock

        if not brain_is_initialized():
            logger.info("Push subscription restore deferred until brain is first used")
            return

        with brain_lock:
            records = brain.search(query="", tags=["push-subscription"], limit=1)
        if records:
            _subscription = json.loads(records[0].content)
            logger.info("Push subscription restored from brain")
        else:
            logger.info("No push subscription found in brain")
    except Exception as e:
        logger.warning(f"Failed to load subscription from brain: {e}")


def remove_subscription():
    """Clear subscription from memory and brain."""
    global _subscription
    _subscription = None

    try:
        from remy.core.agent_tools import brain, brain_lock

        with brain_lock:
            existing = brain.search(query="", tags=["push-subscription"], limit=1)
            for r in existing:
                brain.delete(r.id)
        logger.info("Push subscription removed")
    except Exception as e:
        logger.warning(f"Failed to remove subscription from brain: {e}")


def get_subscription() -> dict | None:
    """Get current subscription."""
    return _subscription


def send_web_push(title: str, body: str, url: str = "/") -> bool:
    """Send a push notification. Returns True on success."""
    if not _subscription:
        return False

    try:
        from pywebpush import webpush, WebPushException

        public_key, private_key = get_vapid_keys()

        payload = json.dumps({
            "title": title,
            "body": body,
            "url": url,
        })

        webpush(
            subscription_info=_subscription,
            data=payload,
            vapid_private_key=private_key,
            vapid_claims={"sub": settings.VAPID_CLAIM_EMAIL},
        )
        logger.info(f"Push notification sent: {title}")
        return True

    except Exception as e:
        error_str = str(e)
        # 410 Gone or 404 = subscription expired
        if "410" in error_str or "404" in error_str:
            logger.warning("Push subscription expired, removing")
            remove_subscription()
        else:
            logger.error(f"Push notification failed: {e}")
        return False
