"""Deterministic safety preflight for local workflows."""

from __future__ import annotations

from typing import Any


def assess_workflow_safety(
    *,
    kind: str,
    steps: list[dict[str, Any]] | None,
    trigger: dict[str, Any] | None = None,
    output_destination: dict[str, Any] | None = None,
    has_successful_run: bool = False,
    mode: str = "manual_run",
) -> dict[str, Any]:
    """Return blockers and warnings before a workflow run.

    The manual run path is intentionally warning-first: users need to run once
    to establish a baseline. Scheduled/external execution can be stricter.
    """
    steps = list(steps or [])
    trigger = trigger or {}
    output_destination = output_destination or {}
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    is_scheduled = kind == "automation" and trigger.get("type") == "schedule"
    is_scheduled_execution = mode == "scheduled_execution"

    if not steps:
        blockers.append(_issue("empty_workflow", "Workflow has no action blocks."))

    for index, step in enumerate(steps, start=1):
        step_type = step.get("type", "")
        label = step.get("label") or step_type or f"Step {index}"
        config = step.get("config") or {}

        if step_type == "memory_save" and not (config.get("dedup_guard") or config.get("deduplicate")):
            warnings.append(_issue(
                "memory_save_without_dedup",
                f"{label} saves memory without a dedup guard.",
                "Enable Skip duplicate saves before repeated or scheduled runs.",
            ))

        if step_type == "page_scrape":
            max_chars = _safe_int(config.get("max_chars"), 0)
            if max_chars <= 0:
                blockers.append(_issue(
                    "page_scrape_without_limit",
                    f"{label} has no max character limit.",
                    "Set Max characters so scraped pages cannot flood the workflow.",
                ))
            elif max_chars > 50000:
                warnings.append(_issue(
                    "page_scrape_large_limit",
                    f"{label} can pass up to {max_chars} characters.",
                    "Keep scraper limits tight for predictable cost and prompt size.",
                ))

        if step_type == "http_request":
            warnings.append(_issue(
                "external_http_request",
                f"{label} calls an external HTTP endpoint.",
                "Review the URL and payload before enabling schedules.",
            ))
            auth_secret_key = str(config.get("auth_secret_key", "") or "").strip()
            if auth_secret_key and not _local_secret_configured(auth_secret_key):
                blockers.append(_issue(
                    "missing_http_auth_secret",
                    f"{label} uses an Authorization secret that is not configured.",
                    "Open Settings -> Local Secrets and save the selected secret before running this workflow.",
                ))
            if is_scheduled_execution and not config.get("require_approval"):
                blockers.append(_issue(
                    "scheduled_http_without_approval",
                    f"{label} is scheduled without explicit approval.",
                    "Run manually first or require approval before scheduled HTTP calls.",
                ))

    output_type = output_destination.get("type", "chat")
    if output_type in {"telegram", "email", "webhook"}:
        warnings.append(_issue(
            "external_output_destination",
            f"Output sends data to {output_type}.",
            "Confirm the destination before running with private documents or memory.",
        ))
        if output_type == "telegram" and not _local_secret_configured("telegram_bot_token"):
            blockers.append(_issue(
                "missing_telegram_secret",
                "Telegram output is selected but the Telegram bot token is not configured.",
                "Open Settings -> Local Secrets and save the Telegram bot token before running this automation.",
            ))
        if output_type == "email":
            if not _settings_value_configured("SMTP_USER"):
                blockers.append(_issue(
                    "missing_email_account",
                    "Email output is selected but the sender email account is not configured.",
                    "Open Settings -> Integrations and save the Gmail address for email delivery.",
                ))
            if not _local_secret_configured("smtp_password"):
                blockers.append(_issue(
                    "missing_email_secret",
                    "Email output is selected but the email app password is not configured.",
                    "Open Settings -> Local Secrets and save the Email app password before running this automation.",
                ))
        if is_scheduled_execution and not output_destination.get("require_approval"):
            blockers.append(_issue(
                "scheduled_external_output_without_approval",
                f"Scheduled output to {output_type} requires approval.",
                "Use Chat/Memory for unattended local runs or require approval.",
            ))

    if is_scheduled and not has_successful_run:
        issue = _issue(
            "scheduled_without_baseline",
            "This scheduled automation has no successful dry-run baseline.",
            "Run it manually once and inspect History before relying on the schedule.",
        )
        if is_scheduled_execution:
            blockers.append(issue)
        else:
            warnings.append(issue)

    return {
        "ok": not blockers,
        "kind": kind,
        "mode": mode,
        "blockers": blockers,
        "warnings": warnings,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
    }


def _issue(code: str, message: str, detail: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "detail": detail}


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _local_secret_configured(secret_key: str) -> bool:
    secret_map = {
        "gemini_api_key": "GEMINI_API_KEY",
        "openrouter_api_key": "OPENROUTER_API_KEY",
        "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
        "smtp_password": "SMTP_PASSWORD",
    }
    setting_name = secret_map.get((secret_key or "").strip())
    if not setting_name:
        return False
    try:
        from remy.config.settings import settings

        return bool(getattr(settings, setting_name, None))
    except Exception:
        return False


def _settings_value_configured(setting_name: str) -> bool:
    try:
        from remy.config.settings import settings

        return bool(getattr(settings, setting_name, None))
    except Exception:
        return False
