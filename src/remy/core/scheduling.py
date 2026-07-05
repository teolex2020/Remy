"""Shared helpers for schedule/reminder normalization."""

from __future__ import annotations

from datetime import datetime


def _repeat_from_cron(cron: str) -> str:
    parts = [p.strip() for p in cron.split() if p.strip()]
    if len(parts) != 5:
        return ""
    minute, hour, day_of_month, month, day_of_week = parts
    if month == "*" and day_of_month == "*" and day_of_week == "*":
        return "daily"
    if month == "*" and day_of_month == "*" and day_of_week != "*":
        return "weekly"
    if month == "*" and day_of_month != "*" and day_of_week == "*":
        return "monthly"
    return ""


def normalize_schedule_args(args: dict) -> tuple[str, str, str | None]:
    due_date = (args.get("due_date") or "").strip()
    repeat = (args.get("repeat") or "").strip().lower()
    cron = (args.get("cron") or "").strip()

    if cron and not repeat:
        repeat = _repeat_from_cron(cron)
    if not due_date:
        due_date = datetime.now().strftime("%Y-%m-%d")

    return due_date, repeat, (cron or None)
