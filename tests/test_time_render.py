"""Tests for time_render — boundary conditions and input-type coverage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from remy.core.time_render import format_age, format_age_labeled


NOW = datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc)


def _ago(seconds: float = 0, minutes: float = 0, hours: float = 0, days: float = 0) -> datetime:
    return NOW - timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days)


class TestFormatAgeBoundaries:
    def test_just_now_under_2min(self):
        assert format_age(_ago(seconds=10), now=NOW) == "just now"
        assert format_age(_ago(seconds=119), now=NOW) == "just now"

    def test_minutes_boundary(self):
        # 2 minutes exactly — crosses into "Nm ago".
        assert format_age(_ago(minutes=2), now=NOW) == "2m ago"
        assert format_age(_ago(minutes=59), now=NOW) == "59m ago"

    def test_hours_boundary(self):
        assert format_age(_ago(hours=1), now=NOW) == "1h ago"
        assert format_age(_ago(hours=23), now=NOW) == "23h ago"

    def test_yesterday_exact(self):
        assert format_age(_ago(days=1), now=NOW) == "yesterday"

    def test_days_range(self):
        assert format_age(_ago(days=2), now=NOW) == "2d ago"
        assert format_age(_ago(days=13), now=NOW) == "13d ago"

    def test_14_days_switches_to_iso_date(self):
        # At exactly 14 days we leave relative form.
        result = format_age(_ago(days=14), now=NOW)
        assert result == "2026-03-31"

    def test_under_year_gives_iso_date(self):
        result = format_age(_ago(days=100), now=NOW)
        assert result == "2026-01-04"

    def test_364_days_still_iso_date(self):
        # Last day inside the "YYYY-MM-DD" window.
        result = format_age(_ago(days=364), now=NOW)
        assert result.count("-") == 2
        assert len(result) == 10

    def test_year_or_more_gives_year_month(self):
        result = format_age(_ago(days=365), now=NOW)
        assert result == "2025-04"

    def test_multi_year(self):
        result = format_age(_ago(days=800), now=NOW)
        assert result.startswith("2024-")
        assert len(result) == 7


class TestFormatAgeInputTypes:
    def test_float_epoch(self):
        past = _ago(hours=3)
        assert format_age(past.timestamp(), now=NOW) == "3h ago"

    def test_int_epoch(self):
        past = _ago(hours=5)
        assert format_age(int(past.timestamp()), now=NOW) == "5h ago"

    def test_iso_string_with_tz(self):
        past = _ago(days=3)
        assert format_age(past.isoformat(), now=NOW) == "3d ago"

    def test_iso_string_with_z(self):
        past = _ago(hours=2).isoformat().replace("+00:00", "Z")
        assert format_age(past, now=NOW) == "2h ago"

    def test_naive_datetime_assumed_utc(self):
        past = _ago(hours=4).replace(tzinfo=None)
        assert format_age(past, now=NOW) == "4h ago"

    def test_naive_iso_string_assumed_utc(self):
        past = _ago(hours=6).replace(tzinfo=None).isoformat()
        assert format_age(past, now=NOW) == "6h ago"


class TestFormatAgeEdgeCases:
    def test_none_returns_empty(self):
        assert format_age(None, now=NOW) == ""

    def test_empty_string_returns_empty(self):
        assert format_age("", now=NOW) == ""

    def test_unparseable_string_returns_empty(self):
        assert format_age("not a date", now=NOW) == ""

    def test_future_timestamp_returns_empty(self):
        future = NOW + timedelta(hours=1)
        assert format_age(future, now=NOW) == ""

    def test_now_parameter_accepts_float(self):
        past = _ago(minutes=30)
        assert format_age(past, now=NOW.timestamp()) == "30m ago"

    def test_now_parameter_accepts_naive_datetime(self):
        past = _ago(minutes=45)
        assert format_age(past, now=NOW.replace(tzinfo=None)) == "45m ago"


class TestFormatAgeLabeled:
    def test_default_record_kind(self):
        assert format_age_labeled(_ago(hours=2), now=NOW) == "stored 2h ago"

    def test_message_kind(self):
        assert format_age_labeled(_ago(days=3), kind="message", now=NOW) == "said 3d ago"

    def test_update_kind(self):
        assert format_age_labeled(_ago(days=1), kind="update", now=NOW) == "updated yesterday"

    def test_unknown_kind_falls_back_to_default(self):
        assert format_age_labeled(_ago(hours=1), kind="weird", now=NOW) == "stored 1h ago"

    def test_empty_when_age_empty(self):
        assert format_age_labeled(None, now=NOW) == ""
        assert format_age_labeled("bogus", now=NOW) == ""
