from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dataset_core.contracts import TemporalRange
from dataset_core.date_windows import build_ui_exact_end_exclusive
from providers.yfinance_provider import RequestValidationError, YFinanceProvider


def test_provider_daily_like_allows_next_midnight_for_today_exact_end():
    provider = YFinanceProvider()
    now_utc = pd.Timestamp("2026-03-29T15:30:00")
    end = build_ui_exact_end_exclusive(
        end_date=date(2026, 3, 29),
        interval="1d",
        now_utc=now_utc,
    )

    effective_start, effective_end, warnings = provider._prepare_request_window(
        interval="1d",
        start=pd.Timestamp("2026-03-20T00:00:00"),
        end=end,
        now_utc=now_utc,
    )

    assert effective_start == pd.Timestamp("2026-03-20T00:00:00")
    assert effective_end == pd.Timestamp("2026-03-30T00:00:00")
    assert warnings == []


def test_provider_intraday_rejects_future_end():
    provider = YFinanceProvider()
    now_utc = pd.Timestamp("2026-03-29T15:30:00")

    with pytest.raises(RequestValidationError, match="Intraday requests cannot use a future end timestamp"):
        provider._prepare_request_window(
            interval="1m",
            start=pd.Timestamp("2026-03-29T14:00:00"),
            end=pd.Timestamp("2026-03-29T16:00:00"),
            now_utc=now_utc,
        )


def test_ui_intraday_exact_end_caps_today_at_current_utc():
    now_utc = pd.Timestamp("2026-03-29T15:30:00")

    effective_end = build_ui_exact_end_exclusive(
        end_date=date(2026, 3, 29),
        interval="1m",
        now_utc=now_utc,
    )

    assert effective_end == now_utc


def test_daily_like_exact_range_matches_cli_and_ui_semantics():
    now_utc = pd.Timestamp("2026-03-29T15:30:00")

    cli_range = TemporalRange.from_inputs(
        years=None,
        start="2021-03-29",
        end="2026-03-29",
        interval="1d",
        now_utc=now_utc,
    )
    ui_range = TemporalRange.from_inputs(
        years=None,
        start=date(2021, 3, 29),
        end=date(2026, 3, 29),
        interval="1d",
        now_utc=now_utc,
    )

    assert cli_range.start == ui_range.start == pd.Timestamp("2021-03-29T00:00:00")
    assert cli_range.end == ui_range.end == pd.Timestamp("2026-03-30T00:00:00")


def test_daily_like_rolling_years_uses_same_inclusive_calendar_end_as_exact_ranges():
    now_utc = pd.Timestamp("2026-03-29T15:30:00")

    rolling_range = TemporalRange.from_inputs(
        years=5,
        start=None,
        end=None,
        interval="1d",
        now_utc=now_utc,
    )

    assert rolling_range.start == pd.Timestamp("2021-03-29T00:00:00")
    assert rolling_range.end == pd.Timestamp("2026-03-30T00:00:00")
