from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

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
