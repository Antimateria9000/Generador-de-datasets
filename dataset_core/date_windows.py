from __future__ import annotations

from datetime import date

import pandas as pd

INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"})
DAILY_LIKE_INTERVALS = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})


class DateWindowError(ValueError):
    """Raised when a temporal window cannot be normalized safely."""


def _normalize_utc_naive(ts: pd.Timestamp | None = None) -> pd.Timestamp:
    value = pd.Timestamp.utcnow() if ts is None else pd.Timestamp(ts)
    if value.tzinfo is not None:
        value = value.tz_convert("UTC").tz_localize(None)
    return value


def next_midnight_utc(now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    return _normalize_utc_naive(now_utc).normalize() + pd.Timedelta(days=1)


def is_intraday_interval(interval: str) -> bool:
    return str(interval or "").strip().lower() in INTRADAY_INTERVALS


def is_daily_like_interval(interval: str) -> bool:
    return str(interval or "").strip().lower() in DAILY_LIKE_INTERVALS


def build_ui_exact_end_exclusive(
    end_date: date,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> pd.Timestamp:
    selected_day = pd.Timestamp(end_date).normalize()
    exclusive_end = selected_day + pd.Timedelta(days=1)
    now = _normalize_utc_naive(now_utc)

    if is_intraday_interval(interval):
        if selected_day > now.normalize():
            raise DateWindowError(
                "Intraday exact ranges cannot use an inclusive end date later than the current UTC day."
            )
        return min(exclusive_end, now)

    _, validated_end, _ = validate_provider_window(
        interval=interval,
        start=None,
        end=exclusive_end,
        now_utc=now,
    )
    if validated_end is None:
        raise DateWindowError("A valid exclusive end timestamp could not be derived from the selected date.")
    return validated_end


def validate_provider_window(
    interval: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    now_utc: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, list[str]]:
    warnings: list[str] = []
    if end is None:
        return start, end, warnings

    normalized_interval = str(interval or "").strip().lower()
    now = _normalize_utc_naive(now_utc)

    if is_intraday_interval(normalized_interval):
        if end > now:
            raise DateWindowError(
                "Intraday requests cannot use a future end timestamp. "
                f"Requested end={end.isoformat()} current_utc={now.isoformat()}."
            )
        return start, end, warnings

    allowed_daily_end = next_midnight_utc(now)
    if is_daily_like_interval(normalized_interval) and end > allowed_daily_end:
        raise DateWindowError(
            "Daily-like requests allow an exclusive end only up to the next UTC midnight. "
            f"Requested end={end.isoformat()} allowed_end={allowed_daily_end.isoformat()}."
        )

    if not is_daily_like_interval(normalized_interval) and end > now:
        raise DateWindowError(
            "Unsupported future end timestamp for the requested interval. "
            f"Requested end={end.isoformat()} current_utc={now.isoformat()}."
        )

    return start, end, warnings
