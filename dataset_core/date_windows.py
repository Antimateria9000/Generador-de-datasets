from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"})
DAILY_LIKE_INTERVALS = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MIDNIGHT_TEXT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]00:00:00(?:\.0+)?$")


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


def _parse_user_timestamp(raw: object, field_name: str) -> pd.Timestamp:
    value = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(value):
        raise DateWindowError(f"Invalid {field_name}: {raw!r}")
    return pd.Timestamp(value).tz_convert(None)


def _looks_like_calendar_input(raw: object) -> bool:
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return True

    text = str(raw or "").strip()
    if not text:
        return False
    return bool(_DATE_ONLY_RE.fullmatch(text) or _MIDNIGHT_TEXT_RE.fullmatch(text))


def normalize_user_start(
    start: object,
    interval: str,
) -> pd.Timestamp:
    start_ts = _parse_user_timestamp(start, "start")
    if is_daily_like_interval(interval) and _looks_like_calendar_input(start):
        return start_ts.normalize()
    return start_ts


def normalize_user_end_exclusive(
    end: object,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> pd.Timestamp:
    end_ts = _parse_user_timestamp(end, "end")
    if not _looks_like_calendar_input(end):
        return end_ts

    selected_day = end_ts.normalize()
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


def resolve_temporal_bounds(
    *,
    years: int | None,
    start: object | None,
    end: object | None,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> tuple[str, pd.Timestamp, pd.Timestamp, int | None, bool]:
    has_years = years is not None
    has_exact = bool(start or end)

    if has_years and has_exact:
        raise DateWindowError("Use either rolling years or an exact date range, not both.")
    if not has_years and not has_exact:
        raise DateWindowError("You must provide either --years or --start/--end.")
    if has_exact and (not start or not end):
        raise DateWindowError("Exact range mode requires both --start and --end.")

    normalized_interval = str(interval or "1d").strip().lower()
    now = _normalize_utc_naive(now_utc)

    if has_years:
        rolling_years = int(years)
        if rolling_years <= 0:
            raise DateWindowError("The number of years must be greater than zero.")

        if is_daily_like_interval(normalized_interval):
            current_day = now.normalize()
            range_start = current_day - pd.DateOffset(years=rolling_years)
            range_end = next_midnight_utc(now)
        else:
            range_end = now
            range_start = range_end - pd.DateOffset(years=rolling_years)

        return "rolling_years", range_start, range_end, rolling_years, False

    range_start = normalize_user_start(start, normalized_interval)
    range_end = normalize_user_end_exclusive(end, normalized_interval, now_utc=now)
    range_start, range_end, _ = validate_provider_window(
        interval=normalized_interval,
        start=range_start,
        end=range_end,
        now_utc=now,
    )
    if range_start is None or range_end is None or range_start >= range_end:
        raise DateWindowError("Invalid exact range: start must be earlier than end.")

    return "exact_dates", range_start, range_end, None, True


def build_ui_exact_end_exclusive(
    end_date: date,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> pd.Timestamp:
    return normalize_user_end_exclusive(
        end=end_date,
        interval=interval,
        now_utc=now_utc,
    )


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
