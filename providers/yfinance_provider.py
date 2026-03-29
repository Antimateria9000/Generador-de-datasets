from __future__ import annotations

"""
Institutional-grade Yahoo Finance provider for OHLCV dataset generation.

Design principles:
- No silent degradation of semantic intent.
- Deterministic, validated request handling.
- Formal provenance attached to every returned DataFrame.
- Explicit intraday-window policy and chunked retrieval.
- Backward-compatible public method `get_history()`.
"""

import asyncio
import logging
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
import yfinance as yf

from dataset_core.date_windows import DateWindowError, is_daily_like_interval, is_intraday_interval, validate_provider_window
from dataset_core.settings import SUPPORTED_INTERVALS

PROVIDER_NAME = "AB3.YFinanceProvider"
PROVIDER_VERSION = "2.1.0"

logger = logging.getLogger(PROVIDER_NAME)
logger.addHandler(logging.NullHandler())

EXPORT_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
]

ALLOWED_INTERVALS = set(SUPPORTED_INTERVALS)

INTERVAL_ALIASES = {
    "d": "1d",
    "1day": "1d",
    "day": "1d",
    "daily": "1d",
    "w": "1wk",
    "1w": "1wk",
    "week": "1wk",
    "weekly": "1wk",
    "m": "1mo",
    "1mth": "1mo",
    "month": "1mo",
    "monthly": "1mo",
    "h": "1h",
    "60m": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc
    return max(value, minimum)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw!r}") from exc
    return max(value, minimum)


DEFAULT_MAX_WORKERS = _env_int("AB3_YF_MAX_WORKERS", 4, minimum=1)
DEFAULT_RETRIES = _env_int("AB3_YF_RETRIES", 4, minimum=1)
DEFAULT_TIMEOUT = _env_float("AB3_YF_TIMEOUT", 10.0, minimum=0.1)
DEFAULT_MIN_DELAY = _env_float("AB3_YF_MIN_DELAY", 0.25, minimum=0.0)
DEFAULT_MAX_INTRADAY_LOOKBACK_DAYS = _env_int("AB3_YF_MAX_INTRADAY_LOOKBACK_DAYS", 60, minimum=1)


class YFinanceProviderError(Exception):
    """Base error for provider failures."""


class ProviderConfigurationError(YFinanceProviderError):
    """Raised when provider configuration is invalid."""


class RequestValidationError(YFinanceProviderError):
    """Raised when a request is semantically invalid."""


class EmptyDatasetError(YFinanceProviderError):
    """Raised when the upstream request returns no usable rows."""


class DownloadFailureError(YFinanceProviderError):
    """Raised when all download strategies fail."""


@dataclass
class DownloadAttempt:
    attempt_number: int
    backend: str
    interval: str
    start: Optional[str]
    end: Optional[str]
    duration_seconds: float
    success: bool
    rows: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FetchMetadata:
    provider_name: str
    provider_version: str
    source: str
    request_id: str
    requested_symbol: str
    resolved_symbol: str
    requested_interval: str
    resolved_interval: str
    requested_start: Optional[str]
    requested_end: Optional[str]
    effective_start: Optional[str]
    effective_end: Optional[str]
    actual_start: Optional[str]
    actual_end: Optional[str]
    extracted_at_utc: str
    auto_adjust: bool
    actions: bool
    backend_used: Optional[str] = None
    chunked: bool = False
    chunk_count: int = 0
    row_count: int = 0
    warnings: List[str] = field(default_factory=list)
    attempts: List[DownloadAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return payload


@dataclass
class FetchResult:
    symbol: str
    data: pd.DataFrame
    metadata: FetchMetadata


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ts_to_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    return ts.isoformat()


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise RequestValidationError("Ticker symbol cannot be empty.")
    return value


def _normalize_interval(interval: Optional[str]) -> Tuple[str, str]:
    raw = "1d" if interval is None else str(interval).strip()
    if not raw:
        raw = "1d"

    lowered = raw.lower().replace(" ", "")
    normalized = INTERVAL_ALIASES.get(lowered, lowered)

    if normalized not in ALLOWED_INTERVALS:
        valid = ", ".join(sorted(ALLOWED_INTERVALS))
        raise RequestValidationError(
            f"Unsupported interval {raw!r}. Allowed intervals are: {valid}."
        )

    return raw, normalized


def _to_naive_utc(ts: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if ts is None or ts == "":
        return None
    out = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(out):
        raise RequestValidationError(f"Invalid timestamp: {ts!r}")
    return out.tz_convert(None)


def _validate_date_range(
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> None:
    if start is not None and end is not None and start >= end:
        raise RequestValidationError(
            f"Invalid date range: start ({start.isoformat()}) must be earlier than end ({end.isoformat()})."
        )


def _intraday_chunk_timedelta(max_lookback_days: int) -> pd.Timedelta:
    safe_days = max(1, max_lookback_days - 1)
    return pd.Timedelta(days=safe_days)


def _build_intraday_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_lookback_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if start >= end:
        return []

    delta = _intraday_chunk_timedelta(max_lookback_days)
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end

    return chunks


def _ensure_datetime_index(
    df: pd.DataFrame,
    preserve_local_dates: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out[~out.index.isna()]

    if getattr(out.index, "tz", None) is not None:
        if preserve_local_dates:
            out.index = out.index.tz_localize(None)
        else:
            out.index = out.index.tz_convert(None)

    out = out.sort_index()

    if getattr(out.index, "has_duplicates", False):
        out = out[~out.index.duplicated(keep="last")]

    return out


def _flatten_columns_if_needed(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if hasattr(out, "columns") and getattr(out.columns, "nlevels", 1) > 1:
        try:
            level0 = out.columns.get_level_values(0)
            if symbol in level0:
                out = out[symbol]
            else:
                out.columns = out.columns.get_level_values(-1)
        except Exception:
            out.columns = out.columns.get_level_values(-1)

    return out


def _clean_df(df: pd.DataFrame, preserve_local_dates: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = _ensure_datetime_index(df, preserve_local_dates=preserve_local_dates)
    if out.empty:
        return out

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    required_ohlc = [column for column in ["Open", "High", "Low", "Close"] if column in out.columns]
    if required_ohlc:
        out = out.dropna(subset=required_ohlc)

    return out


def _calendarize_daily(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if not is_daily_like_interval(interval):
        return _ensure_datetime_index(df, preserve_local_dates=False)

    out = _ensure_datetime_index(df, preserve_local_dates=True)
    if out.empty:
        return out

    out.index = out.index.normalize()

    if getattr(out.index, "has_duplicates", False):
        aggregation = {}
        for column in out.columns:
            lowered = column.lower()
            if lowered == "open":
                aggregation[column] = "first"
            elif lowered == "high":
                aggregation[column] = "max"
            elif lowered == "low":
                aggregation[column] = "min"
            elif lowered in {"close", "adj close"}:
                aggregation[column] = "last"
            elif lowered == "volume":
                aggregation[column] = "sum"
            elif lowered in {"dividends", "stock splits"}:
                aggregation[column] = "sum"
            else:
                aggregation[column] = "last"

        out = out.groupby(out.index).agg(aggregation).sort_index()
        out = out[~out.index.duplicated(keep="last")]

    return out


def _select_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    out = df.copy()

    if "Adj Close" not in out.columns and "Close" in out.columns:
        out["Adj Close"] = out["Close"]
    if "Dividends" not in out.columns:
        out["Dividends"] = 0.0
    if "Stock Splits" not in out.columns:
        out["Stock Splits"] = 0.0
    if "Volume" not in out.columns:
        out["Volume"] = 0

    out = out.reset_index()

    date_column = "Date" if "Date" in out.columns else out.columns[0]

    rename_map = {
        date_column: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    out = out.rename(columns=rename_map)

    for column in EXPORT_COLUMNS:
        if column not in out.columns:
            if column in {"dividends", "stock_splits"}:
                out[column] = 0.0
            elif column == "volume":
                out[column] = 0
            else:
                out[column] = pd.NA

    out = out[EXPORT_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0)
    out["dividends"] = out["dividends"].fillna(0.0)
    out["stock_splits"] = out["stock_splits"].fillna(0.0)

    out = out.sort_values("date").reset_index(drop=True)
    return out


def _empty_export_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPORT_COLUMNS)


def _attach_metadata(df: pd.DataFrame, metadata: FetchMetadata) -> pd.DataFrame:
    out = df.copy()
    out.attrs["ab3_provenance"] = metadata.to_dict()
    out.attrs["ab3_provider"] = metadata.provider_name
    out.attrs["ab3_provider_version"] = metadata.provider_version
    return out


def _extract_actual_bounds(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df is None or df.empty or "date" not in df.columns:
        return None, None

    values = pd.to_datetime(df["date"], errors="coerce").dropna()
    if values.empty:
        return None, None

    return values.min(), values.max()


class YFinanceProvider:
    """
    Yahoo Finance OHLCV provider with institutional-grade validation, provenance,
    and explicit intraday handling.

    Public API:
    - get_history(): backward-compatible DataFrame / dict[str, DataFrame]
    - get_history_bundle(): structured FetchResult / dict[str, FetchResult]
    - get_history_async(): async wrapper around get_history()
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_MAX_WORKERS,
        retries: int = DEFAULT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        min_delay: float = DEFAULT_MIN_DELAY,
        max_intraday_lookback_days: int = DEFAULT_MAX_INTRADAY_LOOKBACK_DAYS,
        allow_partial_intraday: bool = False,
    ) -> None:
        if isinstance(max_workers, dict):
            params = dict(max_workers)
            max_workers = params.get("max_workers", DEFAULT_MAX_WORKERS)
            retries = params.get("retries", DEFAULT_RETRIES)
            timeout = params.get("timeout", DEFAULT_TIMEOUT)
            min_delay = params.get("min_delay", DEFAULT_MIN_DELAY)
            max_intraday_lookback_days = params.get(
                "max_intraday_lookback_days",
                DEFAULT_MAX_INTRADAY_LOOKBACK_DAYS,
            )
            allow_partial_intraday = params.get("allow_partial_intraday", False)

        self.max_workers = int(max_workers)
        self.retries = int(retries)
        self.timeout = float(timeout)
        self.min_delay = float(min_delay)
        self.max_intraday_lookback_days = int(max_intraday_lookback_days)
        self.allow_partial_intraday = bool(allow_partial_intraday)

        if self.max_workers < 1:
            raise ProviderConfigurationError("max_workers must be >= 1.")
        if self.retries < 1:
            raise ProviderConfigurationError("retries must be >= 1.")
        if self.timeout <= 0:
            raise ProviderConfigurationError("timeout must be > 0.")
        if self.min_delay < 0:
            raise ProviderConfigurationError("min_delay must be >= 0.")
        if self.max_intraday_lookback_days < 1:
            raise ProviderConfigurationError("max_intraday_lookback_days must be >= 1.")

        self._configure_project_cache()

    @staticmethod
    def _configure_project_cache() -> None:
        project_root = Path(__file__).resolve().parents[1]
        cache_dir = project_root / "workspace" / "cache" / "yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            probe_path = cache_dir / "probe.db"
            connection = sqlite3.connect(probe_path)
            connection.execute("create table if not exists probe(x int)")
            connection.commit()
            connection.close()
            probe_path.unlink(missing_ok=True)
            yf.set_tz_cache_location(str(cache_dir))
        except Exception as exc:
            logger.warning(
                "yfinance sqlite cache is not available at %s; disabling yfinance caches. Reason: %s",
                cache_dir,
                exc,
            )
            try:
                import yfinance.cache as yf_cache

                yf_cache._TzCacheManager._tz_cache = yf_cache._TzCacheDummy()
                yf_cache._CookieCacheManager._Cookie_cache = yf_cache._CookieCacheDummy()
                yf_cache._ISINCacheManager._isin_cache = yf_cache._ISINCacheDummy()
            except Exception as cache_exc:
                logger.warning("Failed to disable yfinance caches cleanly: %s", cache_exc)

    def _download_via_ticker(
        self,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        try:
            return ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
                timeout=self.timeout,
            )
        except TypeError:
            return ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            )

    def _download_via_download(
        self,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
                progress=False,
                threads=False,
                keepna=True,
            )
        except TypeError:
            raw = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
                progress=False,
                threads=False,
            )
        return _flatten_columns_if_needed(raw, symbol)

    def _prepare_request_window(
        self,
        interval: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        now_utc: Optional[pd.Timestamp] = None,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], List[str]]:
        warnings: List[str] = []
        now = pd.Timestamp.now(tz="UTC").tz_convert(None) if now_utc is None else pd.Timestamp(now_utc)
        if getattr(now, "tzinfo", None) is not None:
            now = now.tz_convert("UTC").tz_localize(None)

        try:
            start, end, validation_warnings = validate_provider_window(
                interval=interval,
                start=start,
                end=end,
                now_utc=now,
            )
        except DateWindowError as exc:
            raise RequestValidationError(str(exc)) from exc
        warnings.extend(validation_warnings)

        if not is_intraday_interval(interval):
            _validate_date_range(start, end)
            return start, end, warnings

        effective_end = end or now
        effective_start = start

        if effective_start is None:
            effective_start = effective_end - pd.Timedelta(days=self.max_intraday_lookback_days - 1)
            warnings.append(
                "Intraday request omitted start; effective_start was inferred from the configured intraday lookback window."
            )

        _validate_date_range(effective_start, effective_end)

        oldest_allowed = effective_end - pd.Timedelta(days=self.max_intraday_lookback_days)
        if effective_start < oldest_allowed:
            if not self.allow_partial_intraday:
                raise RequestValidationError(
                    "Requested intraday history exceeds the configured maximum lookback window. "
                    "Set allow_partial_intraday=True only if you explicitly want truncation."
                )
            warnings.append(
                "Intraday request exceeded the configured lookback window and was truncated to the oldest allowed timestamp."
            )
            effective_start = oldest_allowed

        _validate_date_range(effective_start, effective_end)
        return effective_start, effective_end, warnings

    def _normalize_raw_history(
        self,
        raw: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        preserve_local_dates = is_daily_like_interval(interval)

        frame = _flatten_columns_if_needed(raw, symbol)
        frame = _clean_df(frame, preserve_local_dates=preserve_local_dates)
        frame = _calendarize_daily(frame, interval)
        frame = _select_export_columns(frame)

        if frame.empty:
            return frame

        if frame["date"].duplicated().any():
            frame = frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        frame = frame.sort_values("date").reset_index(drop=True)
        return frame

    def _merge_chunk_frames(
        self,
        frames: List[pd.DataFrame],
        interval: str,
    ) -> pd.DataFrame:
        if not frames:
            return _empty_export_frame()

        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.dropna(subset=["date"])

        if is_daily_like_interval(interval):
            merged["date"] = merged["date"].dt.normalize()
            merged = (
                merged.sort_values("date")
                .groupby("date", as_index=False)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "adj_close": "last",
                        "volume": "sum",
                        "dividends": "sum",
                        "stock_splits": "sum",
                    }
                )
            )
        else:
            merged = (
                merged.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )

        return merged[EXPORT_COLUMNS].copy()

    def _download_window(
        self,
        backend: str,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        if backend == "ticker":
            raw = self._download_via_ticker(symbol, start, end, interval, auto_adjust, actions)
        elif backend == "download":
            raw = self._download_via_download(symbol, start, end, interval, auto_adjust, actions)
        else:
            raise ProviderConfigurationError(f"Unsupported backend: {backend!r}")

        return self._normalize_raw_history(raw, symbol, interval)

    def _download_range(
        self,
        backend: str,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> Tuple[pd.DataFrame, bool, int]:
        if is_intraday_interval(interval) and start is not None and end is not None:
            chunks = _build_intraday_chunks(start, end, self.max_intraday_lookback_days)
        else:
            chunks = [(start, end)]

        frames: List[pd.DataFrame] = []
        for chunk_start, chunk_end in chunks:
            frame = self._download_window(
                backend=backend,
                symbol=symbol,
                start=chunk_start,
                end=chunk_end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            )
            if not frame.empty:
                frames.append(frame)

        merged = self._merge_chunk_frames(frames, interval)
        return merged, len(chunks) > 1, len(chunks)

    def _build_metadata(
        self,
        requested_symbol: str,
        resolved_symbol: str,
        requested_interval: str,
        resolved_interval: str,
        requested_start: Optional[pd.Timestamp],
        requested_end: Optional[pd.Timestamp],
        effective_start: Optional[pd.Timestamp],
        effective_end: Optional[pd.Timestamp],
        auto_adjust: bool,
        actions: bool,
        warnings: List[str],
    ) -> FetchMetadata:
        return FetchMetadata(
            provider_name=PROVIDER_NAME,
            provider_version=PROVIDER_VERSION,
            source="yahoo_finance_via_yfinance",
            request_id=str(uuid4()),
            requested_symbol=requested_symbol,
            resolved_symbol=resolved_symbol,
            requested_interval=requested_interval,
            resolved_interval=resolved_interval,
            requested_start=_ts_to_iso(requested_start),
            requested_end=_ts_to_iso(requested_end),
            effective_start=_ts_to_iso(effective_start),
            effective_end=_ts_to_iso(effective_end),
            actual_start=None,
            actual_end=None,
            extracted_at_utc=_utcnow_iso(),
            auto_adjust=auto_adjust,
            actions=actions,
            warnings=list(warnings),
        )

    def _fetch_one_result(
        self,
        symbol: str,
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> FetchResult:
        requested_symbol = _normalize_symbol(symbol)
        requested_start = _to_naive_utc(start)
        requested_end = _to_naive_utc(end)
        requested_interval, resolved_interval = _normalize_interval(interval)
        effective_start, effective_end, warnings = self._prepare_request_window(
            resolved_interval,
            requested_start,
            requested_end,
        )

        metadata = self._build_metadata(
            requested_symbol=requested_symbol,
            resolved_symbol=requested_symbol,
            requested_interval=requested_interval,
            resolved_interval=resolved_interval,
            requested_start=requested_start,
            requested_end=requested_end,
            effective_start=effective_start,
            effective_end=effective_end,
            auto_adjust=auto_adjust,
            actions=actions,
            warnings=warnings,
        )

        if self.min_delay > 0:
            time.sleep(self.min_delay * random.uniform(0.5, 1.5))

        backends = ("ticker", "download")
        last_error: Optional[Exception] = None

        for attempt_number in range(1, self.retries + 1):
            for backend in backends:
                started = time.perf_counter()
                try:
                    frame, chunked, chunk_count = self._download_range(
                        backend=backend,
                        symbol=requested_symbol,
                        start=effective_start,
                        end=effective_end,
                        interval=resolved_interval,
                        auto_adjust=auto_adjust,
                        actions=actions,
                    )
                    duration = time.perf_counter() - started

                    metadata.attempts.append(
                        DownloadAttempt(
                            attempt_number=attempt_number,
                            backend=backend,
                            interval=resolved_interval,
                            start=_ts_to_iso(effective_start),
                            end=_ts_to_iso(effective_end),
                            duration_seconds=round(duration, 6),
                            success=not frame.empty,
                            rows=int(len(frame)),
                            error=None if not frame.empty else "empty dataframe",
                        )
                    )

                    if frame.empty:
                        last_error = EmptyDatasetError(
                            f"Yahoo returned no usable rows for {requested_symbol} at interval={resolved_interval}."
                        )
                        continue

                    actual_start, actual_end = _extract_actual_bounds(frame)
                    metadata.actual_start = _ts_to_iso(actual_start)
                    metadata.actual_end = _ts_to_iso(actual_end)
                    metadata.backend_used = backend
                    metadata.chunked = chunked
                    metadata.chunk_count = chunk_count
                    metadata.row_count = int(len(frame))

                    frame = _attach_metadata(frame, metadata)
                    return FetchResult(symbol=requested_symbol, data=frame, metadata=metadata)

                except Exception as exc:
                    duration = time.perf_counter() - started
                    last_error = exc
                    metadata.attempts.append(
                        DownloadAttempt(
                            attempt_number=attempt_number,
                            backend=backend,
                            interval=resolved_interval,
                            start=_ts_to_iso(effective_start),
                            end=_ts_to_iso(effective_end),
                            duration_seconds=round(duration, 6),
                            success=False,
                            rows=0,
                            error=str(exc),
                        )
                    )

            if attempt_number < self.retries:
                sleep_seconds = (2 ** (attempt_number - 1)) * 0.8 + random.random() * 0.4
                time.sleep(sleep_seconds)

        if last_error is None:
            last_error = DownloadFailureError(
                f"Unknown download failure for {requested_symbol}."
            )

        raise DownloadFailureError(
            f"Failed to download {requested_symbol} after {self.retries} retries. "
            f"Requested interval={requested_interval!r}, resolved interval={resolved_interval!r}."
        ) from last_error

    def _failure_result(
        self,
        symbol: str,
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
        interval: str,
        auto_adjust: bool,
        actions: bool,
        error: Exception,
    ) -> FetchResult:
        requested_symbol = str(symbol or "").strip().upper()
        requested_start = _to_naive_utc(start) if start not in (None, "") else None
        requested_end = _to_naive_utc(end) if end not in (None, "") else None

        try:
            requested_interval, resolved_interval = _normalize_interval(interval)
        except Exception:
            requested_interval = str(interval)
            resolved_interval = str(interval)

        metadata = self._build_metadata(
            requested_symbol=requested_symbol,
            resolved_symbol=requested_symbol,
            requested_interval=requested_interval,
            resolved_interval=resolved_interval,
            requested_start=requested_start,
            requested_end=requested_end,
            effective_start=requested_start,
            effective_end=requested_end,
            auto_adjust=auto_adjust,
            actions=actions,
            warnings=[f"Symbol failed during batch retrieval: {error}"],
        )
        metadata.attempts.append(
            DownloadAttempt(
                attempt_number=0,
                backend="n/a",
                interval=resolved_interval,
                start=_ts_to_iso(requested_start),
                end=_ts_to_iso(requested_end),
                duration_seconds=0.0,
                success=False,
                rows=0,
                error=str(error),
            )
        )

        frame = _attach_metadata(_empty_export_frame(), metadata)
        return FetchResult(symbol=requested_symbol, data=frame, metadata=metadata)

    def get_history_bundle(
        self,
        symbols: Union[str, Iterable[str]],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        auto_adjust: bool = False,
        actions: bool = True,
    ) -> Union[FetchResult, Dict[str, FetchResult]]:
        if isinstance(symbols, str):
            return self._fetch_one_result(
                symbol=symbols,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            )

        normalized_symbols: List[str] = []
        seen = set()
        for item in symbols:
            symbol = _normalize_symbol(str(item))
            if symbol not in seen:
                normalized_symbols.append(symbol)
                seen.add(symbol)

        if not normalized_symbols:
            return {}

        results: Dict[str, FetchResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_one_result,
                    symbol,
                    start,
                    end,
                    interval,
                    auto_adjust,
                    actions,
                ): symbol
                for symbol in normalized_symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as exc:
                    logger.warning("Batch retrieval failed for %s: %s", symbol, exc)
                    results[symbol] = self._failure_result(
                        symbol=symbol,
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=auto_adjust,
                        actions=actions,
                        error=exc,
                    )

        return results

    def get_history(
        self,
        symbols: Union[str, Iterable[str]],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        auto_adjust: bool = False,
        actions: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        bundle = self.get_history_bundle(
            symbols=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )

        if isinstance(bundle, FetchResult):
            return bundle.data

        return {symbol: result.data for symbol, result in bundle.items()}

    async def get_history_async(
        self,
        symbols: Union[str, Iterable[str]],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        auto_adjust: bool = False,
        actions: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_history(
                symbols=symbols,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            ),
        )

    def get_export_columns(self) -> List[str]:
        return list(EXPORT_COLUMNS)

    def get_provider_info(self) -> dict:
        return {
            "provider_name": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "max_workers": self.max_workers,
            "retries": self.retries,
            "timeout": self.timeout,
            "min_delay": self.min_delay,
            "max_intraday_lookback_days": self.max_intraday_lookback_days,
            "allow_partial_intraday": self.allow_partial_intraday,
            "export_columns": list(EXPORT_COLUMNS),
        }
