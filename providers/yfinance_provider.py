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
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
import yfinance as yf

from dataset_core.date_windows import DateWindowError, is_daily_like_interval, is_intraday_interval, validate_provider_window
from dataset_core.settings import DEFAULT_YFINANCE_CACHE_MODE, SUPPORTED_INTERVALS, normalize_yfinance_cache_mode

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


class FetchState(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


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
    fetch_state: FetchState = FetchState.SUCCESS
    failure_kind: Optional[str] = None
    semantic_flags: Dict[str, object] = field(default_factory=dict)
    structured_warnings: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    attempts: List[DownloadAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["fetch_state"] = self.fetch_state.value
        payload["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return payload


@dataclass
class FetchResult:
    symbol: str
    data: pd.DataFrame
    metadata: FetchMetadata


def _coerce_legacy_init_kwargs(
    max_workers: int | dict[str, object],
    retries: int,
    timeout: float,
    metadata_timeout: float | None,
    min_delay: float,
    max_intraday_lookback_days: int,
    cache_dir: str | Path | None,
    cache_mode: str,
    allow_partial_intraday: bool,
) -> tuple[int, int, float, float | None, float, int, str | Path | None, str, bool]:
    if not isinstance(max_workers, dict):
        return (
            int(max_workers),
            retries,
            timeout,
            metadata_timeout,
            min_delay,
            max_intraday_lookback_days,
            cache_dir,
            cache_mode,
            allow_partial_intraday,
        )

    # Backward-compatibility shim for legacy callers that passed a config dict
    # as the first positional argument. Keep it working, but do not extend it.
    logger.debug(
        "Legacy dict-based YFinanceProvider initialization is deprecated; pass keyword arguments instead."
    )
    params = dict(max_workers)
    return (
        int(params.get("max_workers", DEFAULT_MAX_WORKERS)),
        int(params.get("retries", DEFAULT_RETRIES)),
        float(params.get("timeout", DEFAULT_TIMEOUT)),
        params.get("metadata_timeout"),
        float(params.get("min_delay", DEFAULT_MIN_DELAY)),
        int(params.get("max_intraday_lookback_days", DEFAULT_MAX_INTRADAY_LOOKBACK_DAYS)),
        params.get("cache_dir"),
        str(params.get("cache_mode", DEFAULT_YFINANCE_CACHE_MODE)),
        bool(params.get("allow_partial_intraday", False)),
    )


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
    return _intraday_lookback_timedelta(max_lookback_days)


def _intraday_lookback_timedelta(max_lookback_days: int) -> pd.Timedelta:
    return pd.Timedelta(days=max(1, max_lookback_days - 1))


def _oldest_allowed_intraday_start(end: pd.Timestamp, max_lookback_days: int) -> pd.Timestamp:
    return end - _intraday_lookback_timedelta(max_lookback_days)


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

    semantic_warnings = _frame_semantic_warnings(out)
    semantic_flags = _frame_semantic_flags(out)
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
        semantic_flags["close_source"] = "adj_close_fallback"
        semantic_flags["close_derived_from_adj_close"] = True
        semantic_warnings.append(
            {
                "code": "close_from_adj_close",
                "severity": "warning",
                "target_column": "Close",
                "source_column": "Adj Close",
                "message": "Yahoo omitted Close; Close was derived from Adj Close with explicit semantic trace.",
            }
        )

    required_ohlc = [column for column in ["Open", "High", "Low", "Close"] if column in out.columns]
    if required_ohlc:
        out = out.dropna(subset=required_ohlc)

    return _apply_semantic_attrs(out, warnings=semantic_warnings, flags=semantic_flags)


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
            elif column == "adj_close":
                out[column] = pd.NA
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


_SEMANTIC_WARNING_ATTR = "ab3_structured_warnings"
_SEMANTIC_FLAGS_ATTR = "ab3_semantic_flags"


def _frame_semantic_warnings(df: pd.DataFrame) -> list[dict[str, object]]:
    warnings = df.attrs.get(_SEMANTIC_WARNING_ATTR, [])
    return [dict(item) for item in warnings if isinstance(item, dict)]


def _frame_semantic_flags(df: pd.DataFrame) -> dict[str, object]:
    flags = df.attrs.get(_SEMANTIC_FLAGS_ATTR, {})
    return dict(flags) if isinstance(flags, dict) else {}


def _apply_semantic_attrs(
    df: pd.DataFrame,
    *,
    warnings: list[dict[str, object]],
    flags: dict[str, object],
) -> pd.DataFrame:
    df.attrs[_SEMANTIC_WARNING_ATTR] = [dict(item) for item in warnings]
    df.attrs[_SEMANTIC_FLAGS_ATTR] = dict(flags)
    return df


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

    # yfinance cache managers are process-wide singletons, so every cache
    # reconfiguration is serialized and applied lazily at request time.
    _cache_operation_lock = RLock()
    _cache_configuration_signature: tuple[str, str | None] | None = None

    def __init__(
        self,
        max_workers: int | dict[str, object] = DEFAULT_MAX_WORKERS,
        retries: int = DEFAULT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        metadata_timeout: float | None = None,
        min_delay: float = DEFAULT_MIN_DELAY,
        max_intraday_lookback_days: int = DEFAULT_MAX_INTRADAY_LOOKBACK_DAYS,
        cache_dir: str | Path | None = None,
        cache_mode: str = DEFAULT_YFINANCE_CACHE_MODE,
        allow_partial_intraday: bool = False,
    ) -> None:
        (
            max_workers,
            retries,
            timeout,
            metadata_timeout,
            min_delay,
            max_intraday_lookback_days,
            cache_dir,
            cache_mode,
            allow_partial_intraday,
        ) = _coerce_legacy_init_kwargs(
            max_workers,
            retries,
            timeout,
            metadata_timeout,
            min_delay,
            max_intraday_lookback_days,
            cache_dir,
            cache_mode,
            allow_partial_intraday,
        )

        self.max_workers = int(max_workers)
        self.retries = int(retries)
        self.timeout = float(timeout)
        self.metadata_timeout = self.timeout if metadata_timeout is None else float(metadata_timeout)
        self.min_delay = float(min_delay)
        self.max_intraday_lookback_days = int(max_intraday_lookback_days)
        try:
            self.cache_mode = normalize_yfinance_cache_mode(cache_mode)
        except ValueError as exc:
            raise ProviderConfigurationError(str(exc)) from exc
        self.cache_dir = None if cache_dir in (None, "") else Path(cache_dir).expanduser().resolve()
        self.effective_cache_dir: Path | None = None
        self.cache_enabled = self.cache_mode != "off"
        self.allow_partial_intraday = bool(allow_partial_intraday)

        if self.max_workers < 1:
            raise ProviderConfigurationError("max_workers must be >= 1.")
        if self.retries < 1:
            raise ProviderConfigurationError("retries must be >= 1.")
        if self.timeout <= 0:
            raise ProviderConfigurationError("timeout must be > 0.")
        if self.metadata_timeout <= 0:
            raise ProviderConfigurationError("metadata_timeout must be > 0.")
        if self.min_delay < 0:
            raise ProviderConfigurationError("min_delay must be >= 0.")
        if self.max_intraday_lookback_days < 1:
            raise ProviderConfigurationError("max_intraday_lookback_days must be >= 1.")

        self.effective_cache_dir = self._resolve_cache_dir(self.cache_dir, cache_mode=self.cache_mode)
        self.cache_enabled = self.effective_cache_dir is not None

    @staticmethod
    def _reset_project_cache_state() -> None:
        try:
            import yfinance.cache as yf_cache
        except Exception:
            return

        for manager_name, cache_attr in (
            ("_TzCacheManager", "_tz_cache"),
            ("_CookieCacheManager", "_Cookie_cache"),
            ("_ISINCacheManager", "_isin_cache"),
        ):
            manager = getattr(yf_cache, manager_name, None)
            if manager is None:
                continue
            cache_instance = getattr(manager, cache_attr, None)
            db = getattr(cache_instance, "db", None)
            try:
                if db is not None:
                    db.close()
            except Exception:
                pass
            setattr(manager, cache_attr, None)

        for db_manager_name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
            db_manager = getattr(yf_cache, db_manager_name, None)
            if db_manager is None:
                continue
            close_db = getattr(db_manager, "close_db", None)
            if callable(close_db):
                try:
                    close_db()
                except Exception:
                    pass

    @staticmethod
    def _disable_project_cache() -> None:
        YFinanceProvider._reset_project_cache_state()
        try:
            import yfinance.cache as yf_cache
        except Exception:
            return
        if hasattr(yf_cache, "_TzCacheDummy") and hasattr(yf_cache, "_TzCacheManager"):
            yf_cache._TzCacheManager._tz_cache = yf_cache._TzCacheDummy()
        if hasattr(yf_cache, "_CookieCacheDummy") and hasattr(yf_cache, "_CookieCacheManager"):
            yf_cache._CookieCacheManager._Cookie_cache = yf_cache._CookieCacheDummy()
        if hasattr(yf_cache, "_ISINCacheDummy") and hasattr(yf_cache, "_ISINCacheManager"):
            yf_cache._ISINCacheManager._isin_cache = yf_cache._ISINCacheDummy()

    @staticmethod
    def _resolve_cache_dir(
        cache_dir: Path | None = None,
        *,
        cache_mode: str = DEFAULT_YFINANCE_CACHE_MODE,
    ) -> Path | None:
        normalized_mode = normalize_yfinance_cache_mode(cache_mode)
        if normalized_mode == "off":
            return None

        project_root = Path(__file__).resolve().parents[1]
        resolved_cache_dir = (
            project_root / "workspace" / "cache" / "yfinance"
            if cache_dir is None
            else Path(cache_dir).expanduser().resolve()
        )
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            YFinanceProvider._reset_project_cache_state()
            probe_path = resolved_cache_dir / "probe.db"
            connection = sqlite3.connect(probe_path)
            connection.execute("create table if not exists probe(x int)")
            connection.commit()
            connection.close()
            probe_path.unlink(missing_ok=True)
            try:
                import yfinance.cache as yf_cache

                set_cache_location = getattr(yf_cache, "set_cache_location", None)
                if callable(set_cache_location):
                    set_cache_location(str(resolved_cache_dir))
                else:
                    yf.set_tz_cache_location(str(resolved_cache_dir))
            except Exception:
                yf.set_tz_cache_location(str(resolved_cache_dir))
            return resolved_cache_dir
        except Exception as exc:
            logger.warning(
                "yfinance sqlite cache is not available at %s; disabling yfinance caches. Reason: %s",
                resolved_cache_dir,
                exc,
            )
            return None

    @classmethod
    def _apply_project_cache_policy(
        cls,
        cache_dir: Path | None = None,
        *,
        cache_mode: str = DEFAULT_YFINANCE_CACHE_MODE,
    ) -> Path | None:
        normalized_mode = normalize_yfinance_cache_mode(cache_mode)
        signature = (normalized_mode, None if cache_dir is None else str(Path(cache_dir).expanduser().resolve()))
        if normalized_mode != "off" and cache_dir is None:
            normalized_mode = "off"
            signature = ("off", None)
        if cls._cache_configuration_signature == signature:
            return None if signature[0] == "off" else cache_dir
        if signature[0] == "off":
            cls._disable_project_cache()
            cls._cache_configuration_signature = signature
            return None

        assert cache_dir is not None
        resolved_cache_dir = Path(cache_dir).expanduser().resolve()
        cls._reset_project_cache_state()
        try:
            import yfinance.cache as yf_cache

            set_cache_location = getattr(yf_cache, "set_cache_location", None)
            if callable(set_cache_location):
                set_cache_location(str(resolved_cache_dir))
            else:
                yf.set_tz_cache_location(str(resolved_cache_dir))
            cls._cache_configuration_signature = signature
            return resolved_cache_dir
        except Exception as exc:
            logger.warning(
                "Failed to activate yfinance sqlite cache at %s; disabling yfinance caches for this process. Reason: %s",
                resolved_cache_dir,
                exc,
            )
            cls._disable_project_cache()
            cls._cache_configuration_signature = ("off", None)
            return None

    def _execute_with_cache_lock(self, operation):
        with self._cache_operation_lock:
            active_cache_dir = self._apply_project_cache_policy(
                self.effective_cache_dir,
                cache_mode=self.cache_mode if self.cache_enabled else "off",
            )
            self.effective_cache_dir = active_cache_dir
            self.cache_enabled = active_cache_dir is not None
            return operation()

    def _download_via_ticker(
        self,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        def _history_request() -> pd.DataFrame:
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

        return self._execute_with_cache_lock(_history_request)

    def _download_via_download(
        self,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        def _download_request() -> pd.DataFrame:
            try:
                return yf.download(
                    tickers=symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    timeout=self.timeout,
                    progress=False,
                    threads=False,
                    keepna=True,
                )
            except TypeError:
                return yf.download(
                    tickers=symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    timeout=self.timeout,
                    progress=False,
                    threads=False,
                )

        raw = self._execute_with_cache_lock(_download_request)
        return _flatten_columns_if_needed(raw, symbol)

    def _download_via_download_many(
        self,
        symbols: List[str],
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        tickers = " ".join(symbols)
        def _download_many_request() -> pd.DataFrame:
            try:
                return yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    timeout=self.timeout,
                    progress=False,
                    threads=False,
                    keepna=True,
                )
            except TypeError:
                return yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    timeout=self.timeout,
                    progress=False,
                    threads=False,
                )

        return self._execute_with_cache_lock(_download_many_request)

    @staticmethod
    def _extract_symbol_from_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        if not hasattr(raw, "columns") or getattr(raw.columns, "nlevels", 1) <= 1:
            return _flatten_columns_if_needed(raw, symbol)

        for level in range(raw.columns.nlevels):
            for candidate in raw.columns.get_level_values(level).unique():
                if str(candidate).strip().upper() != symbol:
                    continue
                try:
                    extracted = raw.xs(candidate, axis=1, level=level, drop_level=True)
                except Exception:
                    continue
                if isinstance(extracted, pd.Series):
                    extracted = extracted.to_frame()
                return _flatten_columns_if_needed(extracted, symbol)

        return pd.DataFrame()

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
            effective_start = _oldest_allowed_intraday_start(effective_end, self.max_intraday_lookback_days)
            warnings.append(
                "Intraday request omitted start; effective_start was inferred from the configured intraday lookback window."
            )

        _validate_date_range(effective_start, effective_end)

        oldest_allowed = _oldest_allowed_intraday_start(effective_end, self.max_intraday_lookback_days)
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
        semantic_warnings = _frame_semantic_warnings(frame)
        semantic_flags = _frame_semantic_flags(frame)
        frame = _select_export_columns(frame)
        frame = _apply_semantic_attrs(frame, warnings=semantic_warnings, flags=semantic_flags)

        if frame.empty:
            return frame

        if frame["date"].duplicated().any():
            frame = frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        frame = frame.sort_values("date").reset_index(drop=True)
        return _apply_semantic_attrs(frame, warnings=semantic_warnings, flags=semantic_flags)

    def _merge_chunk_frames(
        self,
        frames: List[pd.DataFrame],
        interval: str,
    ) -> pd.DataFrame:
        if not frames:
            return _empty_export_frame()

        semantic_warnings: list[dict[str, object]] = []
        semantic_flags: dict[str, object] = {}
        for frame in frames:
            semantic_warnings.extend(_frame_semantic_warnings(frame))
            semantic_flags.update(_frame_semantic_flags(frame))

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

        merged = merged[EXPORT_COLUMNS].copy()
        return _apply_semantic_attrs(merged, warnings=semantic_warnings, flags=semantic_flags)

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
                    metadata.fetch_state = FetchState.SUCCESS
                    metadata.failure_kind = None
                    metadata.semantic_flags.update(_frame_semantic_flags(frame))
                    metadata.structured_warnings.extend(_frame_semantic_warnings(frame))
                    metadata.warnings.extend(
                        warning["message"]
                        for warning in metadata.structured_warnings
                        if isinstance(warning, dict) and str(warning.get("message", "")).strip()
                    )
                    if "adj_close" in frame.columns and pd.to_numeric(frame["adj_close"], errors="coerce").isna().all():
                        metadata.warnings.append(
                            "Provider returned no usable adj_close values; Qlib factor policy may need controlled fallback handling."
                        )

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
        metadata.fetch_state = FetchState.FAILED
        metadata.failure_kind = "batch_retrieval_failed"
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

    def _build_fetch_result_from_frame(
        self,
        *,
        symbol: str,
        frame: pd.DataFrame,
        requested_start: Optional[pd.Timestamp],
        requested_end: Optional[pd.Timestamp],
        requested_interval: str,
        resolved_interval: str,
        effective_start: Optional[pd.Timestamp],
        effective_end: Optional[pd.Timestamp],
        auto_adjust: bool,
        actions: bool,
        warnings: List[str],
        attempt_number: int,
        duration_seconds: float,
        backend: str,
    ) -> FetchResult:
        metadata = self._build_metadata(
            requested_symbol=symbol,
            resolved_symbol=symbol,
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
        metadata.attempts.append(
            DownloadAttempt(
                attempt_number=attempt_number,
                backend=backend,
                interval=resolved_interval,
                start=_ts_to_iso(effective_start),
                end=_ts_to_iso(effective_end),
                duration_seconds=round(duration_seconds, 6),
                success=not frame.empty,
                rows=int(len(frame)),
                error=None if not frame.empty else "empty dataframe",
            )
        )
        if frame.empty:
            metadata.fetch_state = FetchState.EMPTY
            metadata.failure_kind = "empty_dataset"
            return FetchResult(symbol=symbol, data=_attach_metadata(_empty_export_frame(), metadata), metadata=metadata)

        actual_start, actual_end = _extract_actual_bounds(frame)
        metadata.actual_start = _ts_to_iso(actual_start)
        metadata.actual_end = _ts_to_iso(actual_end)
        metadata.backend_used = backend
        metadata.chunked = False
        metadata.chunk_count = 1
        metadata.row_count = int(len(frame))
        metadata.fetch_state = FetchState.SUCCESS
        metadata.failure_kind = None
        metadata.semantic_flags.update(_frame_semantic_flags(frame))
        metadata.structured_warnings.extend(_frame_semantic_warnings(frame))
        metadata.warnings.extend(
            warning["message"]
            for warning in metadata.structured_warnings
            if isinstance(warning, dict) and str(warning.get("message", "")).strip()
        )
        if "adj_close" in frame.columns and pd.to_numeric(frame["adj_close"], errors="coerce").isna().all():
            metadata.warnings.append(
                "Provider returned no usable adj_close values; Qlib factor policy may need controlled fallback handling."
            )
        return FetchResult(symbol=symbol, data=_attach_metadata(frame, metadata), metadata=metadata)

    def _fetch_many_download_results(
        self,
        *,
        symbols: List[str],
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> tuple[Dict[str, FetchResult], List[str]]:
        requested_start = _to_naive_utc(start)
        requested_end = _to_naive_utc(end)
        requested_interval, resolved_interval = _normalize_interval(interval)
        effective_start, effective_end, warnings = self._prepare_request_window(
            resolved_interval,
            requested_start,
            requested_end,
        )

        if is_intraday_interval(resolved_interval):
            return {}, list(symbols)

        if self.min_delay > 0:
            time.sleep(self.min_delay * random.uniform(0.5, 1.5))

        last_error: Exception | None = None
        for attempt_number in range(1, self.retries + 1):
            started = time.perf_counter()
            try:
                raw = self._download_via_download_many(
                    symbols,
                    effective_start,
                    effective_end,
                    resolved_interval,
                    auto_adjust,
                    actions,
                )
                duration = time.perf_counter() - started
                batch_results: Dict[str, FetchResult] = {}
                missing_symbols: List[str] = []
                for symbol in symbols:
                    extracted = self._extract_symbol_from_download(raw, symbol)
                    frame = self._normalize_raw_history(extracted, symbol, resolved_interval)
                    fetch_result = self._build_fetch_result_from_frame(
                        symbol=symbol,
                        frame=frame,
                        requested_start=requested_start,
                        requested_end=requested_end,
                        requested_interval=requested_interval,
                        resolved_interval=resolved_interval,
                        effective_start=effective_start,
                        effective_end=effective_end,
                        auto_adjust=auto_adjust,
                        actions=actions,
                        warnings=list(warnings),
                        attempt_number=attempt_number,
                        duration_seconds=duration,
                        backend="download_many",
                    )
                    if frame.empty:
                        missing_symbols.append(symbol)
                    else:
                        batch_results[symbol] = fetch_result
                if batch_results or missing_symbols:
                    return batch_results, missing_symbols
                last_error = EmptyDatasetError(
                    f"Yahoo returned no usable rows for batch request {symbols!r} at interval={resolved_interval}."
                )
            except Exception as exc:
                last_error = exc

            if attempt_number < self.retries:
                sleep_seconds = (2 ** (attempt_number - 1)) * 0.8 + random.random() * 0.4
                time.sleep(sleep_seconds)

        if last_error is not None:
            logger.warning("Block download failed for %s: %s", symbols, last_error)
        return {}, list(symbols)

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

        results, pending_symbols = self._fetch_many_download_results(
            symbols=normalized_symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        if not pending_symbols:
            return {symbol: results[symbol] for symbol in normalized_symbols}

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
                for symbol in pending_symbols
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

        return {symbol: results[symbol] for symbol in normalized_symbols}

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
            "metadata_timeout": self.metadata_timeout,
            "min_delay": self.min_delay,
            "max_intraday_lookback_days": self.max_intraday_lookback_days,
            "cache_mode": self.cache_mode,
            "cache_dir": None if self.effective_cache_dir is None else str(self.effective_cache_dir),
            "cache_enabled": self.cache_enabled,
            "cache_runtime_policy": "serialized_process_global",
            "allow_partial_intraday": self.allow_partial_intraday,
            "export_columns": list(EXPORT_COLUMNS),
        }
