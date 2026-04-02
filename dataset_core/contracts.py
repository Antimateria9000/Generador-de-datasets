from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

from dataset_core.date_windows import DateWindowError, resolve_temporal_bounds
from dataset_core.path_safety import normalize_filename_override
from dataset_core.settings import (
    DEFAULT_EODHD_BACKOFF_SECONDS,
    DEFAULT_EODHD_BASE_URL,
    DEFAULT_EODHD_CACHE_TTL_SECONDS,
    DEFAULT_EODHD_MAX_RETRIES,
    DEFAULT_EODHD_PRICE_LOOKBACK_DAYS,
    DEFAULT_EODHD_TIMEOUT_SECONDS,
    DEFAULT_OUTPUT_ROOT,
    DQ_MODES,
    LISTING_PREFERENCES,
    OPTIONAL_COLUMNS,
    PRESET_NAMES,
    SUPPORTED_INTERVALS,
    register_secret,
)

_TOKEN_SPLIT_RE = re.compile(r"[\s,;]+")
_NULL_TICKER_TOKENS = {"", "NAN", "NONE", "NULL", "NAT", "<NA>"}
_FILE_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-_=^/]*$")


class RequestContractError(ValueError):
    """Raised when a request cannot be normalized safely."""


def _parse_timestamp(raw: str, field_name: str) -> pd.Timestamp:
    value = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(value):
        raise RequestContractError(f"Invalid {field_name}: {raw!r}")
    return value.tz_convert(None)


def _normalize_symbol(symbol: str) -> str:
    cleaned = str(symbol or "").strip().upper()
    if not cleaned:
        return ""
    return cleaned.strip("\"'")


def _normalize_file_ticker_candidate(symbol: object) -> str:
    if symbol is None:
        return ""
    try:
        if pd.isna(symbol):
            return ""
    except TypeError:
        pass

    normalized = _normalize_symbol(str(symbol))
    if normalized in _NULL_TICKER_TOKENS:
        return ""
    if not _FILE_TICKER_RE.fullmatch(normalized):
        return ""
    return normalized


def dedupe_preserve_order(
    items: Iterable[object],
    normalizer: Callable[[object], str] = _normalize_symbol,
) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = normalizer(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def parse_tickers_text(raw: str) -> list[str]:
    tokens = [token for token in _TOKEN_SPLIT_RE.split(str(raw or "")) if token.strip()]
    return dedupe_preserve_order(tokens)


def load_tickers_from_file(path: Path) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise RequestContractError(f"Ticker file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path)
        candidate_columns = ["ticker", "tickers", "symbol", "symbols"]
        for column in candidate_columns:
            if column in frame.columns:
                return dedupe_preserve_order(frame[column].tolist(), normalizer=_normalize_file_ticker_candidate)
        flattened = frame.values.ravel().tolist()
        return dedupe_preserve_order(flattened, normalizer=_normalize_file_ticker_candidate)

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            content = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        return parse_tickers_text(content)

    raise RequestContractError(f"Ticker file could not be decoded: {file_path}")


def resolve_ticker_inputs(
    ticker: Optional[str] = None,
    tickers: Optional[str] = None,
    tickers_file: Optional[str] = None,
) -> list[str]:
    active_inputs = [value for value in (ticker, tickers, tickers_file) if str(value or "").strip()]
    if len(active_inputs) != 1:
        raise RequestContractError(
            "Exactly one of --ticker, --tickers or --tickers-file must be provided."
        )

    if str(ticker or "").strip():
        symbols = dedupe_preserve_order([ticker])
    elif str(tickers or "").strip():
        symbols = parse_tickers_text(str(tickers))
    else:
        symbols = load_tickers_from_file(Path(str(tickers_file)))

    if not symbols:
        raise RequestContractError("No valid tickers were found in the selected input.")
    return symbols


def parse_extras(raw: Optional[Sequence[str] | str]) -> list[str]:
    if raw is None:
        return []

    if isinstance(raw, str):
        candidates = parse_tickers_text(raw.replace("\n", ","))
    else:
        candidates = dedupe_preserve_order([str(item) for item in raw])

    normalized: list[str] = []
    valid = set(OPTIONAL_COLUMNS)
    for item in candidates:
        lowered = item.lower()
        if lowered not in valid:
            allowed = ", ".join(OPTIONAL_COLUMNS)
            raise RequestContractError(f"Unsupported extra column: {item!r}. Allowed: {allowed}.")
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


@dataclass(frozen=True)
class TemporalRange:
    mode: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    years: Optional[int]
    reproducible: bool
    start_iso: Optional[str]
    end_iso: Optional[str]

    @classmethod
    def from_inputs(
        cls,
        years: Optional[int],
        start: Optional[object],
        end: Optional[object],
        *,
        interval: str = "1d",
        now_utc: pd.Timestamp | None = None,
    ) -> "TemporalRange":
        has_years = years is not None
        if has_years:
            years = int(years)
        try:
            mode, range_start, range_end, normalized_years, reproducible = resolve_temporal_bounds(
                years=years,
                start=start,
                end=end,
                interval=interval,
                now_utc=now_utc,
            )
        except DateWindowError as exc:
            raise RequestContractError(str(exc)) from exc

        return cls(
            mode=mode,
            start=range_start,
            end=range_end,
            years=normalized_years,
            reproducible=reproducible,
            start_iso=range_start.isoformat(),
            end_iso=range_end.isoformat(),
        )


@dataclass(frozen=True)
class ProviderConfig:
    max_workers: Optional[int] = None
    retries: Optional[int] = None
    timeout: Optional[float] = None
    metadata_timeout: Optional[float] = None
    min_delay: Optional[float] = None
    max_intraday_lookback_days: Optional[int] = None
    cache_dir: Optional[Path] = None
    allow_partial_intraday: bool = False
    metadata_candidate_limit: Optional[int] = None
    context_cache_ttl_seconds: Optional[int] = None
    batch_max_workers: Optional[int] = None
    batch_chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_workers is not None and int(self.max_workers) < 1:
            raise RequestContractError("provider.max_workers must be >= 1.")
        if self.retries is not None and int(self.retries) < 1:
            raise RequestContractError("provider.retries must be >= 1.")
        if self.timeout is not None and float(self.timeout) <= 0:
            raise RequestContractError("provider.timeout must be > 0.")
        if self.metadata_timeout is not None and float(self.metadata_timeout) <= 0:
            raise RequestContractError("provider.metadata_timeout must be > 0.")
        if self.min_delay is not None and float(self.min_delay) < 0:
            raise RequestContractError("provider.min_delay must be >= 0.")
        if self.max_intraday_lookback_days is not None and int(self.max_intraday_lookback_days) < 1:
            raise RequestContractError("provider.max_intraday_lookback_days must be >= 1.")
        if self.metadata_candidate_limit is not None and int(self.metadata_candidate_limit) < 1:
            raise RequestContractError("provider.metadata_candidate_limit must be >= 1.")
        if self.context_cache_ttl_seconds is not None and int(self.context_cache_ttl_seconds) < 0:
            raise RequestContractError("provider.context_cache_ttl_seconds must be >= 0.")
        if self.batch_max_workers is not None and int(self.batch_max_workers) < 1:
            raise RequestContractError("provider.batch_max_workers must be >= 1.")
        if self.batch_chunk_size is not None and int(self.batch_chunk_size) < 1:
            raise RequestContractError("provider.batch_chunk_size must be >= 1.")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser().resolve())

    def to_runtime_kwargs(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.max_workers is not None:
            payload["max_workers"] = int(self.max_workers)
        if self.retries is not None:
            payload["retries"] = int(self.retries)
        if self.timeout is not None:
            payload["timeout"] = float(self.timeout)
        if self.metadata_timeout is not None:
            payload["metadata_timeout"] = float(self.metadata_timeout)
        if self.min_delay is not None:
            payload["min_delay"] = float(self.min_delay)
        if self.max_intraday_lookback_days is not None:
            payload["max_intraday_lookback_days"] = int(self.max_intraday_lookback_days)
        if self.cache_dir is not None:
            payload["cache_dir"] = self.cache_dir
        if self.allow_partial_intraday:
            payload["allow_partial_intraday"] = True
        return payload

    def to_kwargs(self) -> dict[str, object]:
        return self.to_runtime_kwargs()

    def to_dict(self) -> dict[str, object]:
        payload = self.to_runtime_kwargs()
        if self.metadata_candidate_limit is not None:
            payload["metadata_candidate_limit"] = int(self.metadata_candidate_limit)
        if self.context_cache_ttl_seconds is not None:
            payload["context_cache_ttl_seconds"] = int(self.context_cache_ttl_seconds)
        if self.batch_max_workers is not None:
            payload["batch_max_workers"] = int(self.batch_max_workers)
        if self.batch_chunk_size is not None:
            payload["batch_chunk_size"] = int(self.batch_chunk_size)
        return payload


@dataclass(frozen=True)
class EODHDExternalValidationConfig:
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: str = DEFAULT_EODHD_BASE_URL
    timeout_seconds: float = DEFAULT_EODHD_TIMEOUT_SECONDS
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    cache_ttl_seconds: int = DEFAULT_EODHD_CACHE_TTL_SECONDS
    allow_partial_coverage: bool = False
    max_retries: int = DEFAULT_EODHD_MAX_RETRIES
    backoff_seconds: float = DEFAULT_EODHD_BACKOFF_SECONDS
    price_lookback_days: int = DEFAULT_EODHD_PRICE_LOOKBACK_DAYS
    exchange_hint: Optional[str] = None
    symbol_map_file: Optional[Path] = None

    def __post_init__(self) -> None:
        api_key = None if self.api_key is None else str(self.api_key).strip()
        if api_key == "":
            api_key = None
        object.__setattr__(self, "api_key", api_key)
        register_secret(api_key)

        base_url = str(self.base_url or DEFAULT_EODHD_BASE_URL).strip().rstrip("/")
        if not base_url:
            raise RequestContractError("external_validation.eodhd.base_url cannot be empty.")
        object.__setattr__(self, "base_url", base_url)

        if float(self.timeout_seconds) <= 0:
            raise RequestContractError("external_validation.eodhd.timeout_seconds must be > 0.")
        if int(self.cache_ttl_seconds) < 0:
            raise RequestContractError("external_validation.eodhd.cache_ttl_seconds must be >= 0.")
        if int(self.max_retries) < 1:
            raise RequestContractError("external_validation.eodhd.max_retries must be >= 1.")
        if float(self.backoff_seconds) < 0:
            raise RequestContractError("external_validation.eodhd.backoff_seconds must be >= 0.")
        if int(self.price_lookback_days) < 1:
            raise RequestContractError("external_validation.eodhd.price_lookback_days must be >= 1.")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser().resolve())
        exchange_hint = None if self.exchange_hint is None else str(self.exchange_hint).strip().upper()
        if exchange_hint == "":
            exchange_hint = None
        object.__setattr__(self, "exchange_hint", exchange_hint)
        if self.symbol_map_file is not None:
            object.__setattr__(self, "symbol_map_file", Path(self.symbol_map_file).expanduser().resolve())

    def to_dict(self) -> dict[str, object]:
        return {
            "api_key_configured": bool(self.api_key),
            "base_url": self.base_url,
            "timeout_seconds": float(self.timeout_seconds),
            "use_cache": bool(self.use_cache),
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir.resolve()),
            "cache_ttl_seconds": int(self.cache_ttl_seconds),
            "allow_partial_coverage": bool(self.allow_partial_coverage),
            "max_retries": int(self.max_retries),
            "backoff_seconds": float(self.backoff_seconds),
            "price_lookback_days": int(self.price_lookback_days),
            "exchange_hint": self.exchange_hint,
            "symbol_map_file": None
            if self.symbol_map_file is None
            else str(self.symbol_map_file.resolve()),
        }


@dataclass(frozen=True)
class ExternalValidationConfig:
    enabled: Optional[bool] = None
    provider: Optional[str] = None
    reference_dir: Optional[Path] = None
    manual_events_file: Optional[Path] = None
    eodhd: EODHDExternalValidationConfig = field(default_factory=EODHDExternalValidationConfig)

    def __post_init__(self) -> None:
        if self.reference_dir is not None:
            object.__setattr__(self, "reference_dir", Path(self.reference_dir).expanduser().resolve())
        if self.manual_events_file is not None:
            object.__setattr__(self, "manual_events_file", Path(self.manual_events_file).expanduser().resolve())
        if self.provider is not None:
            normalized_provider = str(self.provider).strip().lower()
            if normalized_provider not in {"csv", "eodhd"}:
                raise RequestContractError(
                    "external_validation.provider must be either 'csv' or 'eodhd'."
                )
            object.__setattr__(self, "provider", normalized_provider)
        if self.enabled is not None:
            object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def has_legacy_sources(self) -> bool:
        return self.reference_dir is not None or self.manual_events_file is not None

    def resolved_provider(self) -> str | None:
        if self.provider:
            return self.provider
        if self.has_legacy_sources:
            return "csv"
        if self.eodhd.api_key:
            return "eodhd"
        return None

    def is_enabled(self) -> bool:
        if self.enabled is not None:
            return bool(self.enabled)
        return self.resolved_provider() is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.is_enabled(),
            "provider": self.resolved_provider(),
            "reference_dir": None if self.reference_dir is None else str(self.reference_dir.resolve()),
            "manual_events_file": None
            if self.manual_events_file is None
            else str(self.manual_events_file.resolve()),
            "eodhd": self.eodhd.to_dict(),
        }


@dataclass(frozen=True)
class DatasetRequest:
    tickers: list[str]
    time_range: TemporalRange
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    interval: str = "1d"
    mode: str = "base"
    extras: list[str] = field(default_factory=list)
    listing_preference: str = "exact_symbol"
    dq_mode: str = "report"
    dq_market: str = "AUTO"
    auto_adjust: bool = False
    actions: bool = True
    qlib_sanitization: bool = False
    filename_override: Optional[str] = None
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    external_validation: ExternalValidationConfig = field(default_factory=ExternalValidationConfig)

    def __post_init__(self) -> None:
        normalized_tickers = dedupe_preserve_order(self.tickers)
        if not normalized_tickers:
            raise RequestContractError("At least one ticker is required.")
        if self.interval not in SUPPORTED_INTERVALS:
            raise RequestContractError(f"Unsupported interval: {self.interval}")
        if self.mode not in PRESET_NAMES:
            raise RequestContractError(f"Unsupported mode: {self.mode}")
        if self.listing_preference not in LISTING_PREFERENCES:
            raise RequestContractError(f"Unsupported listing_preference: {self.listing_preference}")
        if self.dq_mode not in DQ_MODES:
            raise RequestContractError(f"Unsupported dq_mode: {self.dq_mode}")
        if self.filename_override and len(normalized_tickers) != 1:
            raise RequestContractError("--filename can only be used with a single ticker request.")

        object.__setattr__(self, "tickers", normalized_tickers)
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if self.filename_override:
            try:
                object.__setattr__(self, "filename_override", normalize_filename_override(self.filename_override))
            except ValueError as exc:
                raise RequestContractError(str(exc)) from exc
        normalized_extras = parse_extras(self.extras)
        if self.mode == "qlib":
            disallowed_qlib_extras = [extra for extra in normalized_extras if extra != "factor"]
            if disallowed_qlib_extras:
                raise RequestContractError(
                    "Preset qlib is closed and only allows the mandatory extra 'factor'. "
                    f"Forbidden extras: {', '.join(disallowed_qlib_extras)}."
                )
            normalized_extras = ["factor"]
        object.__setattr__(self, "extras", normalized_extras)
        object.__setattr__(self, "dq_market", str(self.dq_market or "AUTO").strip().upper())
        if self.mode == "qlib" and not self.qlib_sanitization:
            object.__setattr__(self, "qlib_sanitization", True)

    @property
    def requires_factor(self) -> bool:
        return self.mode == "qlib" or "factor" in self.extras

    @property
    def produces_parallel_qlib_artifact(self) -> bool:
        return self.mode != "qlib" and self.qlib_sanitization

    @property
    def batch_size(self) -> int:
        return len(self.tickers)

    def to_dict(self) -> dict[str, object]:
        return {
            "tickers": list(self.tickers),
            "time_range": {
                "mode": self.time_range.mode,
                "start": self.time_range.start_iso,
                "end": self.time_range.end_iso,
                "years": self.time_range.years,
                "reproducible": self.time_range.reproducible,
            },
            "output_dir": str(self.output_dir.resolve()),
            "interval": self.interval,
            "mode": self.mode,
            "extras": list(self.extras),
            "listing_preference": self.listing_preference,
            "dq_mode": self.dq_mode,
            "dq_market": self.dq_market,
            "auto_adjust": bool(self.auto_adjust),
            "actions": bool(self.actions),
            "qlib_sanitization": bool(self.qlib_sanitization),
            "filename_override": self.filename_override,
            "provider": self.provider.to_dict(),
            "external_validation": self.external_validation.to_dict(),
        }
