from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock, get_ident
from types import SimpleNamespace

import pandas as pd

from providers.yfinance_provider import FetchMetadata, FetchResult


def make_provider_frame(symbol: str = "MSFT", periods: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [100 + idx for idx in range(periods)],
            "high": [101 + idx for idx in range(periods)],
            "low": [99 + idx for idx in range(periods)],
            "close": [100.5 + idx for idx in range(periods)],
            "adj_close": [100.5 + idx for idx in range(periods)],
            "volume": [1_000_000 + idx * 10 for idx in range(periods)],
            "dividends": [0.0 for _ in range(periods)],
            "stock_splits": [0.0 for _ in range(periods)],
        }
    )
    frame.attrs["ab3_provenance"] = {"requested_symbol": symbol, "resolved_symbol": symbol}
    return frame


def make_raw_split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [400.0, 100.0, 101.0, 102.0],
            "high": [420.0, 105.0, 106.0, 107.0],
            "low": [390.0, 95.0, 96.0, 97.0],
            "close": [410.0, 102.0, 103.0, 104.0],
            "adj_close": [pd.NA, pd.NA, pd.NA, pd.NA],
            "volume": [1_000.0, 4_000.0, 4_100.0, 4_200.0],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 4.0, 0.0, 0.0],
        }
    )


def make_split_frame() -> pd.DataFrame:
    return make_raw_split_frame()


def make_nvda_like_split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12"]),
            "open": [120.0, 121.0, 123.0, 124.0],
            "high": [122.0, 123.0, 125.0, 126.0],
            "low": [118.5, 119.5, 121.5, 122.5],
            "close": [120.5, 121.2, 123.1, 124.2],
            "adj_close": [120.5, 121.2, 123.1, 124.2],
            "volume": [10_500_000.0, 28_000_000.0, 29_500_000.0, 27_900_000.0],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 10.0, 0.0, 0.0],
        }
    )


def make_nvda_like_frame_without_adj_close() -> pd.DataFrame:
    frame = make_nvda_like_split_frame()
    frame["adj_close"] = pd.NA
    return frame


def make_double_adjusted_qlib_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12"]),
            "open": [30.0, 121.0, 123.0, 124.0],
            "high": [30.5, 123.0, 125.0, 126.0],
            "low": [29.6, 119.5, 121.5, 122.5],
            "close": [30.125, 121.2, 123.1, 124.2],
            "volume": [105_000_000.0, 28_000_000.0, 29_500_000.0, 27_900_000.0],
            "factor": [0.25, 1.0, 1.0, 1.0],
        }
    )


def make_metadata(symbol: str, frame: pd.DataFrame) -> FetchMetadata:
    return FetchMetadata(
        provider_name="TestProvider",
        provider_version="0.0.1",
        source="unit-test",
        request_id=f"req-{symbol}",
        requested_symbol=symbol,
        resolved_symbol=symbol,
        requested_interval="1d",
        resolved_interval="1d",
        requested_start=str(frame["date"].min()),
        requested_end=str(frame["date"].max()),
        effective_start=str(frame["date"].min()),
        effective_end=str(frame["date"].max()),
        actual_start=str(frame["date"].min()),
        actual_end=str(frame["date"].max()),
        extracted_at_utc="2026-03-28T00:00:00+00:00",
        auto_adjust=False,
        actions=True,
        row_count=len(frame),
        warnings=[],
        attempts=[],
    )


def make_fetch_result(symbol: str, frame: pd.DataFrame) -> FetchResult:
    return FetchResult(symbol=symbol, data=frame.copy(), metadata=make_metadata(symbol, frame))


class DummyAcquisitionService:
    def __init__(self, datasets: dict[str, pd.DataFrame | Exception]) -> None:
        self.datasets = {key.upper(): value for key, value in datasets.items()}
        self.last_session = None
        self._lock = Lock()
        self.fetch_many_inputs: list[tuple[str, ...]] = []
        self.fetch_many_thread_ids: set[int] = set()

    def create_session(self, request):
        session = SimpleNamespace(
            request=request,
            bundle_cache={},
            lock=self._lock,
            metrics={
                "provider_instances": 1,
                "fetch_calls": 0,
                "fetch_many_calls": 0,
                "bundle_cache_hits": 0,
                "bundle_cache_misses": 0,
            },
        )
        self.last_session = session
        return session

    @staticmethod
    def _normalize_symbols(symbols):
        seen = set()
        normalized = []
        for item in symbols:
            symbol = str(item or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        return normalized

    def fetch_many(self, symbols, request, auto_adjust, actions, *, session=None):
        active_session = session or self.create_session(request)
        normalized_symbols = self._normalize_symbols(symbols)
        with self._lock:
            self.fetch_many_inputs.append(tuple(normalized_symbols))
            self.fetch_many_thread_ids.add(get_ident())
        cache_key = (
            tuple(normalized_symbols),
            request.interval,
            request.time_range.start_iso,
            request.time_range.end_iso,
            bool(auto_adjust),
            bool(actions),
        )
        with self._lock:
            cached = active_session.bundle_cache.get(cache_key)
        if cached is not None:
            with self._lock:
                active_session.metrics["bundle_cache_hits"] += 1
            return {symbol: cached[symbol] for symbol in normalized_symbols}

        with self._lock:
            active_session.metrics["bundle_cache_misses"] += 1
            active_session.metrics["fetch_many_calls"] += 1
        bundle = {}
        for symbol in normalized_symbols:
            value = self.datasets.get(symbol)
            if value is None:
                raise ValueError(f"Unknown symbol in test service: {symbol}")
            if isinstance(value, Exception):
                raise value
            bundle[symbol] = make_fetch_result(symbol, value)

        with self._lock:
            active_session.bundle_cache[cache_key] = dict(bundle)
        return bundle

    def fetch(self, symbol, request, auto_adjust, actions, *, session=None):
        active_session = session or self.create_session(request)
        with self._lock:
            active_session.metrics["fetch_calls"] += 1
        return self.fetch_many([symbol], request, auto_adjust, actions, session=active_session)[symbol.upper()]


@dataclass
class FakeContext:
    requested_symbol: str
    preferred_symbol: str
    listing_preference: str = "exact_symbol"
    warnings: list[str] | None = None
    market: str | None = "XNYS"
    asset_type: str = "equity"
    asset_family: str = "cash_equity"
    quote_type: str = "EQUITY"
    calendar: str | None = "XNYS"
    timezone: str | None = "America/New_York"
    currency: str | None = "USD"
    exchange_name: str | None = "NYSE"
    exchange_code: str | None = "NYQ"
    region: str = "USA"
    is_24_7: bool = False
    volume_expected: bool = True
    corporate_actions_expected: bool = True
    calendar_validation_supported: bool = True
    dq_profile: str = "equity"
    confidence: str = "high"
    inference_sources: list[str] | None = None
    structured_warnings: list[dict] | None = None
    resolution_trace: list[dict] | None = None
    resolver_metrics: dict | None = None
    raw_metadata: dict | None = None
    resolved_symbol: str | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.inference_sources is None:
            self.inference_sources = ["test"]
        if self.structured_warnings is None:
            self.structured_warnings = []
        if self.resolution_trace is None:
            self.resolution_trace = []
        if self.resolver_metrics is None:
            self.resolver_metrics = {}
        if self.raw_metadata is None:
            self.raw_metadata = {}
        if self.resolved_symbol is None:
            self.resolved_symbol = self.preferred_symbol

    def to_dict(self):
        return asdict(self)


def make_dq_context_payload(context: FakeContext) -> dict[str, object]:
    return {
        "asset_type": context.asset_type,
        "asset_family": context.asset_family,
        "quote_type": context.quote_type,
        "market": context.market,
        "calendar": context.calendar,
        "timezone": context.timezone,
        "currency": context.currency,
        "exchange_name": context.exchange_name,
        "exchange_code": context.exchange_code,
        "region": context.region,
        "requested_symbol": context.requested_symbol,
        "preferred_symbol": context.preferred_symbol,
        "listing_preference": context.listing_preference,
        "is_24_7": context.is_24_7,
        "volume_expected": context.volume_expected,
        "corporate_actions_expected": context.corporate_actions_expected,
        "calendar_validation_supported": context.calendar_validation_supported,
        "dq_profile": context.dq_profile,
        "confidence": context.confidence,
        "warnings": list(context.warnings),
        "structured_warnings": list(context.structured_warnings),
        "inference_sources": list(context.inference_sources),
        "resolution_trace": list(context.resolution_trace),
        "resolver_metrics": dict(context.resolver_metrics),
    }
