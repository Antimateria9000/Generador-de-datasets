from __future__ import annotations

from dataclasses import asdict, dataclass

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


def make_split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [400.0, 100.0, 101.0, 102.0],
            "high": [420.0, 105.0, 106.0, 107.0],
            "low": [390.0, 95.0, 96.0, 97.0],
            "close": [410.0, 102.0, 103.0, 104.0],
            "adj_close": [410.0, 102.0, 103.0, 104.0],
            "volume": [1_000.0, 4_000.0, 4_100.0, 4_200.0],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 4.0, 0.0, 0.0],
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

    def fetch(self, symbol, request, auto_adjust, actions):
        value = self.datasets.get(symbol.upper())
        if value is None:
            raise ValueError(f"Unknown symbol in test service: {symbol}")
        if isinstance(value, Exception):
            raise value
        return make_fetch_result(symbol.upper(), value)


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
    raw_metadata: dict | None = None
    resolved_symbol: str | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.inference_sources is None:
            self.inference_sources = ["test"]
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
        "inference_sources": list(context.inference_sources),
    }
