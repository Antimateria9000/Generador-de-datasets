from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from dataset_core.external_sources import extract_source_metadata
from dataset_core.external_sources.base import (
    ExternalSourceAuthError,
    ExternalSourceCoverageError,
    ExternalSourceNetworkError,
    ExternalSourceNotFoundError,
    ExternalSourcePayloadError,
    ExternalSourceRateLimitError,
)
from dataset_core.external_sources.eodhd import (
    EODHDCorporateActionsReferenceSource,
    EODHDClient,
    EODHDPriceReferenceSource,
    EODHDPayload,
    parse_eodhd_dividends,
    parse_eodhd_prices,
    parse_eodhd_splits,
)


class _FakeResponse:
    def __init__(self, status_code: int, *, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSession:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, float]] = []

    def get(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_parse_eodhd_prices_returns_canonical_frame():
    frame = parse_eodhd_prices(
        [
            {
                "date": "2025-01-02",
                "open": 248.93,
                "high": 249.1,
                "low": 241.82,
                "close": 243.85,
                "adjusted_close": 242.5252,
                "volume": 55740700,
            }
        ]
    )

    assert list(frame.columns) == [
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
    assert frame.loc[0, "adj_close"] == pytest.approx(242.5252)
    assert frame.loc[0, "volume"] == pytest.approx(55740700)


def test_parse_eodhd_dividends_and_splits_return_sparse_event_frames():
    dividends = parse_eodhd_dividends(
        [
            {
                "date": "2024-02-09",
                "value": 0.24,
                "unadjustedValue": 0.24,
            }
        ]
    )
    splits = parse_eodhd_splits([{"date": "2020-08-31", "split": "4.000000/1.000000"}])

    assert dividends.loc[0, "dividends"] == pytest.approx(0.24)
    assert dividends.loc[0, "stock_splits"] == pytest.approx(0.0)
    assert splits.loc[0, "stock_splits"] == pytest.approx(4.0)
    assert splits.loc[0, "dividends"] == pytest.approx(0.0)


def test_eodhd_client_raises_timeout_as_network_error(tmp_path):
    session = _FakeSession([requests.Timeout("boom")])
    client = EODHDClient(
        api_key="demo",
        session=session,
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=1,
    )

    with pytest.raises(ExternalSourceNetworkError):
        client.fetch_prices("AAPL.US", "2025-01-02", "2025-01-10")


def test_eodhd_client_redacts_api_token_from_transport_errors(tmp_path):
    session = _FakeSession(
        [
            requests.RequestException(
                "boom https://eodhd.test/api/eod/AAPL.US?api_token=transport-secret&fmt=json"
            )
        ]
    )
    client = EODHDClient(
        api_key="transport-secret",
        session=session,
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=1,
    )

    with pytest.raises(ExternalSourceNetworkError) as exc_info:
        client.fetch_prices("AAPL.US", "2025-01-02", "2025-01-10")

    assert "transport-secret" not in str(exc_info.value)
    assert "***redacted***" in str(exc_info.value)


def test_eodhd_client_raises_rate_limit_error_on_429(tmp_path):
    session = _FakeSession([_FakeResponse(429, text="Too Many Requests")])
    client = EODHDClient(
        api_key="demo",
        session=session,
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=1,
    )

    with pytest.raises(ExternalSourceRateLimitError):
        client.fetch_prices("AAPL.US", "2025-01-02", "2025-01-10")


def test_eodhd_client_classifies_auth_errors_explicitly(tmp_path):
    session = _FakeSession([_FakeResponse(401, text="Invalid API key")])
    client = EODHDClient(
        api_key="bad-key",
        session=session,
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=1,
    )

    with pytest.raises(ExternalSourceAuthError):
        client.fetch_prices("AAPL.US", "2025-01-02", "2025-01-10")


def test_eodhd_client_reuses_cache_between_calls(tmp_path):
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload=[
                    {
                        "date": "2025-01-02",
                        "open": 248.93,
                        "high": 249.1,
                        "low": 241.82,
                        "close": 243.85,
                        "adjusted_close": 242.5252,
                        "volume": 55740700,
                    }
                ],
            )
        ]
    )
    client = EODHDClient(
        api_key="demo",
        session=session,
        cache_dir=tmp_path / "eodhd_cache",
        use_cache=True,
        cache_ttl_seconds=3600,
    )

    first = client.fetch_prices("AAPL.US", "2025-01-02", "2025-01-10")
    second = client.fetch_prices("AAPL.US", "2025-01-02", "2025-01-10")

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert len(session.calls) == 1
    assert client.metrics["cache_hits"] == 1
    assert client.metrics["cache_misses"] == 1
    assert "api_token=%2A%2A%2A" in first.url
    assert "demo" not in first.url


def test_eodhd_price_source_attaches_observable_metadata(tmp_path):
    class _FallbackClient:
        def __init__(self) -> None:
            self.metrics = {"request_count": 2, "cache_hits": 0, "cache_misses": 2}
            self.seen = []

        def fetch_prices(self, symbol, start, end):
            self.seen.append(symbol)
            if symbol == "MSFT":
                raise ExternalSourceNotFoundError("symbol not found")
            return EODHDPayload(
                payload=[
                    {
                        "date": "2025-01-02",
                        "open": 248.93,
                        "high": 249.1,
                        "low": 241.82,
                        "close": 243.85,
                        "adjusted_close": 242.5252,
                        "volume": 55740700,
                    }
                ],
                url="https://eodhd.test/api/eod/MSFT.US?api_token=%2A%2A%2A",
                cache_status="miss",
                endpoint="/api/eod/MSFT.US",
            )

    client = _FallbackClient()
    source = EODHDPriceReferenceSource(client)

    frame = source.fetch_reference("MSFT", "2025-01-02", "2025-01-10")
    metadata = extract_source_metadata(frame)

    assert metadata["provider"] == "eodhd"
    assert metadata["provider_symbol"] == "MSFT.US"
    assert metadata["symbol_mapping"] == "exact_symbol"
    assert metadata["scope"] == "price"
    assert metadata["candidates_tried"] == ["MSFT", "MSFT.US"]


def test_eodhd_price_source_applies_local_temporal_filtering_with_end_exclusive():
    class _Client:
        def __init__(self) -> None:
            self.metrics = {"request_count": 1, "cache_hits": 0, "cache_misses": 1}

        def fetch_prices(self, symbol, start, end):
            return EODHDPayload(
                payload=[
                    {
                        "date": "2024-01-01",
                        "open": 10,
                        "high": 10,
                        "low": 10,
                        "close": 10,
                        "adjusted_close": 10,
                        "volume": 100,
                    },
                    {
                        "date": "2024-01-02",
                        "open": 11,
                        "high": 11,
                        "low": 11,
                        "close": 11,
                        "adjusted_close": 11,
                        "volume": 110,
                    },
                    {
                        "date": "2024-01-03",
                        "open": 12,
                        "high": 12,
                        "low": 12,
                        "close": 12,
                        "adjusted_close": 12,
                        "volume": 120,
                    },
                ],
                url="https://eodhd.test/api/eod/MSFT.US?api_token=%2A%2A%2A",
                cache_status="miss",
                endpoint="/api/eod/MSFT.US",
            )

    source = EODHDPriceReferenceSource(_Client(), price_lookback_days=3650)

    frame = source.fetch_reference("MSFT.US", "2024-01-02", "2024-01-03")

    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    assert extract_source_metadata(frame)["requested_start"] == "2024-01-02"


def test_eodhd_price_source_preserves_metadata_even_when_local_filter_yields_no_rows():
    class _Client:
        def __init__(self) -> None:
            self.metrics = {"request_count": 1, "cache_hits": 0, "cache_misses": 1}

        def fetch_prices(self, symbol, start, end):
            return EODHDPayload(
                payload=[
                    {
                        "date": "2024-01-01",
                        "open": 10,
                        "high": 10,
                        "low": 10,
                        "close": 10,
                        "adjusted_close": 10,
                        "volume": 100,
                    }
                ],
                url="https://eodhd.test/api/eod/MSFT.US?api_token=%2A%2A%2A",
                cache_status="miss",
                endpoint="/api/eod/MSFT.US",
            )

    source = EODHDPriceReferenceSource(_Client(), price_lookback_days=3650)

    frame = source.fetch_reference("MSFT.US", "2024-02-01", "2024-02-02")
    metadata = extract_source_metadata(frame)

    assert frame.empty
    assert metadata["provider"] == "eodhd"
    assert metadata["scope"] == "price"


def test_eodhd_corporate_actions_source_allows_partial_coverage_when_configured():
    class _PartialClient:
        def __init__(self) -> None:
            self.metrics = {"request_count": 1, "cache_hits": 0, "cache_misses": 0}

        def fetch_dividends(self, symbol, start, end):
            raise ExternalSourceCoverageError("dividends unavailable on current plan")

        def fetch_splits(self, symbol, start, end):
            return EODHDPayload(
                payload=[{"date": "2020-08-31", "split": "4.000000/1.000000"}],
                url="https://eodhd.example/splits",
                cache_status="miss",
                endpoint="/api/splits/AAPL.US",
            )

    source = EODHDCorporateActionsReferenceSource(_PartialClient(), allow_partial_coverage=True)

    frame = source.fetch_events("AAPL.US", "2020-01-01", "2021-01-01")
    metadata = extract_source_metadata(frame)

    assert frame.loc[0, "stock_splits"] == pytest.approx(4.0)
    assert metadata["partial_coverage"] is True
    assert metadata["coverage_notes"]


def test_eodhd_corporate_actions_source_applies_local_temporal_filtering_with_end_exclusive():
    class _Client:
        def __init__(self) -> None:
            self.metrics = {"request_count": 2, "cache_hits": 0, "cache_misses": 2}

        def fetch_dividends(self, symbol, start, end):
            return EODHDPayload(
                payload=[
                    {"date": "2024-01-01", "value": 0.4},
                    {"date": "2024-01-02", "value": 0.5},
                ],
                url="https://eodhd.example/div",
                cache_status="miss",
                endpoint="/api/div/MSFT.US",
            )

        def fetch_splits(self, symbol, start, end):
            return EODHDPayload(
                payload=[{"date": "2024-01-03", "split": "2/1"}],
                url="https://eodhd.example/splits",
                cache_status="miss",
                endpoint="/api/splits/MSFT.US",
            )

    source = EODHDCorporateActionsReferenceSource(_Client(), allow_partial_coverage=True)

    frame = source.fetch_events("MSFT.US", "2024-01-02", "2024-01-03")

    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]


def test_parse_eodhd_prices_rejects_invalid_payload_shape():
    with pytest.raises(ExternalSourcePayloadError):
        parse_eodhd_prices({"date": "2025-01-02"})
