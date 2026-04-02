from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from dataset_core.acquisition import AcquisitionService
from dataset_core.contracts import DatasetRequest, ProviderConfig, TemporalRange
from providers.yfinance_provider import (
    DownloadAttempt,
    FetchMetadata,
    FetchResult,
    FetchState,
    YFinanceProvider,
    _clean_df,
)
from tests.fixtures.sample_data import make_fetch_result, make_provider_frame


def test_acquisition_service_passes_effective_workspace_cache_dir(tmp_path):
    captured_kwargs = {}

    class _Provider:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def get_history_bundle(self, **kwargs):
            return make_fetch_result("MSFT", make_provider_frame("MSFT"))

    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
    )

    AcquisitionService(provider_factory=_Provider).fetch(
        symbol="MSFT",
        request=request,
        auto_adjust=False,
        actions=True,
    )

    assert captured_kwargs["cache_dir"] == (tmp_path / "cache" / "yfinance").resolve()


def test_acquisition_service_uses_process_scoped_cache_dir_when_requested(tmp_path):
    captured_kwargs = {}

    class _Provider:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def get_history_bundle(self, **kwargs):
            return make_fetch_result("MSFT", make_provider_frame("MSFT"))

    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        provider=ProviderConfig(cache_mode="process"),
    )

    AcquisitionService(provider_factory=_Provider).fetch(
        symbol="MSFT",
        request=request,
        auto_adjust=False,
        actions=True,
    )

    cache_dir = captured_kwargs["cache_dir"]
    assert cache_dir.parent == (tmp_path / "cache" / "yfinance").resolve()
    assert cache_dir.name.startswith("process-")


def test_acquisition_service_can_disable_yfinance_cache_entirely(tmp_path):
    captured_kwargs = {}

    class _Provider:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def get_history_bundle(self, **kwargs):
            return make_fetch_result("MSFT", make_provider_frame("MSFT"))

    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        provider=ProviderConfig(cache_mode="off"),
    )

    AcquisitionService(provider_factory=_Provider).fetch(
        symbol="MSFT",
        request=request,
        auto_adjust=False,
        actions=True,
    )

    assert captured_kwargs["cache_mode"] == "off"
    assert captured_kwargs["cache_dir"] is None


def test_acquisition_service_uses_run_scoped_cache_namespace_when_requested(tmp_path):
    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_history_bundle(self, **kwargs):
            return make_fetch_result("MSFT", make_provider_frame("MSFT"))

    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        provider=ProviderConfig(cache_mode="run"),
    )

    session = AcquisitionService(provider_factory=_Provider).create_session(
        request,
        cache_namespace="run-123",
    )

    assert session.cache_namespace == "run-123"
    assert session.cache_dir == (tmp_path / "cache" / "yfinance" / "run-123").resolve()


def test_clean_df_flags_when_close_is_derived_from_adj_close():
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000, 1_100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    cleaned = _clean_df(raw)

    assert cleaned["Close"].tolist() == [100.5, 101.5]
    assert cleaned.attrs["ab3_semantic_flags"]["close_derived_from_adj_close"] is True
    assert cleaned.attrs["ab3_semantic_flags"]["close_source"] == "adj_close_fallback"
    assert any(warning["code"] == "close_from_adj_close" for warning in cleaned.attrs["ab3_structured_warnings"])


def test_acquisition_service_reuses_session_cache_for_repeated_fetch_many(tmp_path):
    provider_calls = []

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_history_bundle(self, **kwargs):
            symbols = kwargs["symbols"]
            provider_calls.append(tuple(symbols))
            return {symbol: make_fetch_result(symbol, make_provider_frame(symbol)) for symbol in symbols}

    request = DatasetRequest(
        tickers=["MSFT", "AAPL"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
    )

    service = AcquisitionService(provider_factory=_Provider)
    session = service.create_session(request)
    first = service.fetch_many(["MSFT", "AAPL"], request, auto_adjust=False, actions=True, session=session)
    second = service.fetch_many(["MSFT", "AAPL"], request, auto_adjust=False, actions=True, session=session)

    assert list(first) == ["MSFT", "AAPL"]
    assert list(second) == ["MSFT", "AAPL"]
    assert provider_calls == [("MSFT", "AAPL")]
    assert session.metrics["bundle_cache_misses"] == 1
    assert session.metrics["bundle_cache_hits"] == 1
    assert session.metrics["provider_instances"] == 1


def test_acquisition_service_classifies_fetch_results_with_structured_state():
    base_metadata = dict(
        provider_name="TestProvider",
        provider_version="0.0.1",
        source="unit-test",
        request_id="req-1",
        requested_symbol="MSFT",
        resolved_symbol="MSFT",
        requested_interval="1d",
        resolved_interval="1d",
        requested_start="2024-01-01T00:00:00",
        requested_end="2024-01-02T00:00:00",
        effective_start="2024-01-01T00:00:00",
        effective_end="2024-01-02T00:00:00",
        actual_start="2024-01-01T00:00:00",
        actual_end="2024-01-02T00:00:00",
        extracted_at_utc="2026-03-31T00:00:00+00:00",
        auto_adjust=False,
        actions=True,
        warnings=[],
    )

    success = make_fetch_result("MSFT", make_provider_frame("MSFT"))
    empty = FetchResult(
        symbol="EMPTY",
        data=make_provider_frame("EMPTY").iloc[0:0].copy(),
        metadata=FetchMetadata(
            **{
                **base_metadata,
                "requested_symbol": "EMPTY",
                "resolved_symbol": "EMPTY",
                "row_count": 0,
                "backend_used": "download_many",
                "fetch_state": FetchState.EMPTY,
                "failure_kind": "empty_dataset",
                "attempts": [
                    DownloadAttempt(
                        attempt_number=1,
                        backend="download_many",
                        interval="1d",
                        start="2024-01-01T00:00:00",
                        end="2024-01-02T00:00:00",
                        duration_seconds=0.1,
                        success=False,
                        rows=0,
                        error="empty dataframe",
                    )
                ],
            },
        ),
    )
    failed = FetchResult(
        symbol="BAD",
        data=make_provider_frame("BAD").iloc[0:0].copy(),
        metadata=FetchMetadata(
            **{
                **base_metadata,
                "requested_symbol": "BAD",
                "resolved_symbol": "BAD",
                "row_count": 0,
                "fetch_state": FetchState.FAILED,
                "failure_kind": "batch_retrieval_failed",
                "warnings": ["provider returned a structured failure"],
                "attempts": [
                    DownloadAttempt(
                        attempt_number=1,
                        backend="n/a",
                        interval="1d",
                        start="2024-01-01T00:00:00",
                        end="2024-01-02T00:00:00",
                        duration_seconds=0.1,
                        success=False,
                        rows=0,
                        error="boom",
                    )
                ],
            },
        ),
    )

    assert AcquisitionService._classify_fetch_result(success) == "success"
    assert AcquisitionService._classify_fetch_result(empty) == "empty"
    assert AcquisitionService._classify_fetch_result(failed) == "failed"
    assert AcquisitionService._is_failure_result(failed) is True
    assert AcquisitionService._is_failure_result(empty) is False


def test_acquisition_service_does_not_depend_on_failure_warning_strings(tmp_path):
    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_history_bundle(self, **kwargs):
            return {
                "MSFT": make_fetch_result("MSFT", make_provider_frame("MSFT")),
                "BAD": FetchResult(
                    symbol="BAD",
                    data=make_provider_frame("BAD").iloc[0:0].copy(),
                    metadata=FetchMetadata(
                        provider_name="TestProvider",
                        provider_version="0.0.1",
                        source="unit-test",
                        request_id="req-bad",
                        requested_symbol="BAD",
                        resolved_symbol="BAD",
                        requested_interval="1d",
                        resolved_interval="1d",
                        requested_start="2024-01-01T00:00:00",
                        requested_end="2024-01-02T00:00:00",
                        effective_start="2024-01-01T00:00:00",
                        effective_end="2024-01-02T00:00:00",
                        actual_start=None,
                        actual_end=None,
                        extracted_at_utc="2026-03-31T00:00:00+00:00",
                        auto_adjust=False,
                        actions=True,
                        row_count=0,
                        fetch_state=FetchState.FAILED,
                        failure_kind="batch_retrieval_failed",
                        warnings=["human text changed and no longer contains the legacy marker"],
                        attempts=[
                            DownloadAttempt(
                                attempt_number=1,
                                backend="n/a",
                                interval="1d",
                                start="2024-01-01T00:00:00",
                                end="2024-01-02T00:00:00",
                                duration_seconds=0.1,
                                success=False,
                                rows=0,
                                error="boom",
                            )
                        ],
                    ),
                ),
            }

    request = DatasetRequest(
        tickers=["MSFT", "BAD"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
    )

    result = AcquisitionService(provider_factory=_Provider).fetch_many(
        ["MSFT", "BAD"],
        request,
        auto_adjust=False,
        actions=True,
    )

    assert list(result) == ["MSFT"]


def test_fetch_many_matches_single_ticker_results_for_compatible_daily_batch(monkeypatch, tmp_path):
    def _raw_history(symbol: str) -> pd.DataFrame:
        frame = make_provider_frame(symbol)
        raw = frame.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj_close": "Adj Close",
                "volume": "Volume",
                "dividends": "Dividends",
                "stock_splits": "Stock Splits",
            }
        )
        return raw.set_index("Date")

    raw_frames = {
        "MSFT": _raw_history("MSFT"),
        "AAPL": _raw_history("AAPL"),
    }
    download_calls = []

    class _FailingTicker:
        def history(self, **kwargs):
            raise RuntimeError("force download backend")

    def _fake_download(tickers, **kwargs):
        download_calls.append(tickers)
        if isinstance(tickers, str) and " " in tickers:
            symbols = tickers.split()
            return pd.concat({symbol: raw_frames[symbol] for symbol in symbols}, axis=1)
        return raw_frames[str(tickers).strip().upper()]

    monkeypatch.setattr("providers.yfinance_provider.yf.Ticker", lambda symbol: _FailingTicker())
    monkeypatch.setattr("providers.yfinance_provider.yf.download", _fake_download)

    provider = YFinanceProvider(cache_dir=tmp_path / "cache", retries=1, min_delay=0)
    single_msft = provider.get_history_bundle("MSFT", start="2024-01-01", end="2024-01-06", interval="1d")
    single_aapl = provider.get_history_bundle("AAPL", start="2024-01-01", end="2024-01-06", interval="1d")
    many = provider.get_history_bundle(["MSFT", "AAPL"], start="2024-01-01", end="2024-01-06", interval="1d")

    assert list(many) == ["MSFT", "AAPL"]
    assert many["MSFT"].metadata.backend_used == "download_many"
    assert many["AAPL"].metadata.backend_used == "download_many"
    assert_frame_equal(many["MSFT"].data, single_msft.data)
    assert_frame_equal(many["AAPL"].data, single_aapl.data)
    assert "MSFT AAPL" in download_calls


def test_download_backend_propagates_timeout_for_single_symbol_fallback(monkeypatch, tmp_path):
    download_calls = []

    class _FailingTicker:
        def history(self, **kwargs):
            raise RuntimeError("force download backend")

    raw = make_provider_frame("MSFT").rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
            "dividends": "Dividends",
            "stock_splits": "Stock Splits",
        }
    ).set_index("Date")

    def _fake_download(tickers, **kwargs):
        download_calls.append({"tickers": tickers, **kwargs})
        return raw

    monkeypatch.setattr("providers.yfinance_provider.yf.Ticker", lambda symbol: _FailingTicker())
    monkeypatch.setattr("providers.yfinance_provider.yf.download", _fake_download)

    provider = YFinanceProvider(cache_dir=tmp_path / "cache", retries=1, min_delay=0, timeout=7.5)
    result = provider.get_history_bundle("MSFT", start="2024-01-01", end="2024-01-06", interval="1d")

    assert result.metadata.backend_used == "download"
    assert download_calls
    assert download_calls[0]["timeout"] == 7.5


def test_download_many_propagates_timeout_for_grouped_requests(monkeypatch, tmp_path):
    raw_frames = {}
    for symbol in ("MSFT", "AAPL"):
        raw_frames[symbol] = make_provider_frame(symbol).rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj_close": "Adj Close",
                "volume": "Volume",
                "dividends": "Dividends",
                "stock_splits": "Stock Splits",
            }
        ).set_index("Date")

    download_calls = []

    def _fake_download(tickers, **kwargs):
        download_calls.append({"tickers": tickers, **kwargs})
        symbols = str(tickers).split()
        return pd.concat({symbol: raw_frames[symbol] for symbol in symbols}, axis=1)

    monkeypatch.setattr("providers.yfinance_provider.yf.download", _fake_download)

    provider = YFinanceProvider(cache_dir=tmp_path / "cache", retries=1, min_delay=0, timeout=9.25)
    result = provider.get_history_bundle(["MSFT", "AAPL"], start="2024-01-01", end="2024-01-06", interval="1d")

    assert list(result) == ["MSFT", "AAPL"]
    assert all(item.metadata.backend_used == "download_many" for item in result.values())
    assert download_calls
    assert download_calls[0]["timeout"] == 9.25


def test_download_timeout_is_preserved_across_retries(monkeypatch, tmp_path):
    download_timeouts = []

    class _FailingTicker:
        def history(self, **kwargs):
            raise RuntimeError("force download backend")

    raw = make_provider_frame("MSFT").rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
            "dividends": "Dividends",
            "stock_splits": "Stock Splits",
        }
    ).set_index("Date")

    def _flaky_download(tickers, **kwargs):
        download_timeouts.append(kwargs["timeout"])
        if len(download_timeouts) == 1:
            raise RuntimeError("transient download failure")
        return raw

    monkeypatch.setattr("providers.yfinance_provider.yf.Ticker", lambda symbol: _FailingTicker())
    monkeypatch.setattr("providers.yfinance_provider.yf.download", _flaky_download)

    provider = YFinanceProvider(cache_dir=tmp_path / "cache", retries=2, min_delay=0, timeout=4.5)
    result = provider.get_history_bundle("MSFT", start="2024-01-01", end="2024-01-06", interval="1d")

    assert result.metadata.backend_used == "download"
    assert download_timeouts == [4.5, 4.5]


def test_yfinance_provider_keeps_legacy_dict_init_shim(tmp_path):
    provider = YFinanceProvider(
        {
            "cache_dir": tmp_path / "cache",
            "retries": 2,
            "timeout": 3.5,
            "metadata_timeout": 1.5,
            "min_delay": 0.0,
            "cache_mode": "process",
        }
    )

    info = provider.get_provider_info()

    assert info["cache_mode"] == "process"
    assert info["cache_enabled"] is True
    assert str((tmp_path / "cache").resolve()) in str(info["cache_dir"])
    assert info["retries"] == 2
    assert info["timeout"] == 3.5
    assert info["metadata_timeout"] == 1.5
