from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from dataset_core.acquisition import AcquisitionService
from dataset_core.contracts import DatasetRequest, TemporalRange
from providers.yfinance_provider import YFinanceProvider, _clean_df
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
