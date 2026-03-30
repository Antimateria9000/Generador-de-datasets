from __future__ import annotations

from providers.market_context import resolve_instrument_context


class _FakeTicker:
    def __init__(self, history_metadata=None, fast_info=None, info=None):
        self.history_metadata = history_metadata or {}
        self.fast_info = fast_info or {}
        self.info = info or {}


def test_resolve_instrument_context_uses_offline_yahoo_metadata_snapshot(monkeypatch):
    monkeypatch.setattr(
        "providers.market_context.yf.Ticker",
        lambda symbol: _FakeTicker(
            history_metadata={"exchangeTimezoneName": "America/New_York"},
            fast_info={"currency": "USD"},
            info={
                "quoteType": "EQUITY",
                "exchange": "NYQ",
                "fullExchangeName": "NYSE",
                "symbol": symbol,
                "longName": "Microsoft Corporation",
            },
        ),
    )

    context = resolve_instrument_context("MSFT", listing_preference="exact_symbol")

    assert context.requested_symbol == "MSFT"
    assert context.preferred_symbol == "MSFT"
    assert context.market == "XNYS"
    assert context.region == "USA"
    assert context.asset_type == "equity"
    assert context.warnings == []


def test_resolve_instrument_context_reports_missing_metadata_without_network(monkeypatch):
    monkeypatch.setattr(
        "providers.market_context.yf.Ticker",
        lambda symbol: _FakeTicker(),
    )

    context = resolve_instrument_context("MSFT", listing_preference="exact_symbol")

    assert context.requested_symbol == "MSFT"
    assert context.asset_type == "equity"
    assert any("metadata de Yahoo" in warning for warning in context.warnings)
    assert any("quoteType" in warning for warning in context.warnings)
