from __future__ import annotations

import time

from providers.market_context import ContextResolver, resolve_instrument_context


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


def test_resolve_instrument_context_respects_metadata_timeout(monkeypatch):
    class _SlowTicker:
        @property
        def history_metadata(self):
            time.sleep(0.05)
            return {}

        @property
        def fast_info(self):
            time.sleep(0.05)
            return {}

        @property
        def info(self):
            time.sleep(0.05)
            return {}

    monkeypatch.setattr("providers.market_context.yf.Ticker", lambda symbol: _SlowTicker())

    context = resolve_instrument_context("MSFT", listing_preference="exact_symbol", metadata_timeout=0.01)

    assert context.requested_symbol == "MSFT"
    assert any("timeout" in warning.lower() for warning in context.warnings)
    assert any(warning.get("code") == "metadata_timeout" for warning in context.structured_warnings)
    assert context.resolver_metrics["metadata_timeout"] == 0.01


def test_context_resolver_reuses_cached_snapshots_within_same_run(monkeypatch):
    calls = []

    def _ticker(symbol):
        calls.append(symbol)
        return _FakeTicker(
            history_metadata={"exchangeTimezoneName": "America/New_York"},
            fast_info={"currency": "USD"},
            info={
                "quoteType": "EQUITY",
                "exchange": "NYQ",
                "fullExchangeName": "NYSE",
                "symbol": symbol,
                "longName": "Microsoft Corporation",
            },
        )

    monkeypatch.setattr("providers.market_context.yf.Ticker", _ticker)

    resolver = ContextResolver(candidate_limit=2)
    first = resolve_instrument_context("MSFT", listing_preference="exact_symbol", resolver=resolver)
    second = resolve_instrument_context("MSFT", listing_preference="exact_symbol", resolver=resolver)

    assert first.preferred_symbol == "MSFT"
    assert second.preferred_symbol == "MSFT"
    assert calls == ["MSFT"]
    assert resolver.metrics["cache_misses"] == 1
    assert resolver.metrics["cache_hits"] >= 1
    assert second.resolution_trace[0]["cache_hit"] is True
    assert all(item.get("from_cache") is True for item in second.resolution_trace[0]["query_trace"])
