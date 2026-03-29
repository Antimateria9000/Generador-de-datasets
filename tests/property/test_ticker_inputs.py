from __future__ import annotations

import pytest

from dataset_core.contracts import dedupe_preserve_order, parse_tickers_text


@pytest.mark.parametrize(
    ("raw_items", "expected"),
    [
        (["msft", "aapl", "msft", "nvda"], ["MSFT", "AAPL", "NVDA"]),
        (["", " ", " ibm ", "IBM", "ibm"], ["IBM"]),
        (["brk.b", "BRK.B", "tsla", "tsla", "orcl"], ["BRK.B", "TSLA", "ORCL"]),
        (["áccent", "áccent", "ETF-1", "etf-1"], ["ÁCCENT", "ETF-1"]),
        (["meta", "goog", "amzn", "meta", "goog", "meta"], ["META", "GOOG", "AMZN"]),
    ],
)
def test_dedupe_preserve_order_never_reorders_unique_items(raw_items, expected):
    result = dedupe_preserve_order(raw_items)
    assert result == expected


@pytest.mark.parametrize(
    ("symbols", "raw"),
    [
        (["MSFT", "AAPL", "NVDA"], "MSFT, \nAAPL, NVDA, MSFT, AAPL"),
        (["BRK.B", "TSLA", "ORCL"], "brk.b tsla orcl brk.b"),
        (["ETF-1", "SPY", "QQQ"], "ETF-1; SPY\nQQQ ETF-1"),
        (["SAN.MC", "IBE.MC"], "san.mc, ibe.mc, SAN.MC"),
    ],
)
def test_parse_tickers_text_deduplicates_symbols_preserving_order(symbols, raw):
    parsed = parse_tickers_text(raw)

    assert parsed == list(dict.fromkeys([item.strip().upper() for item in symbols if item.strip()]))
