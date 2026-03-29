from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from dataset_core.contracts import dedupe_preserve_order, parse_tickers_text


@given(st.lists(st.text(min_size=0, max_size=8), min_size=1, max_size=12))
def test_dedupe_preserve_order_never_reorders_unique_items(raw_items):
    result = dedupe_preserve_order(raw_items)
    assert result == list(dict.fromkeys(result))


@given(st.lists(st.from_regex(r"[A-Za-z0-9.\-_]{1,8}", fullmatch=True), min_size=1, max_size=8))
def test_parse_tickers_text_deduplicates_symbols_preserving_order(symbols):
    raw = ", \n".join(symbols + symbols[:2])
    parsed = parse_tickers_text(raw)

    assert parsed == list(dict.fromkeys([item.strip().upper() for item in symbols if item.strip()]))
