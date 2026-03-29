from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.naming import build_csv_filename, sanitize_symbol_for_csv


@given(st.text(min_size=1, max_size=20).filter(lambda value: bool(str(value).strip())))
def test_sanitize_symbol_for_csv_only_keeps_safe_characters(raw_symbol: str):
    sanitized = sanitize_symbol_for_csv(raw_symbol)
    assert sanitized
    assert all(char.isalnum() or char in "._-" for char in sanitized)


def test_qlib_naming_is_ticker_csv():
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        mode="qlib",
    )
    assert build_csv_filename("MSFT", request) == "MSFT.csv"


def test_non_qlib_filename_keeps_range_and_interval():
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        mode="base",
        interval="1d",
    )
    assert build_csv_filename("MSFT", request).startswith("MSFT_1d_5y")
