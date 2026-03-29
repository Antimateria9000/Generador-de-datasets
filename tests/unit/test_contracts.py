from __future__ import annotations

from pathlib import Path

import pytest

from dataset_core.contracts import (
    DatasetRequest,
    RequestContractError,
    TemporalRange,
    load_tickers_from_file,
    parse_extras,
    parse_tickers_text,
    resolve_ticker_inputs,
)


def test_parse_tickers_text_deduplicates_preserving_order():
    assert parse_tickers_text("MSFT, AAPL\nMSFT   NVDA") == ["MSFT", "AAPL", "NVDA"]


def test_load_tickers_from_file_supports_txt_fixture():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "universe.txt"
    assert load_tickers_from_file(fixture) == ["MSFT", "AAPL", "NVDA"]


def test_resolve_ticker_inputs_rejects_ambiguity():
    with pytest.raises(RequestContractError):
        resolve_ticker_inputs(ticker="MSFT", tickers="AAPL")


def test_parse_extras_normalizes_and_validates():
    assert parse_extras("adj_close, FACTOR, dividends") == ["adj_close", "factor", "dividends"]
    with pytest.raises(RequestContractError):
        parse_extras("foo")


def test_dataset_request_rejects_batch_filename_override():
    with pytest.raises(RequestContractError):
        DatasetRequest(
            tickers=["MSFT", "AAPL"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            filename_override="custom.csv",
        )
