from __future__ import annotations

import pytest

from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.naming import (
    build_csv_filename,
    build_csv_output_path,
    build_run_id,
    sanitize_symbol_for_csv,
    summarize_tickers_for_run_id,
)


@pytest.mark.parametrize(
    "raw_symbol",
    [
        "MSFT",
        " msft ",
        "BRK.B",
        "eur/usd",
        "te sla",
        "áccent",
        "株式-123",
        "tick\"er'",
        "NVDA🚀",
        "a" * 20,
    ],
)
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


def test_filename_override_is_treated_as_safe_basename_only(tmp_path):
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        filename_override="../nested/..\\escape?.csv",
    )

    target = build_csv_output_path(tmp_path, "MSFT", request)

    assert request.filename_override == "escape_.csv"
    assert target.parent == tmp_path.resolve()
    assert target.name == "escape_.csv"


def test_run_id_is_human_readable_for_long_ticker_lists(monkeypatch):
    monkeypatch.setattr("dataset_core.naming.utc_now_token", lambda: "20260329_210000")
    monkeypatch.setattr("dataset_core.naming.utc_now_token_microseconds", lambda: "20260329_210000_123456")
    request = DatasetRequest(
        tickers=["MICROSOFT", "NVIDIA", "ALPHABET", "AMAZON", "META"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        mode="extended",
        interval="1d",
    )

    run_id = build_run_id(request)
    short_hash = run_id.rsplit("_", 1)[-1]

    assert run_id.startswith("20260329_210000_extended_1d_")
    assert len(short_hash) == 6
    assert len(run_id) < 80
    assert "plus" in summarize_tickers_for_run_id(request.tickers, max_length=18)
