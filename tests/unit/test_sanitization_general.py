from __future__ import annotations

import pandas as pd

from dataset_core.sanitization_general import GeneralSanitizer
from tests.fixtures.sample_data import make_provider_frame


def test_general_sanitizer_removes_duplicates_and_impossible_rows():
    frame = make_provider_frame(periods=4)
    duplicate_row = frame.iloc[[1]].copy()
    impossible_row = frame.iloc[[2]].copy()
    impossible_row.loc[:, "high"] = impossible_row["low"] - 1
    impossible_row.loc[:, "date"] = pd.Timestamp("2024-01-10")
    duplicate_row.loc[:, "date"] = frame.loc[1, "date"]

    dirty = pd.concat([frame, duplicate_row, impossible_row], ignore_index=True)
    result = GeneralSanitizer().sanitize(dirty, requested_extras=["adj_close"])

    assert result.frame["date"].is_monotonic_increasing
    assert result.frame["date"].duplicated().sum() == 0
    assert len(result.frame) == 4
    assert any("duplicate" in warning.lower() for warning in result.warnings)
    assert any("impossible" in warning.lower() for warning in result.warnings)


def test_general_sanitizer_marks_missing_adj_close_without_reconstructing_from_close():
    frame = make_provider_frame(periods=3).drop(columns=["adj_close"])
    result = GeneralSanitizer().sanitize(frame, requested_extras=["adj_close"])

    assert "adj_close" in result.frame.columns
    assert result.frame["adj_close"].isna().all()
    assert result.column_provenance["adj_close"]["state"] == "provider_missing"
    assert result.column_provenance["adj_close"]["synthetic"] is False
    assert result.column_provenance["adj_close"]["materialized_empty_column"] is True
    assert any("provider did not supply usable adjusted-close values" in warning for warning in result.warnings)


def test_general_sanitizer_removes_rows_with_all_zero_ohlc():
    frame = make_provider_frame(periods=3)
    frame.loc[1, ["open", "high", "low", "close"]] = 0.0

    result = GeneralSanitizer().sanitize(frame, requested_extras=[])

    assert len(result.frame) == 2
    assert not ((result.frame[["open", "high", "low", "close"]] <= 0).any(axis=1)).any()
    assert any("non-positive ohlc prices" in warning.lower() for warning in result.warnings)


def test_general_sanitizer_removes_rows_with_single_zero_price_but_keeps_zero_volume():
    frame = make_provider_frame(periods=3)
    frame.loc[0, "open"] = 0.0
    frame.loc[1, "volume"] = 0.0

    result = GeneralSanitizer().sanitize(frame, requested_extras=[])

    assert len(result.frame) == 2
    assert (result.frame["volume"] == 0.0).any()
    assert not (result.frame["open"] <= 0).any()


def test_general_sanitizer_keeps_tiny_positive_prices():
    frame = make_provider_frame(periods=2)
    frame = frame.astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64"})
    frame.loc[0, ["open", "high", "low", "close"]] = [1e-9, 2e-9, 5e-10, 1.5e-9]

    result = GeneralSanitizer().sanitize(frame, requested_extras=[])

    assert len(result.frame) == 2
    assert (result.frame.loc[0, ["open", "high", "low", "close"]] > 0).all()
