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


def test_general_sanitizer_backfills_adj_close_when_requested():
    frame = make_provider_frame(periods=3).drop(columns=["adj_close"])
    result = GeneralSanitizer().sanitize(frame, requested_extras=["adj_close"])

    assert "adj_close" in result.frame.columns
    assert result.frame["adj_close"].tolist() == result.frame["close"].tolist()
