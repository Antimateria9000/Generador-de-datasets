from __future__ import annotations

import pandas as pd

from dataset_core.qlib_contract import OHLC_GEOMETRY_ABS_TOL, validate_qlib_frame
from dataset_core.sanitization_qlib import QlibSanitizer
from tests.fixtures.sample_data import make_double_adjusted_qlib_frame, make_nvda_like_split_frame


def _minimal_qlib_frame(*, open_value: float, high_value: float, low_value: float, close_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-03-23")],
            "open": [open_value],
            "high": [high_value],
            "low": [low_value],
            "close": [close_value],
            "volume": [1_000_000.0],
            "factor": [0.7686071282708494],
        }
    )


def test_validate_qlib_frame_rejects_double_adjusted_pattern():
    result = validate_qlib_frame(make_double_adjusted_qlib_frame())

    assert result.compatible is False
    assert any("double adjustment" in reason.lower() for reason in result.reasons)
    assert any(check["name"] == "factor_step_semantics" and not check["passed"] for check in result.checks)


def test_validate_qlib_frame_accepts_clean_nvda_like_contract():
    qlib_frame = QlibSanitizer().sanitize(make_nvda_like_split_frame()).frame
    result = validate_qlib_frame(qlib_frame)

    assert result.compatible is True
    assert result.reasons == []
    assert result.metrics["factor_step_events"] >= 0


def test_validate_qlib_frame_is_index_agnostic():
    qlib_frame = QlibSanitizer().sanitize(make_nvda_like_split_frame()).frame

    range_result = validate_qlib_frame(qlib_frame.reset_index(drop=True))
    sparse_result = validate_qlib_frame(qlib_frame.set_index(pd.Index([10, 20, 30, 40])))
    datetime_result = validate_qlib_frame(qlib_frame.set_index(pd.date_range("2024-06-07", periods=len(qlib_frame), freq="D")))

    assert range_result.compatible is True
    assert sparse_result.compatible == range_result.compatible
    assert sparse_result.reasons == range_result.reasons
    assert datetime_result.compatible == range_result.compatible
    assert datetime_result.reasons == range_result.reasons


def test_validate_qlib_frame_accepts_bbva_like_geometry_with_floating_point_noise():
    frame = _minimal_qlib_frame(
        open_value=4.11665973797062,
        high_value=4.136643540897683,
        low_value=3.9959883689880376,
        close_value=3.995988368988037,
    )

    result = validate_qlib_frame(frame)

    assert result.compatible is True
    assert not any("geometry" in reason.lower() for reason in result.reasons)
    assert result.metrics["bad_geometry_count"] == 0


def test_validate_qlib_frame_rejects_really_invalid_geometry():
    frame = _minimal_qlib_frame(
        open_value=10.0,
        high_value=9.0,
        low_value=8.0,
        close_value=9.5,
    )

    result = validate_qlib_frame(frame)

    assert result.compatible is False
    assert any("geometry" in reason.lower() for reason in result.reasons)
    assert result.metrics["bad_geometry_count"] == 1
    assert result.metrics["bad_geometry_examples"][0]["date"] == "2022-03-23"


def test_validate_qlib_frame_accepts_low_delta_within_geometry_tolerance():
    frame = _minimal_qlib_frame(
        open_value=4.0,
        high_value=4.2,
        low_value=3.5 + 1e-16,
        close_value=3.5,
    )

    result = validate_qlib_frame(frame)

    assert result.compatible is True
    assert result.metrics["bad_geometry_count"] == 0


def test_validate_qlib_frame_rejects_low_delta_above_geometry_tolerance():
    frame = _minimal_qlib_frame(
        open_value=4.0,
        high_value=4.2,
        low_value=3.5 + max(OHLC_GEOMETRY_ABS_TOL * 10, 1e-8),
        close_value=3.5,
    )

    result = validate_qlib_frame(frame)

    assert result.compatible is False
    assert any("geometry" in reason.lower() for reason in result.reasons)
    assert result.metrics["bad_geometry_count"] == 1
