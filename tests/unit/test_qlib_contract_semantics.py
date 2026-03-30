from __future__ import annotations

import pandas as pd

from dataset_core.qlib_contract import validate_qlib_frame
from dataset_core.sanitization_qlib import QlibSanitizer
from tests.fixtures.sample_data import make_double_adjusted_qlib_frame, make_nvda_like_split_frame


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
