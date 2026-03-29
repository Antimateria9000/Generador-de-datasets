from __future__ import annotations

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
