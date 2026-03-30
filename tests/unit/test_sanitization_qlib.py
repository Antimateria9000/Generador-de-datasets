from __future__ import annotations

import pytest

from dataset_core.sanitization_qlib import QlibSanitizationError, QlibSanitizer
from tests.fixtures.sample_data import make_nvda_like_split_frame, make_raw_split_frame


def test_qlib_sanitizer_emits_contract_ready_frame_from_adj_close_ratio():
    result = QlibSanitizer().sanitize(make_nvda_like_split_frame())

    assert list(result.frame.columns) == ["date", "open", "high", "low", "close", "volume", "factor"]
    assert result.contract.compatible is True
    assert result.factor_policy == "qlib_adjusted_from_adj_close_ratio"
    assert result.factor_source == "adj_close_ratio"
    assert result.technical_report["qlib_compatible"] is True


def test_qlib_sanitizer_can_use_controlled_split_fallback():
    result = QlibSanitizer().sanitize(make_raw_split_frame())

    assert result.contract.compatible is True
    assert result.factor_source == "stock_splits_fallback"


def test_qlib_sanitizer_rejects_adj_close_output_bypass():
    with pytest.raises(QlibSanitizationError, match="does not allow adj_close"):
        QlibSanitizer().sanitize(make_nvda_like_split_frame(), include_adj_close=True)
