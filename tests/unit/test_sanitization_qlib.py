from __future__ import annotations

from dataset_core.sanitization_qlib import QlibSanitizer
from tests.fixtures.sample_data import make_split_frame


def test_qlib_sanitizer_emits_contract_ready_frame():
    result = QlibSanitizer().sanitize(make_split_frame())

    assert list(result.frame.columns) == ["date", "open", "high", "low", "close", "volume", "factor"]
    assert result.contract.compatible is True
    assert result.factor_policy == "qlib_split_adjusted_from_raw_ohlcv"
    assert result.technical_report["qlib_compatible"] is True
