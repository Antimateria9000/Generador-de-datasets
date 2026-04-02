from __future__ import annotations

import pandas as pd

from dataset_core.validation_internal import InternalDQService
from tests.fixtures.sample_data import make_provider_frame


def test_internal_validation_no_longer_emits_exchange_calendar_timezone_bug():
    frame = make_provider_frame("MSFT", periods=40)
    frame["date"] = pd.bdate_range("2024-01-02", periods=len(frame))

    report = InternalDQService().run(
        frame=frame,
        symbol="MSFT",
        interval="1d",
        actions=True,
        dq_mode="report",
        dq_context={
            "market": "XNAS",
            "asset_type": "equity",
            "quote_type": "EQUITY",
            "dq_profile": "equity",
            "is_24_7": False,
            "volume_expected": True,
            "corporate_actions_expected": True,
            "calendar_validation_supported": True,
        },
    )

    warnings = report["report"]["warnings"]
    assert not any(item["check"] == "calendar_validation_error" for item in warnings)


def test_internal_validation_treats_high_volume_spikes_as_warning_only():
    frame = make_provider_frame("BBVA.MC", periods=255)
    frame["date"] = pd.bdate_range("2025-04-02", periods=len(frame))
    frame["volume"] = pd.Series([8_000_000.0 + idx * 12_000.0 for idx in range(len(frame))], dtype="float64")
    frame.loc[[140, 157, 158, 167, 254], "volume"] = [
        33_423_900.0,
        78_356_240.0,
        89_936_640.0,
        36_597_990.0,
        70_754_430.0,
    ]

    report = InternalDQService().run(
        frame=frame,
        symbol="BBVA.MC",
        interval="1d",
        actions=True,
        dq_mode="report",
        dq_context={
            "market": "XMAD",
            "asset_type": "equity",
            "quote_type": "EQUITY",
            "dq_profile": "equity",
            "is_24_7": False,
            "volume_expected": True,
            "corporate_actions_expected": True,
            "calendar_validation_supported": True,
        },
    )

    findings = report["report"]["findings"]
    volume_outlier_findings = [item for item in findings if item["check"] == "volume_zscore_robust"]

    assert report["status"] == "passed_with_warnings"
    assert volume_outlier_findings
    assert volume_outlier_findings[0]["severity"] == "warning"
    assert volume_outlier_findings[0]["details"]["high_spike_count"] == 5
    assert volume_outlier_findings[0]["details"]["low_spike_count"] == 0
