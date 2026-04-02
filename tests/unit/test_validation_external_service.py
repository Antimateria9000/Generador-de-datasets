from __future__ import annotations

import pandas as pd

from dataset_core.external_sources.base import attach_source_metadata
from dataset_core.external_sources.base import ExternalSourceCoverageError
from dataset_core.validation_external import ExternalValidationService
from tests.fixtures.sample_data import make_provider_frame


class _InlinePriceAdapter:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def name(self):
        return "inline_price"

    def fetch_reference(self, symbol, start, end):
        return self.frame.copy()


class _CoverageGapAdapter:
    def name(self):
        return "coverage_gap"

    def fetch_reference(self, symbol, start, end):
        raise ExternalSourceCoverageError("symbol is not covered by the configured external source")


class _InlineEventAdapter:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def name(self):
        return "inline_events"

    def validation_scope(self):
        return "event"

    def fetch_events(self, symbol, start, end):
        return self.frame.copy()


def test_external_validation_service_passes_for_identical_price_data():
    dataset = make_provider_frame("MSFT", periods=3)
    report = ExternalValidationService(adapters=[_InlinePriceAdapter(dataset)]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "passed"
    assert report["coverage_status"] == "full"
    assert report["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["status"] == "passed"
    assert report["adapter_reports"][0]["coverage_status"] == "full"
    assert report["adapter_reports"][0]["comparison_status"] == "passed"


def test_external_validation_service_fails_when_reference_has_missing_dates():
    dataset = make_provider_frame("MSFT", periods=5)
    reference = dataset.iloc[:2].copy()
    report = ExternalValidationService(adapters=[_InlinePriceAdapter(reference)]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "failed"
    assert report["comparison_status"] == "failed"
    assert report["adapter_reports"][0]["gap_count"] > 0


def test_external_validation_service_fails_for_large_relative_difference():
    dataset = make_provider_frame("MSFT", periods=3)
    reference = dataset.copy()
    reference["close"] = reference["close"] * 0.8
    report = ExternalValidationService(adapters=[_InlinePriceAdapter(reference)]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "failed"
    assert report["adapter_reports"][0]["max_relative_diff"] > 0


def test_external_validation_service_reports_coverage_gaps_as_not_validated():
    dataset = make_provider_frame("MSFT", periods=3)
    report = ExternalValidationService(adapters=[_CoverageGapAdapter()]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "not_validated"
    assert report["coverage_status"] == "partial"
    assert report["comparison_status"] == "not_validated"
    assert report["adapter_reports"][0]["error_kind"] == "coverage_insufficient"


def test_external_validation_service_compares_sparse_events_without_calendar_penalty():
    dataset = make_provider_frame("MSFT", periods=3)
    dataset.loc[1, "stock_splits"] = 2.0
    events = pd.DataFrame(
        [
            {
                "date": dataset.loc[1, "date"],
                "stock_splits": 2.0,
            }
        ]
    )

    report = ExternalValidationService(event_adapters=[_InlineEventAdapter(events)]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "passed"
    assert report["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["checked_event_count"] == 1


def test_external_validation_service_accepts_declared_partial_recent_price_coverage():
    dataset = make_provider_frame("MSFT", periods=260)
    reference = attach_source_metadata(
        dataset.iloc[-120:].copy(),
        {
            "provider": "eodhd",
            "partial_coverage": True,
            "effective_start": str(dataset.iloc[-120]["date"]),
            "effective_end": str(dataset.iloc[-1]["date"] + pd.Timedelta(days=1)),
        },
    )
    reference["volume"] = reference["volume"] * 1.05

    report = ExternalValidationService(adapters=[_InlinePriceAdapter(reference)]).validate(
        frame=dataset,
        symbol="MSFT",
        start="2024-01-01T00:00:00",
        end="2025-12-31T00:00:00",
    ).to_dict()

    assert report["status"] == "passed_partial"
    assert report["coverage_status"] == "partial"
    assert report["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["status"] == "passed_partial"
    assert report["adapter_reports"][0]["coverage_status"] == "partial"
    assert report["adapter_reports"][0]["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["coverage"]["uncovered_prefix_count"] > 0
    assert "coverage_limited" in report["adapter_reports"][0]["partial_validation_kinds"]


def test_external_validation_service_fails_when_partial_provider_overlap_has_blocking_mismatch():
    dataset = make_provider_frame("MSFT", periods=260)
    reference = attach_source_metadata(
        dataset.iloc[-120:].copy(),
        {
            "provider": "eodhd",
            "partial_coverage": True,
            "effective_start": str(dataset.iloc[-120]["date"]),
            "effective_end": str(dataset.iloc[-1]["date"] + pd.Timedelta(days=1)),
        },
    )
    reference["close"] = reference["close"] * 0.8

    report = ExternalValidationService(adapters=[_InlinePriceAdapter(reference)]).validate(
        frame=dataset,
        symbol="MSFT",
        start="2024-01-01T00:00:00",
        end="2025-12-31T00:00:00",
    ).to_dict()

    assert report["status"] == "failed"
    assert report["coverage_status"] == "partial"
    assert report["comparison_status"] == "failed"
    assert report["adapter_reports"][0]["coverage_status"] == "partial"
    assert report["adapter_reports"][0]["comparison_status"] == "failed"


def test_external_validation_service_fails_when_partial_provider_overlap_has_large_internal_gaps():
    dataset = make_provider_frame("MSFT", periods=260)
    reference = dataset.iloc[-120:].copy().reset_index(drop=True)
    reference = reference.iloc[::3].reset_index(drop=True)
    reference = attach_source_metadata(
        reference,
        {
            "provider": "eodhd",
            "partial_coverage": True,
            "effective_start": str(dataset.iloc[-120]["date"]),
            "effective_end": str(dataset.iloc[-1]["date"] + pd.Timedelta(days=1)),
        },
    )

    report = ExternalValidationService(adapters=[_InlinePriceAdapter(reference)]).validate(
        frame=dataset,
        symbol="MSFT",
        start="2024-01-01T00:00:00",
        end="2025-12-31T00:00:00",
    ).to_dict()

    assert report["status"] == "failed"
    assert report["coverage_status"] == "partial"
    assert report["comparison_status"] == "failed"
    assert report["adapter_reports"][0]["gap_ratio"] > 0.10


def test_external_validation_service_downgrades_systematic_dividend_scale_mismatch_to_partial():
    dataset = make_provider_frame("BBVA.MC", periods=10)
    dataset.loc[[2, 5, 8], "dividends"] = [0.80, 0.50, 0.30]
    events = pd.DataFrame(
        [
            {"date": dataset.loc[2, "date"], "dividends": 0.648},
            {"date": dataset.loc[5, "date"], "dividends": 0.405},
            {"date": dataset.loc[8, "date"], "dividends": 0.243},
        ]
    )

    report = ExternalValidationService(event_adapters=[_InlineEventAdapter(events)]).validate(
        frame=dataset,
        symbol="BBVA.MC",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "passed_partial"
    assert report["coverage_status"] == "partial"
    assert report["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["status"] == "passed_partial"
    assert report["adapter_reports"][0]["coverage_status"] == "full"
    assert report["adapter_reports"][0]["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["systematic_dividend_scale_factor"] == 0.81


def test_external_validation_service_accepts_two_event_dividend_scale_mismatch_for_short_windows():
    dataset = make_provider_frame("BBVA.MC", periods=8)
    dataset.loc[[2, 6], "dividends"] = [0.41, 0.32]
    events = pd.DataFrame(
        [
            {"date": dataset.loc[2, "date"], "dividends": 0.3321},
            {"date": dataset.loc[6, "date"], "dividends": 0.2592},
        ]
    )

    report = ExternalValidationService(event_adapters=[_InlineEventAdapter(events)]).validate(
        frame=dataset,
        symbol="BBVA.MC",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "passed_partial"
    assert report["comparison_status"] == "passed"
    assert report["adapter_reports"][0]["status"] == "passed_partial"
    assert report["adapter_reports"][0]["systematic_dividend_scale_factor"] == 0.81
