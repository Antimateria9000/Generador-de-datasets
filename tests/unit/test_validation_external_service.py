from __future__ import annotations

import pandas as pd

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
    assert report["adapter_reports"][0]["status"] == "passed"


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
    assert report["adapter_reports"][0]["checked_event_count"] == 1
