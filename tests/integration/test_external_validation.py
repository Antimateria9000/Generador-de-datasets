from __future__ import annotations

import json
import pandas as pd

from dataset_core.reference_adapters import CSVReferenceAdapter, ManualEventAdapter
from dataset_core.validation_external import ExternalValidationService
from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, ExternalValidationConfig, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def test_external_validation_reports_pass_when_reference_matches(tmp_path, patch_market_context, reference_dir):
    frame = make_provider_frame("MSFT")
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
        external_validation=ExternalValidationConfig(reference_dir=reference_dir),
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]

    assert result.external_validation_status == "passed"
    assert result.artifacts.external_json.exists()
    assert result.artifacts.external_txt.exists()


def test_external_validation_reports_not_validated_without_reference(tmp_path, patch_market_context):
    frame = make_provider_frame("MSFT")
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]

    assert result.external_validation_status == "not_validated"
    assert result.external_validation_coverage_status == "none"
    assert result.external_validation_comparison_status == "not_validated"
    assert result.status == "warning"
    assert result.validation_outcome == "success_partial_validation"
    assert any("external validation did not validate the dataset" in reason.lower() for reason in result.status_reasons)


def test_external_validation_reports_adapter_error_separately_from_missing_reference(tmp_path):
    dataset = make_provider_frame("MSFT")
    service = ExternalValidationService(adapters=[CSVReferenceAdapter(tmp_path / "missing")])

    report = service.validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "not_validated"
    assert report["adapter_reports"][0]["status"] == "not_validated"


def test_external_validation_reports_runtime_adapter_failures_explicitly():
    dataset = make_provider_frame("MSFT")

    class _BrokenAdapter:
        def name(self):
            return "broken"

        def fetch_reference(self, symbol, start, end):
            raise RuntimeError("boom")

    report = ExternalValidationService(adapters=[_BrokenAdapter()]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "adapter_error"
    assert report["adapter_reports"][0]["status"] == "adapter_error"


def test_external_validation_zero_reference_does_not_false_pass():
    dataset = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [10.0, 10.0],
            "adj_close": [1.0, 1.0],
        }
    )
    reference = dataset.copy()
    reference["volume"] = [0.0, 0.0]

    class _Adapter:
        def name(self):
            return "inline"

        def fetch_reference(self, symbol, start, end):
            return reference

    report = ExternalValidationService(adapters=[_Adapter()]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "passed_partial"
    adapter_report = report["adapter_reports"][0]
    assert adapter_report["status"] == "passed_partial"
    assert adapter_report["comparison_status"] == "passed"
    assert "advisory_zero_baseline_mismatch" in adapter_report["partial_validation_kinds"]
    assert adapter_report["zero_reference_mismatch_count"] == 2


def test_external_validation_blocking_zero_reference_mismatch_still_fails():
    dataset = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [10.0, 10.0],
            "adj_close": [1.0, 1.0],
        }
    )
    reference = dataset.copy()
    reference["close"] = [0.0, 0.0]

    class _Adapter:
        def name(self):
            return "inline"

        def fetch_reference(self, symbol, start, end):
            return reference

    report = ExternalValidationService(adapters=[_Adapter()]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "failed"
    assert report["comparison_status"] == "failed"
    assert report["adapter_reports"][0]["status"] == "failed"
    assert report["adapter_reports"][0]["comparison_status"] == "failed"


def test_external_validation_normalizes_naive_and_utc_aware_dates_before_merge():
    dataset = make_provider_frame("MSFT", periods=2)
    reference = dataset.copy()
    reference["date"] = pd.to_datetime(reference["date"], utc=True)

    class _Adapter:
        def name(self):
            return "inline"

        def fetch_reference(self, symbol, start, end):
            return reference

    report = ExternalValidationService(adapters=[_Adapter()]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    assert report["status"] == "passed"
    assert report["adapter_reports"][0]["overlap_rows"] == 2


def test_manual_event_adapter_is_validated_as_sparse_events_not_price_reference(tmp_path):
    dataset = make_provider_frame("MSFT", periods=3)
    dataset.loc[1, "stock_splits"] = 2.0
    events_file = tmp_path / "manual_events.json"
    events_file.write_text(
        json.dumps(
            [
                {
                    "symbol": "MSFT",
                    "date": str(dataset.loc[1, "date"].date()),
                    "stock_splits": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = ExternalValidationService(event_adapters=[ManualEventAdapter(events_file)]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    adapter_report = report["adapter_reports"][0]
    assert report["status"] == "passed"
    assert adapter_report["scope"] == "event"
    assert adapter_report["checked_event_count"] == 1
    assert adapter_report["matched_event_count"] == 1
    assert "gap_ratio" not in adapter_report


def test_manual_event_adapter_does_not_fail_on_sparse_reference_calendar(tmp_path):
    dataset = make_provider_frame("MSFT", periods=5)
    dataset.loc[2, "dividends"] = 1.5
    dataset.loc[4, "stock_splits"] = 3.0
    events_file = tmp_path / "manual_events.csv"
    pd.DataFrame(
        [
            {"symbol": "MSFT", "date": dataset.loc[2, "date"], "dividends": 1.5},
        ]
    ).to_csv(events_file, index=False)

    report = ExternalValidationService(event_adapters=[ManualEventAdapter(events_file)]).validate(
        frame=dataset,
        symbol="MSFT",
        start=None,
        end=None,
    ).to_dict()

    adapter_report = report["adapter_reports"][0]
    assert report["status"] == "passed"
    assert adapter_report["status"] == "passed"
    assert adapter_report["scope"] == "event"
    assert adapter_report["checked_event_count"] == 1
