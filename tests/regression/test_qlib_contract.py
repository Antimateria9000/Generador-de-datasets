from __future__ import annotations

import json

import pandas as pd

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_nvda_like_split_frame


def test_qlib_contract_writes_expected_columns_and_filename(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_nvda_like_split_frame()})
    )
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        mode="qlib",
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]
    frame = pd.read_csv(result.artifacts.csv)
    manifest = json.loads(result.artifacts.manifest.read_text(encoding="utf-8"))

    assert result.artifacts.csv.name == "MSFT.csv"
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume", "factor"]
    assert manifest["qlib_compatible"] is True
    assert result.status == "warning"
    assert result.validation_outcome == "success_partial_validation"
    assert result.factor_source == "adj_close_ratio"


def test_batch_manifest_keeps_global_summary_and_paths(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService(
            {
                "MSFT": make_nvda_like_split_frame(),
                "AAPL": make_nvda_like_split_frame(),
            }
        )
    )
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT", "AAPL"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        mode="qlib",
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    manifest = json.loads(batch_result.manifest_json_path.read_text(encoding="utf-8"))

    assert manifest["status_counts"]["error"] == 0
    assert manifest["validation_outcome_counts"]["success_partial_validation"] == 2
    assert len(manifest["results"]) == 2
    assert manifest["results"][0]["csv_path"]
    assert manifest["run_log_path"] == str(batch_result.run_log_path.resolve())
    assert manifest["results"][0]["status"] == "warning"


def test_qlib_failure_does_not_advertise_missing_dq_artifact(tmp_path, patch_market_context):
    invalid_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "open": [10.0],
            "high": [9.0],
            "low": [8.0],
            "close": [9.5],
            "adj_close": [9.5],
            "volume": [1_000.0],
            "dividends": [0.0],
            "stock_splits": [0.0],
        }
    )
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": invalid_frame})
    )
    batch_result = BatchOrchestrator(export_service=export_service).run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            mode="qlib",
            dq_mode="off",
        )
    )
    result = batch_result.results[0]
    meta_payload = json.loads(result.artifacts.meta.read_text(encoding="utf-8"))

    assert result.status == "error"
    assert meta_payload["artifacts"]["dq_path"] is None
    assert not (batch_result.report_dir / "MSFT.dq.json").exists()
