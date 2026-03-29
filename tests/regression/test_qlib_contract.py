from __future__ import annotations

import json

import pandas as pd

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_split_frame


def test_qlib_contract_writes_expected_columns_and_filename(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_split_frame()})
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


def test_batch_manifest_keeps_global_summary_and_paths(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService(
            {
                "MSFT": make_split_frame(),
                "AAPL": make_split_frame(),
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
    assert len(manifest["results"]) == 2
    assert manifest["results"][0]["csv_path"]
    assert manifest["run_log_path"] == str(batch_result.run_log_path.resolve())
