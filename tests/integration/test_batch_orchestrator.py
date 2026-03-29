from __future__ import annotations

import json

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def test_batch_orchestrator_survives_partial_failures(tmp_path, patch_market_context):
    datasets = {
        "MSFT": make_provider_frame("MSFT"),
        "AAPL": make_provider_frame("AAPL"),
        "NVDA": make_provider_frame("NVDA"),
        "AMZN": make_provider_frame("AMZN"),
        "META": make_provider_frame("META"),
        "GOOG": make_provider_frame("GOOG"),
        "TSLA": make_provider_frame("TSLA"),
        "IBM": make_provider_frame("IBM"),
        "ORCL": make_provider_frame("ORCL"),
        "BAD": ValueError("Ticker not found"),
    }
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService(datasets))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOG", "TSLA", "IBM", "ORCL", "BAD"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)

    assert len(batch_result.results) == 10
    assert batch_result.status_counts["error"] == 1
    assert batch_result.manifest_json_path.exists()
    assert batch_result.manifest_txt_path.exists()
    assert [result.ticker for result in batch_result.results[:3]] == ["MSFT", "AAPL", "NVDA"]


def test_batch_orchestrator_deduplicates_before_processing(tmp_path, patch_market_context):
    datasets = {
        "MSFT": make_provider_frame("MSFT"),
        "AAPL": make_provider_frame("AAPL"),
    }
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService(datasets))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT", "AAPL", "MSFT", "AAPL"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)

    assert [result.ticker for result in batch_result.results] == ["MSFT", "AAPL"]


def test_batch_orchestrator_persists_run_log_and_enriched_error_meta(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": RuntimeError("forced acquisition failure")})
    )
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]
    meta_payload = json.loads(result.artifacts.meta.read_text(encoding="utf-8"))

    assert batch_result.run_log_path is not None
    assert batch_result.run_log_path.exists()
    assert result.status == "error"
    assert meta_payload["stage"] == "acquisition"
    assert meta_payload["exception_type"] == "RuntimeError"
    assert meta_payload["run_log_path"] == str(batch_result.run_log_path.resolve())
