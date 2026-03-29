from __future__ import annotations

import json

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame, make_split_frame


def test_prepare_run_directories_recovers_from_collision(monkeypatch, tmp_path):
    run_ids = iter(["20260328_010101_000000_deadbeef", "20260328_010101_000000_feedbeef"])
    monkeypatch.setattr("dataset_core.export_service.build_run_id", lambda: next(run_ids))
    (tmp_path / "runs" / "20260328_010101_000000_deadbeef").mkdir(parents=True, exist_ok=False)

    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": make_provider_frame("MSFT")}))
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    run_dirs = export_service.prepare_run_directories(request)

    assert run_dirs.run_id == "20260328_010101_000000_feedbeef"
    assert run_dirs.output_root == tmp_path / "runs" / "20260328_010101_000000_feedbeef"


def test_parallel_qlib_sanitization_writes_general_and_qlib_outputs(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_split_frame()})
    )
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        mode="extended",
        extras=["adj_close"],
        qlib_sanitization=True,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]
    qlib_report = json.loads(result.artifacts.qlib_report.read_text(encoding="utf-8"))

    assert result.artifacts.csv.exists()
    assert result.artifacts.qlib_csv.exists()
    assert result.artifacts.csv != result.artifacts.qlib_csv
    assert result.qlib_compatible is True
    assert qlib_report["qlib_compatible"] is True
