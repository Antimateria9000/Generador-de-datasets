from __future__ import annotations

import json

from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.export_service import DatasetExportService
from dataset_core.workspace_cleanup import cleanup_runs, select_runs_for_cleanup
from dataset_core.workspace_inventory import list_workspace_runs
from tests.fixtures.sample_data import DummyAcquisitionService, make_nvda_like_split_frame, make_provider_frame, make_raw_split_frame


def test_prepare_run_directories_recovers_from_collision(monkeypatch, tmp_path):
    run_ids = iter(["20260328_010101_000000_deadbeef", "20260328_010101_000000_feedbeef"])
    monkeypatch.setattr("dataset_core.export_service.build_run_id", lambda *_args, **_kwargs: next(run_ids))
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
        acquisition_service=DummyAcquisitionService({"MSFT": make_raw_split_frame()})
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
    assert result.status == "warning"
    assert result.artifacts.meta.exists()


def test_qlib_preset_is_closed_in_backend_even_with_incompatible_extras(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"NVDA": make_nvda_like_split_frame()})
    )
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["NVDA"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        mode="qlib",
        extras=["dividends", "stock_splits", "adj_close"],
        qlib_sanitization=False,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]
    frame = json.loads(result.artifacts.meta.read_text(encoding="utf-8"))

    assert request.qlib_sanitization is True
    assert request.extras == ["factor", "adj_close"]
    assert result.status == "success"
    assert frame["datasets"]["qlib"]["contract"] == "qlib_strict"
    assert frame["datasets"]["general"]["contract"] == "canonical_general"


def test_workspace_inventory_and_cleanup_support_selective_runs(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService(
            {
                "MSFT": make_provider_frame("MSFT"),
                "NVDA": make_nvda_like_split_frame(),
            }
        )
    )
    orchestrator = BatchOrchestrator(export_service=export_service)
    first = orchestrator.run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            mode="base",
            dq_mode="off",
        )
    )
    second = orchestrator.run(
        DatasetRequest(
            tickers=["NVDA"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            mode="qlib",
            dq_mode="off",
        )
    )

    inventory = list_workspace_runs(tmp_path)

    assert {record.run_id for record in inventory} >= {first.run_id, second.run_id}
    assert any(record.preset == "qlib" for record in inventory)

    selected = select_runs_for_cleanup(tmp_path, run_ids=[first.run_id])
    result = cleanup_runs(tmp_path, run_ids=[record.run_id for record in selected], dry_run=False)

    assert first.run_id in result.run_ids
    assert not (tmp_path / "runs" / first.run_id).exists()
    assert (tmp_path / "runs" / second.run_id).exists()
