from __future__ import annotations

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from scripts.clean_workspace import main as clean_workspace_main
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def test_clean_workspace_script_supports_selective_dry_run(tmp_path, patch_market_context, capsys):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_provider_frame("MSFT")})
    )
    batch_result = BatchOrchestrator(export_service=export_service).run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            dq_mode="off",
        )
    )

    exit_code = clean_workspace_main(
        [
            "--workspace-root",
            str(tmp_path),
            "--run-id",
            batch_result.run_id,
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert batch_result.run_id in captured.out
    assert (tmp_path / "runs" / batch_result.run_id).exists()


def test_clean_workspace_script_rejects_negative_older_than_days(tmp_path):
    try:
        clean_workspace_main(
            [
                "--workspace-root",
                str(tmp_path),
                "--older-than-days",
                "-1",
            ]
        )
    except SystemExit as exc:
        assert "--older-than-days must be >= 0" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for negative older-than-days.")
