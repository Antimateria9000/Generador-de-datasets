from __future__ import annotations

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.export_service import DatasetExportService
from export_ohlcv_csv import run_cli
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def test_cli_smoke_runs_batch_pipeline_with_real_core(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService(
            {
                "MSFT": make_provider_frame("MSFT"),
                "AAPL": make_provider_frame("AAPL"),
            }
        )
    )
    orchestrator = BatchOrchestrator(export_service=export_service)

    batch_result = run_cli(
        [
            "--tickers",
            "MSFT,AAPL",
            "--years",
            "5",
            "--outdir",
            str(tmp_path),
            "--mode",
            "extended",
            "--dq-mode",
            "off",
        ],
        orchestrator=orchestrator,
    )

    assert batch_result.status_counts["error"] == 0
    assert len(batch_result.results) == 2
    assert batch_result.results[0].artifacts.csv.exists()
