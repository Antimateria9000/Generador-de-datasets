from __future__ import annotations

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def test_european_tickers_keep_suffixes_and_export_cleanly(tmp_path, patch_market_context):
    symbols = ["BBVA.MC", "AIR.PA", "SAP.DE", "ASML.AS"]
    datasets = {symbol: make_provider_frame(symbol) for symbol in symbols}
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService(datasets))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=symbols,
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)

    assert [result.ticker for result in batch_result.results] == symbols
    assert all(result.status != "error" for result in batch_result.results)
    assert [result.artifacts.csv.name for result in batch_result.results] == [
        "BBVA.MC_1d_5y.csv",
        "AIR.PA_1d_5y.csv",
        "SAP.DE_1d_5y.csv",
        "ASML.AS_1d_5y.csv",
    ]
