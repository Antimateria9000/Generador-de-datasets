from __future__ import annotations

import json

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import (
    DatasetRequest,
    EODHDExternalValidationConfig,
    ExternalValidationConfig,
    TemporalRange,
)
from dataset_core.export_service import DatasetExportService
from dataset_core.external_sources.eodhd import EODHDPayload
from dataset_core.external_sources.base import ExternalSourceNotFoundError
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def _price_payload_from_frame(frame):
    payload = []
    for row in frame.itertuples(index=False):
        payload.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "adjusted_close": float(row.adj_close),
                "volume": float(row.volume),
            }
        )
    return payload


def test_external_validation_eodhd_provider_integrates_with_batch_orchestrator(tmp_path, patch_market_context, monkeypatch):
    frame = make_provider_frame("MSFT")

    class _MockEODHDClient:
        def __init__(self, **_kwargs) -> None:
            self.metrics = {"request_count": 3, "cache_hits": 0, "cache_misses": 3}

        def fetch_prices(self, symbol, start, end):
            if symbol == "MSFT":
                raise ExternalSourceNotFoundError("symbol not found")
            return EODHDPayload(
                payload=_price_payload_from_frame(frame),
                url="https://eodhd.test/api/eod/MSFT.US",
                cache_status="miss",
                endpoint="/api/eod/MSFT.US",
            )

        def fetch_dividends(self, symbol, start, end):
            return EODHDPayload(
                payload=[],
                url="https://eodhd.test/api/div/MSFT.US",
                cache_status="miss",
                endpoint="/api/div/MSFT.US",
            )

        def fetch_splits(self, symbol, start, end):
            return EODHDPayload(
                payload=[],
                url="https://eodhd.test/api/splits/MSFT.US",
                cache_status="miss",
                endpoint="/api/splits/MSFT.US",
            )

    monkeypatch.setattr("dataset_core.external_sources.factory.EODHDClient", _MockEODHDClient)
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
        external_validation=ExternalValidationConfig(
            enabled=True,
            provider="eodhd",
            eodhd=EODHDExternalValidationConfig(api_key="secret"),
        ),
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]
    report = json.loads(result.artifacts.external_json.read_text(encoding="utf-8"))

    assert result.external_validation_status == "passed"
    assert report["adapter_reports"][0]["source_metadata"]["provider"] == "eodhd"
    assert report["adapter_reports"][0]["source_metadata"]["provider_symbol"] == "MSFT.US"


def test_external_validation_explicit_csv_provider_keeps_legacy_flow(tmp_path, patch_market_context, reference_dir):
    frame = make_provider_frame("MSFT")
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
        external_validation=ExternalValidationConfig(
            enabled=True,
            provider="csv",
            reference_dir=reference_dir,
        ),
    )

    batch_result = orchestrator.run(request)

    assert batch_result.results[0].external_validation_status == "passed"
