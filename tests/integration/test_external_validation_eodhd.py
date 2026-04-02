from __future__ import annotations

import json

import pandas as pd

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
    start = str(frame["date"].min().date())
    end = str((frame["date"].max() + pd.Timedelta(days=1)).date())
    monkeypatch.setattr(
        "dataset_core.external_sources.factory.EODHDClient",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("EODHDClient should not be instantiated while external validation is disabled.")
        ),
    )
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=None, start=start, end=end, interval="1d"),
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

    assert result.external_validation_status == "disabled"
    assert result.external_validation_coverage_status is None
    assert result.external_validation_comparison_status is None
    assert report["enabled"] is False
    assert report["status"] == "disabled"
    assert report["adapter_reports"] == []


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

    assert batch_result.results[0].external_validation_status == "disabled"
