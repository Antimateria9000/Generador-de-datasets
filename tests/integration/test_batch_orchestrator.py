from __future__ import annotations

import json

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, FakeContext, make_provider_frame


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


def test_contextual_market_notes_do_not_degrade_status(monkeypatch, tmp_path):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_provider_frame("MSFT")})
    )
    monkeypatch.setattr(
        "dataset_core.export_service.resolve_instrument_context",
        lambda *args, **kwargs: FakeContext(
            requested_symbol="MSFT",
            preferred_symbol="MSFT",
            warnings=[
                "No se obtuvo metadata de Yahoo para enriquecer el contexto del instrumento.",
                "quoteType no disponible; se asumio perfil por heuristica.",
            ],
        ),
    )
    monkeypatch.setattr(
        "dataset_core.export_service.build_dq_context_payload",
        lambda context: {
            "asset_type": context.asset_type,
            "asset_family": context.asset_family,
            "quote_type": context.quote_type,
            "market": context.market,
            "calendar": context.calendar,
            "timezone": context.timezone,
            "currency": context.currency,
            "exchange_name": context.exchange_name,
            "exchange_code": context.exchange_code,
            "region": context.region,
            "requested_symbol": context.requested_symbol,
            "preferred_symbol": context.preferred_symbol,
            "listing_preference": context.listing_preference,
            "is_24_7": context.is_24_7,
            "volume_expected": context.volume_expected,
            "corporate_actions_expected": context.corporate_actions_expected,
            "calendar_validation_supported": context.calendar_validation_supported,
            "dq_profile": context.dq_profile,
            "confidence": context.confidence,
            "warnings": list(context.warnings),
            "inference_sources": list(context.inference_sources),
        },
    )

    result = BatchOrchestrator(export_service=export_service).run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            dq_mode="off",
        )
    ).results[0]

    assert result.status == "success"
    assert result.warnings == []
    assert any("metadata de Yahoo" in note for note in result.neutral_notes)


def test_internal_validation_unsupported_does_not_degrade_status(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_provider_frame("MSFT")})
    )
    result = BatchOrchestrator(export_service=export_service).run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None, interval="1h"),
            output_dir=tmp_path,
            interval="1h",
            dq_mode="report",
        )
    ).results[0]

    assert result.internal_validation_status == "unsupported"
    assert result.status == "success"
    assert any("interval=1d" in note for note in result.neutral_notes)


def test_batch_orchestrator_cleans_orphan_tmp_files_after_run(tmp_path, patch_market_context):
    stray_tmp = tmp_path / "reports" / "orphan.meta.json.deadbeef.tmp"
    stray_tmp.parent.mkdir(parents=True, exist_ok=True)
    stray_tmp.write_text("{}", encoding="utf-8")

    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_provider_frame("MSFT")})
    )
    BatchOrchestrator(export_service=export_service).run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            dq_mode="off",
        )
    )

    assert not stray_tmp.exists()
