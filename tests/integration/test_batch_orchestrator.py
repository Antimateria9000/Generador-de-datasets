from __future__ import annotations

import json
import threading
import time

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, ProviderConfig, TemporalRange
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


def test_batch_orchestrator_sequential_mode_smoke(tmp_path, patch_market_context):
    export_service = DatasetExportService(
        acquisition_service=DummyAcquisitionService({"MSFT": make_provider_frame("MSFT")})
    )

    batch_result = BatchOrchestrator(export_service=export_service).run(
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            output_dir=tmp_path,
            dq_mode="off",
        ),
        execution_mode="sequential",
    )

    assert batch_result.results[0].status == "success"
    assert batch_result.results[0].artifacts.csv.exists()
    assert export_service.acquisition_service.last_session.metrics["fetch_calls"] == 1


def test_batch_orchestrator_keeps_logical_equivalence_between_sequential_and_concurrent_modes(tmp_path, patch_market_context):
    datasets = {
        "MSFT": make_provider_frame("MSFT"),
        "AAPL": make_provider_frame("AAPL"),
        "NVDA": make_provider_frame("NVDA"),
    }

    sequential_service = DatasetExportService(acquisition_service=DummyAcquisitionService(datasets))
    concurrent_service = DatasetExportService(acquisition_service=DummyAcquisitionService(datasets))

    sequential_request = DatasetRequest(
        tickers=["MSFT", "AAPL", "NVDA"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path / "sequential",
        dq_mode="off",
    )
    concurrent_request = DatasetRequest(
        tickers=["MSFT", "AAPL", "NVDA"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path / "concurrent",
        dq_mode="off",
    )

    sequential = BatchOrchestrator(export_service=sequential_service).run(
        sequential_request,
        execution_mode="sequential",
    )
    concurrent = BatchOrchestrator(export_service=concurrent_service).run(
        concurrent_request,
        execution_mode="concurrent",
    )

    sequential_manifest = json.loads(sequential.manifest_json_path.read_text(encoding="utf-8"))
    concurrent_manifest = json.loads(concurrent.manifest_json_path.read_text(encoding="utf-8"))

    def _logical_signature(batch_result):
        return [
            (
                result.requested_ticker,
                result.resolved_ticker,
                result.status,
                result.qlib_compatible,
                tuple(result.columns),
            )
            for result in batch_result.results
        ]

    assert _logical_signature(sequential) == _logical_signature(concurrent)
    assert [item["requested_ticker"] for item in sequential_manifest["results"]] == ["MSFT", "AAPL", "NVDA"]
    assert [item["requested_ticker"] for item in concurrent_manifest["results"]] == ["MSFT", "AAPL", "NVDA"]
    assert [item["status"] for item in sequential_manifest["results"]] == [
        item["status"] for item in concurrent_manifest["results"]
    ]
    assert sequential_service.acquisition_service.last_session.metrics["fetch_calls"] == 3
    assert concurrent_service.acquisition_service.last_session.metrics["fetch_calls"] == 0
    assert concurrent_service.acquisition_service.last_session.metrics["fetch_many_calls"] == 1


def test_batch_orchestrator_runs_planning_and_finalization_concurrently(tmp_path, patch_market_context, monkeypatch):
    datasets = {
        "MSFT": make_provider_frame("MSFT"),
        "AAPL": make_provider_frame("AAPL"),
        "NVDA": make_provider_frame("NVDA"),
        "AMZN": make_provider_frame("AMZN"),
    }
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService(datasets))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT", "AAPL", "NVDA", "AMZN"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
        provider=ProviderConfig(batch_max_workers=3, batch_chunk_size=2),
    )

    planning_threads: set[int] = set()
    export_threads: set[int] = set()
    planning_lock = threading.Lock()
    export_lock = threading.Lock()
    original_resolve_context = export_service.resolve_context
    original_export_ticker = export_service.export_ticker

    def _resolve_context(*args, **kwargs):
        time.sleep(0.03)
        with planning_lock:
            planning_threads.add(threading.get_ident())
        return original_resolve_context(*args, **kwargs)

    def _export_ticker(*args, **kwargs):
        time.sleep(0.03)
        with export_lock:
            export_threads.add(threading.get_ident())
        return original_export_ticker(*args, **kwargs)

    monkeypatch.setattr(export_service, "resolve_context", _resolve_context)
    monkeypatch.setattr(export_service, "export_ticker", _export_ticker)

    batch_result = orchestrator.run(request, execution_mode="concurrent")

    assert batch_result.status_counts["error"] == 0
    assert len(planning_threads) > 1
    assert len(export_threads) > 1


def test_batch_orchestrator_chunks_grouped_acquisition_requests(tmp_path, patch_market_context):
    datasets = {
        "MSFT": make_provider_frame("MSFT"),
        "AAPL": make_provider_frame("AAPL"),
        "NVDA": make_provider_frame("NVDA"),
        "AMZN": make_provider_frame("AMZN"),
        "META": make_provider_frame("META"),
    }
    acquisition_service = DummyAcquisitionService(datasets)
    export_service = DatasetExportService(acquisition_service=acquisition_service)
    request = DatasetRequest(
        tickers=["MSFT", "AAPL", "NVDA", "AMZN", "META"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
        provider=ProviderConfig(batch_max_workers=3, batch_chunk_size=2),
    )

    batch_result = BatchOrchestrator(export_service=export_service).run(
        request,
        execution_mode="concurrent",
    )

    assert batch_result.status_counts["error"] == 0
    assert len(acquisition_service.fetch_many_inputs) == 3
    assert set(acquisition_service.fetch_many_inputs) == {
        ("MSFT", "AAPL"),
        ("NVDA", "AMZN"),
        ("META",),
    }
