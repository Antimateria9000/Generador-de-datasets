from __future__ import annotations

import pytest

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import RequestContractError
from dataset_core.export_service import DatasetExportService
from dataset_core.settings import EXTERNAL_VALIDATION_DISABLED_REASON
from export_ohlcv_csv import build_parser, build_request_from_args, export_one_ticker, run_cli
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


def test_cli_build_request_exposes_runtime_controls(tmp_path):
    args = build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--outdir",
            str(tmp_path),
            "--provider-metadata-timeout",
            "1.5",
            "--provider-metadata-candidate-limit",
            "2",
            "--provider-context-cache-ttl-seconds",
            "600",
            "--provider-batch-max-workers",
            "3",
            "--provider-batch-chunk-size",
            "2",
            "--execution-mode",
            "sequential",
        ]
    )

    request = build_request_from_args(args)

    assert request.provider.metadata_timeout == 1.5
    assert request.provider.metadata_candidate_limit == 2
    assert request.provider.context_cache_ttl_seconds == 600
    assert request.provider.batch_max_workers == 3
    assert request.provider.batch_chunk_size == 2
    assert args.execution_mode == "sequential"


def test_cli_build_request_ignores_external_validation_runtime_arguments_while_module_is_disabled(tmp_path):
    args = build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--outdir",
            str(tmp_path),
            "--external-validation-enabled",
            "--external-validation-provider",
            "eodhd",
            "--eodhd-api-key",
            "secret",
            "--eodhd-timeout-seconds",
            "4.5",
            "--eodhd-cache-ttl-seconds",
            "600",
            "--eodhd-max-retries",
            "3",
            "--eodhd-backoff-seconds",
            "0.75",
            "--eodhd-allow-partial-coverage",
        ]
    )

    request = build_request_from_args(args)

    assert request.external_validation.is_enabled() is False
    assert request.external_validation.resolved_provider() is None
    assert request.external_validation.eodhd.api_key is None
    assert request.external_validation.to_dict()["status"] == "disabled"


def test_cli_help_hides_external_validation_flags_when_runtime_is_disabled():
    help_text = build_parser().format_help()

    assert "External validation runtime is disabled in this build." in help_text
    assert "Compatibility flags remain accepted but are hidden from --help." in help_text
    assert "Module disabled by" in help_text
    assert "--external-validation-provider" not in help_text
    assert "--eodhd-api-key" not in help_text


def test_cli_and_api_share_the_same_qlib_filename_contract(tmp_path):
    args = build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--outdir",
            str(tmp_path),
            "--mode",
            "qlib",
            "--filename",
            "custom.csv",
        ]
    )

    with pytest.raises(RequestContractError, match="Custom filenames are not supported in qlib mode."):
        build_request_from_args(args)
    with pytest.raises(RequestContractError, match="Custom filenames are not supported in qlib mode."):
        export_one_ticker(
            ticker="MSFT",
            years=5,
            start=None,
            end=None,
            outdir=str(tmp_path),
            mode="qlib",
            filename="custom.csv",
        )
