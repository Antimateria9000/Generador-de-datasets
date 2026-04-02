from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from dataset_core.acquisition import AcquisitionService, ProviderSession
from dataset_core.contracts import DatasetRequest
from dataset_core.factor_policy import resolve_provider_flags
from dataset_core.logging_runtime import bind_runtime_logger, configure_run_logger
from dataset_core.manifest_service import build_ticker_manifest
from dataset_core.naming import (
    artifact_stem,
    build_csv_output_path,
    build_range_tag,
    build_run_directory,
    build_run_id,
    sanitize_symbol_for_csv,
)
from dataset_core.path_safety import assert_within_root
from dataset_core.result_models import ArtifactPaths, TickerResult
from dataset_core.sanitization_general import GeneralSanitizer
from dataset_core.sanitization_qlib import QlibSanitizationError, QlibSanitizer
from dataset_core.schema_builder import DatasetSchemaBuilder
from dataset_core.serialization import compute_sha256, write_csv, write_json, write_text
from dataset_core.settings import (
    DEFAULT_METADATA_CANDIDATE_LIMIT,
    ensure_directory,
    ensure_workspace_tree,
    resolve_effective_cache_paths,
    sanitize_secret_text,
    utc_now_iso,
)
from dataset_core.status_resolution import resolve_ticker_status
from dataset_core.validation_external import ExternalValidationService
from dataset_core.validation_internal import InternalDQService
from providers.market_context import ContextResolver, build_dq_context_payload, resolve_instrument_context


@dataclass(frozen=True)
class RunDirectories:
    run_id: str
    workspace_root: Path
    output_root: Path
    csv_dir: Path
    canonical_dir: Path
    qlib_dir: Path
    meta_dir: Path
    report_dir: Path
    temp_dir: Path
    log_dir: Path
    run_log_path: Path

    def primary_csv_dir(self, request: DatasetRequest) -> Path:
        return self.qlib_dir if request.mode == "qlib" else self.csv_dir


def _existing_artifacts_only(artifacts: ArtifactPaths) -> ArtifactPaths:
    return ArtifactPaths(
        csv=artifacts.csv if artifacts.csv is not None and artifacts.csv.exists() else None,
        canonical_csv=artifacts.canonical_csv
        if artifacts.canonical_csv is not None and artifacts.canonical_csv.exists()
        else None,
        qlib_csv=artifacts.qlib_csv if artifacts.qlib_csv is not None and artifacts.qlib_csv.exists() else None,
        meta=artifacts.meta,
        dq=artifacts.dq if artifacts.dq is not None and artifacts.dq.exists() else None,
        external_json=artifacts.external_json
        if artifacts.external_json is not None and artifacts.external_json.exists()
        else None,
        external_txt=artifacts.external_txt
        if artifacts.external_txt is not None and artifacts.external_txt.exists()
        else None,
        manifest=artifacts.manifest,
        qlib_report=artifacts.qlib_report
        if artifacts.qlib_report is not None and artifacts.qlib_report.exists()
        else None,
    )


class DatasetExportService:
    def __init__(
        self,
        acquisition_service: AcquisitionService | None = None,
        schema_builder: DatasetSchemaBuilder | None = None,
        general_sanitizer: GeneralSanitizer | None = None,
        qlib_sanitizer: QlibSanitizer | None = None,
        internal_validation: InternalDQService | None = None,
        external_validation: ExternalValidationService | None = None,
    ) -> None:
        self.acquisition_service = acquisition_service or AcquisitionService()
        self.schema_builder = schema_builder or DatasetSchemaBuilder()
        self.general_sanitizer = general_sanitizer or GeneralSanitizer()
        self.qlib_sanitizer = qlib_sanitizer or QlibSanitizer()
        self.internal_validation = internal_validation or InternalDQService()
        self.external_validation = external_validation or ExternalValidationService()

    def resolve_context(
        self,
        ticker: str,
        request: DatasetRequest,
        *,
        context_resolver: ContextResolver | None = None,
    ):
        market_override = None if request.dq_market == "AUTO" else request.dq_market
        cache_paths = resolve_effective_cache_paths(request.output_dir, request.provider.cache_dir)
        return resolve_instrument_context(
            symbol=str(ticker or "").strip().upper(),
            market_override=market_override,
            listing_preference=request.listing_preference,
            metadata_timeout=request.provider.metadata_timeout or request.provider.timeout,
            resolver=context_resolver,
            candidate_limit=request.provider.metadata_candidate_limit or DEFAULT_METADATA_CANDIDATE_LIMIT,
            cache_dir=cache_paths["market_context"],
            cache_ttl_seconds=request.provider.context_cache_ttl_seconds,
        )

    def prepare_run_directories(self, request: DatasetRequest) -> RunDirectories:
        workspace = ensure_workspace_tree(request.output_dir)

        last_error: FileExistsError | None = None
        for _ in range(20):
            run_id = build_run_id(request)
            output_root = build_run_directory(request, run_id)
            try:
                output_root.mkdir(parents=True, exist_ok=False)
                export_root = ensure_directory(workspace["exports"] / run_id)
                meta_dir = ensure_directory(workspace["manifests"] / run_id)
                report_dir = ensure_directory(workspace["reports"] / run_id)
                temp_dir = ensure_directory(workspace["temp"] / run_id)
                log_dir = ensure_directory(workspace["logs"] / run_id)
                csv_dir = ensure_directory(export_root / "csv")
                canonical_dir = ensure_directory(export_root / "canonical")
                qlib_dir = ensure_directory(export_root / "qlib")
                return RunDirectories(
                    run_id=run_id,
                    workspace_root=workspace["workspace_root"],
                    output_root=output_root,
                    csv_dir=csv_dir,
                    canonical_dir=canonical_dir,
                    qlib_dir=qlib_dir,
                    meta_dir=meta_dir,
                    report_dir=report_dir,
                    temp_dir=temp_dir,
                    log_dir=log_dir,
                    run_log_path=log_dir / "dataset_factory.log",
                )
            except FileExistsError as exc:
                last_error = exc
                continue

        raise RuntimeError("Could not create a unique run directory.") from last_error

    @staticmethod
    def _canonical_csv_filename(symbol: str, request: DatasetRequest) -> str:
        safe_symbol = sanitize_symbol_for_csv(symbol)
        return f"{safe_symbol}_canonical_{request.interval}_{build_range_tag(request.time_range)}.csv"

    def export_ticker(
        self,
        ticker: str,
        request: DatasetRequest,
        run_dirs: RunDirectories,
        *,
        prefetched_context=None,
        prefetched_fetch_result=None,
        provider_session: ProviderSession | None = None,
        context_resolver: ContextResolver | None = None,
        resolved_provider_flags: tuple[bool, bool, list[str]] | None = None,
    ) -> TickerResult:
        material_warnings: list[str] = []
        neutral_notes: list[str] = []
        errors: list[str] = []
        error_context: dict[str, object] | None = None
        stage = "ticker_start"
        requested_ticker = ticker.upper()
        resolved_ticker = requested_ticker
        provider_symbol = requested_ticker
        provider_metadata: dict[str, object] = {"warnings": []}
        provider_warnings: list[str] = []
        auto_adjust = bool(request.auto_adjust)
        actions = bool(request.actions)
        artifact_key = artifact_stem(requested_ticker)
        runtime_logger = bind_runtime_logger(
            configure_run_logger(run_dirs.run_id, run_dirs.workspace_root).logger,
            ticker=requested_ticker,
        )
        runtime_logger.info("Starting ticker export.", extra={"stage": stage})
        artifacts = ArtifactPaths(
            meta=run_dirs.report_dir / f"{artifact_key}.meta.json",
            dq=run_dirs.report_dir / f"{artifact_key}.dq.json",
            external_json=run_dirs.report_dir / f"{artifact_key}.external_validation.json",
            external_txt=run_dirs.report_dir / f"{artifact_key}.external_validation.txt",
            manifest=run_dirs.meta_dir / f"{artifact_key}.manifest.json",
            qlib_report=run_dirs.report_dir / f"{artifact_key}.qlib.json",
        )

        try:
            stage = "context_resolution"
            context = prefetched_context
            if context is None:
                context = self.resolve_context(
                    requested_ticker,
                    request,
                    context_resolver=context_resolver,
                )
            dq_context = build_dq_context_payload(context)
            resolved_ticker = str(context.preferred_symbol or requested_ticker).upper()
            provider_symbol = resolved_ticker
            neutral_notes.extend(list(context.warnings))

            stage = "acquisition"
            runtime_logger.info(
                "Fetching source data for resolved ticker %s.",
                resolved_ticker,
                extra={"stage": stage},
            )
            if resolved_provider_flags is None:
                auto_adjust, actions, factor_warnings = resolve_provider_flags(
                    auto_adjust=request.auto_adjust,
                    actions=request.actions,
                    requires_factor=request.requires_factor,
                )
            else:
                auto_adjust, actions, factor_warnings = resolved_provider_flags
            neutral_notes.extend(factor_warnings)

            fetch_result = prefetched_fetch_result
            if fetch_result is None:
                fetch_result = self.acquisition_service.fetch(
                    symbol=resolved_ticker,
                    request=request,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    session=provider_session,
                )
            provider_metadata = fetch_result.metadata.to_dict()
            provider_warnings = list(provider_metadata.get("warnings", []))
            neutral_notes.extend(provider_warnings)

            stage = "general_sanitization"
            runtime_logger.info("Running general sanitization.", extra={"stage": stage})
            general_result = self.general_sanitizer.sanitize(
                frame=fetch_result.data,
                requested_extras=request.extras,
            )
            material_warnings.extend(general_result.warnings)

            stage = "internal_validation"
            runtime_logger.info("Running internal validation.", extra={"stage": stage})
            internal_report = self.internal_validation.run(
                frame=general_result.frame,
                symbol=resolved_ticker,
                interval=request.interval,
                actions=actions,
                dq_mode=request.dq_mode,
                dq_context=dq_context,
            )

            stage = "external_validation"
            runtime_logger.info("Running external validation.", extra={"stage": stage})
            external_report = self.external_validation.validate(
                frame=general_result.frame,
                symbol=resolved_ticker,
                start=request.time_range.start_iso,
                end=request.time_range.end_iso,
            ).to_dict()

            qlib_payload: dict[str, object] | None = None
            qlib_compatible = False
            general_factor_policy = None
            general_factor_source = None
            qlib_factor_policy = None
            qlib_factor_source = None
            qlib_errors: list[str] = []
            general_dataset_payload: dict[str, object] | None = None
            qlib_dataset_payload: dict[str, object] | None = None

            if request.mode != "qlib":
                stage = "schema_build"
                runtime_logger.info("Building general output schema.", extra={"stage": stage})
                general_schema = self.schema_builder.build(
                    frame=general_result.frame,
                    mode=request.mode,
                    extras=request.extras,
                )
                neutral_notes.extend(general_schema.warnings)
                artifacts.csv = build_csv_output_path(run_dirs.csv_dir, resolved_ticker, request)
                artifacts.canonical_csv = artifacts.csv
                write_csv(general_schema.frame, artifacts.csv, temp_dir=run_dirs.temp_dir)
                general_factor_policy = general_schema.factor_policy
                general_factor_source = general_schema.factor_source
                primary_frame = general_schema.frame
                general_dataset_payload = {
                    "contract": general_schema.resolved_preset.preset.name,
                    "csv_path": str(artifacts.csv.resolve()),
                    "sha256": compute_sha256(artifacts.csv),
                    "row_count": int(len(primary_frame)),
                    "columns": list(primary_frame.columns),
                    "factor_policy": general_factor_policy,
                    "factor_source": general_factor_source,
                    "qlib_compatible": False,
                    "qlib_reasons": [],
                }
            else:
                stage = "schema_build"
                runtime_logger.info("Persisting canonical dataset for Qlib mode.", extra={"stage": stage})
                canonical_path = assert_within_root(
                    run_dirs.canonical_dir / self._canonical_csv_filename(resolved_ticker, request),
                    run_dirs.canonical_dir,
                )
                artifacts.canonical_csv = canonical_path
                write_csv(general_result.frame, canonical_path, temp_dir=run_dirs.temp_dir)
                primary_frame = general_result.frame
                general_dataset_payload = {
                    "contract": "canonical_general",
                    "csv_path": str(canonical_path.resolve()),
                    "sha256": compute_sha256(canonical_path),
                    "row_count": int(len(primary_frame)),
                    "columns": list(primary_frame.columns),
                    "factor_policy": None,
                    "factor_source": None,
                    "qlib_compatible": False,
                    "qlib_reasons": [],
                }

            if request.qlib_sanitization:
                stage = "qlib_sanitization"
                runtime_logger.info("Running Qlib sanitization.", extra={"stage": stage})
                try:
                    qlib_result = self.qlib_sanitizer.sanitize(general_result.frame)
                    qlib_schema = self.schema_builder.build(
                        frame=qlib_result.frame,
                        mode="qlib",
                        extras=[],
                    )
                    qlib_compatible = qlib_schema.qlib_compatible
                    qlib_factor_policy = qlib_result.factor_policy
                    qlib_factor_source = qlib_result.factor_source
                    neutral_notes.extend(qlib_result.warnings)
                    artifacts.qlib_csv = build_csv_output_path(
                        run_dirs.qlib_dir,
                        resolved_ticker,
                        request,
                        force_qlib_contract=True,
                    )
                    write_csv(qlib_schema.frame, artifacts.qlib_csv, temp_dir=run_dirs.temp_dir)
                    qlib_payload = dict(qlib_result.technical_report)
                    qlib_payload["status"] = "generated"
                    qlib_payload["csv_path"] = str(artifacts.qlib_csv.resolve())
                    qlib_payload["sha256"] = compute_sha256(artifacts.qlib_csv)
                    write_json(artifacts.qlib_report, qlib_payload, temp_dir=run_dirs.temp_dir)
                    qlib_dataset_payload = {
                        "contract": "qlib_strict",
                        "csv_path": str(artifacts.qlib_csv.resolve()),
                        "sha256": qlib_payload["sha256"],
                        "row_count": int(len(qlib_schema.frame)),
                        "columns": list(qlib_schema.frame.columns),
                        "factor_policy": qlib_factor_policy,
                        "factor_source": qlib_factor_source,
                        "qlib_compatible": qlib_compatible,
                        "qlib_reasons": list(qlib_schema.qlib_reasons),
                    }
                    if request.mode == "qlib":
                        artifacts.csv = artifacts.qlib_csv
                        primary_frame = qlib_schema.frame
                except QlibSanitizationError as exc:
                    qlib_errors.append(str(exc))
                    qlib_payload = {
                        "status": "failed",
                        "factor_policy": qlib_factor_policy,
                        "factor_source": qlib_factor_source,
                        "qlib_compatible": False,
                        "columns_emitted": [],
                        "warnings": [],
                        "reasons": qlib_errors,
                        "factor_semantic_checks": [],
                        "contract_checks": [],
                        "metrics": {},
                    }
                    qlib_dataset_payload = {
                        "contract": "qlib_strict",
                        "csv_path": None,
                        "sha256": None,
                        "row_count": 0,
                        "columns": [],
                        "factor_policy": qlib_factor_policy,
                        "factor_source": qlib_factor_source,
                        "qlib_compatible": False,
                        "qlib_reasons": list(qlib_errors),
                    }
                    write_json(artifacts.qlib_report, qlib_payload, temp_dir=run_dirs.temp_dir)
                    if request.mode == "qlib":
                        raise
                    material_warnings.append(f"Qlib sanitization failed: {exc}")
                    runtime_logger.warning("Qlib sanitization failed: %s", exc, extra={"stage": stage})

            if artifacts.csv is None:
                raise RuntimeError("The primary export CSV was not generated.")

            csv_sha256 = compute_sha256(artifacts.csv)
            primary_factor_policy = qlib_factor_policy if request.mode == "qlib" else general_factor_policy
            primary_factor_source = qlib_factor_source if request.mode == "qlib" else general_factor_source

            if qlib_dataset_payload is None and request.mode == "qlib":
                qlib_dataset_payload = {
                    "contract": "qlib_strict",
                    "csv_path": None,
                    "sha256": None,
                    "row_count": 0,
                    "columns": [],
                    "factor_policy": qlib_factor_policy,
                    "factor_source": qlib_factor_source,
                    "qlib_compatible": False,
                    "qlib_reasons": list(qlib_errors),
                }

            status_resolution = resolve_ticker_status(
                warnings=material_warnings,
                neutral_notes=neutral_notes,
                internal_validation_status=internal_report.get("status"),
                internal_validation_reason=internal_report.get("reason"),
                external_validation_status=external_report.get("status"),
                external_validation_reason=external_report.get("reason"),
                qlib_requested=request.qlib_sanitization,
                qlib_compatible=qlib_compatible,
                qlib_errors=qlib_errors,
            )
            status = status_resolution.status

            meta_payload = {
                "export_id": str(uuid4()),
                "generated_at_utc": utc_now_iso(),
                "run_id": run_dirs.run_id,
                "request": request.to_dict(),
                "ticker_resolution": {
                    "requested_ticker": requested_ticker,
                    "resolved_ticker": resolved_ticker,
                    "provider_symbol": provider_symbol,
                    "listing_preference": request.listing_preference,
                },
                "provider_request": {
                    "auto_adjust": auto_adjust,
                    "actions": actions,
                },
                "provider_metadata": provider_metadata,
                "instrument_context": context.to_dict(),
                "sanitization_general": {
                    "removed_rows": general_result.removed_rows,
                    "warnings": general_result.warnings,
                    "columns": general_result.columns,
                },
                "status_resolution": {
                    "status": status_resolution.status,
                    "validation_outcome": status_resolution.validation_outcome,
                    "reasons": list(status_resolution.reasons),
                    "neutral_notes": list(status_resolution.neutral_notes),
                },
                "artifacts": artifacts.to_dict(),
                "dataset": {
                    "csv_path": str(artifacts.csv.resolve()),
                    "sha256": csv_sha256,
                    "row_count": int(len(primary_frame)),
                    "columns": list(primary_frame.columns),
                    "qlib_compatible": qlib_compatible,
                    "qlib_reasons": qlib_errors,
                    "factor_policy": primary_factor_policy,
                    "factor_source": primary_factor_source,
                },
                "datasets": {
                    "primary_contract": "qlib_strict" if request.mode == "qlib" else request.mode,
                    "general": general_dataset_payload,
                    "qlib": qlib_dataset_payload,
                },
                "qlib_artifact": qlib_payload,
                "internal_validation": internal_report,
                "external_validation": external_report,
                "warnings": material_warnings,
                "neutral_notes": list(status_resolution.neutral_notes),
                "run_log_path": str(run_dirs.run_log_path.resolve()),
            }

            stage = "persistence"
            runtime_logger.info("Persisting ticker artifacts and manifests.", extra={"stage": stage})
            write_json(artifacts.meta, meta_payload, temp_dir=run_dirs.temp_dir)
            write_json(artifacts.dq, internal_report, temp_dir=run_dirs.temp_dir)
            write_json(artifacts.external_json, external_report, temp_dir=run_dirs.temp_dir)
            write_text(
                artifacts.external_txt,
                self.external_validation.render_text(external_report, resolved_ticker),
                temp_dir=run_dirs.temp_dir,
            )

            result = TickerResult(
                ticker=resolved_ticker,
                requested_ticker=requested_ticker,
                resolved_ticker=resolved_ticker,
                status=status,
                qlib_compatible=qlib_compatible,
                validation_outcome=status_resolution.validation_outcome,
                columns=list(primary_frame.columns),
                status_reasons=status_resolution.reasons,
                neutral_notes=status_resolution.neutral_notes,
                warnings=material_warnings,
                errors=errors,
                internal_validation_status=str(internal_report["status"]),
                external_validation_status=str(external_report["status"]),
                factor_policy=primary_factor_policy,
                factor_source=primary_factor_source,
                provider_symbol=provider_symbol,
                provider_warnings=provider_warnings,
                run_log_path=run_dirs.run_log_path,
                artifacts=artifacts,
            )

            write_json(artifacts.manifest, build_ticker_manifest(result), temp_dir=run_dirs.temp_dir)
            bind_runtime_logger(
                runtime_logger,
                ticker=resolved_ticker,
                stage="ticker_finish",
            ).info(
                "Ticker export finished with status=%s validation_outcome=%s.",
                status,
                status_resolution.validation_outcome,
            )
            return result

        except Exception as exc:
            traceback_text = traceback.format_exc()
            exception_type = type(exc).__name__
            safe_message = sanitize_secret_text(str(exc)) or exception_type
            safe_traceback_excerpt = sanitize_secret_text(
                "\n".join(traceback_text.strip().splitlines()[-20:])[-4000:]
            ) or ""
            error_summary = f"{stage}: {exception_type}: {safe_message}"
            errors.append(error_summary)
            error_artifacts = _existing_artifacts_only(artifacts)
            error_context = {
                "stage": stage,
                "exception_type": exception_type,
                "message": safe_message,
                "traceback_excerpt": safe_traceback_excerpt,
                "requested_ticker": requested_ticker,
                "resolved_ticker": resolved_ticker,
                "provider_symbol": provider_symbol,
                "run_log_path": str(run_dirs.run_log_path.resolve()),
            }
            runtime_logger.exception(
                "Ticker export failed.",
                extra={"stage": stage, "ticker": requested_ticker},
            )
            result = TickerResult(
                ticker=resolved_ticker,
                requested_ticker=requested_ticker,
                resolved_ticker=resolved_ticker,
                status="error",
                qlib_compatible=False,
                validation_outcome="failure",
                columns=[],
                status_reasons=list(errors),
                neutral_notes=neutral_notes,
                warnings=material_warnings,
                errors=errors,
                provider_symbol=provider_symbol,
                provider_warnings=provider_warnings,
                error_context=error_context,
                run_log_path=run_dirs.run_log_path,
                artifacts=error_artifacts,
            )
            error_payload = {
                "generated_at_utc": utc_now_iso(),
                "run_id": run_dirs.run_id,
                "ticker": requested_ticker,
                "status": "error",
                "validation_outcome": "failure",
                "request": request.to_dict(),
                "warnings": material_warnings,
                "neutral_notes": neutral_notes,
                "errors": errors,
                "requested_ticker": requested_ticker,
                "resolved_ticker": resolved_ticker,
                "provider_symbol": provider_symbol,
                "provider_warnings": provider_warnings,
                "run_log_path": str(run_dirs.run_log_path.resolve()),
                "error": error_context,
                "stage": stage,
                "exception_type": exception_type,
                "message": safe_message,
                "traceback_excerpt": error_context["traceback_excerpt"],
                "artifacts": error_artifacts.to_dict(),
            }
            write_json(artifacts.manifest, build_ticker_manifest(result), temp_dir=run_dirs.temp_dir)
            write_json(artifacts.meta, error_payload, temp_dir=run_dirs.temp_dir)
            return result
