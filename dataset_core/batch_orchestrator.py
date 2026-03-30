from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from dataset_core.contracts import DatasetRequest, ExternalValidationConfig
from dataset_core.export_service import DatasetExportService
from dataset_core.factor_policy import resolve_provider_flags
from dataset_core.logging_runtime import bind_runtime_logger, close_run_logger, configure_run_logger
from dataset_core.manifest_service import build_batch_manifest, render_batch_manifest_text
from dataset_core.reference_adapters import CSVReferenceAdapter, ManualEventAdapter
from dataset_core.result_models import BatchResult
from dataset_core.serialization import cleanup_orphan_temp_files, write_json, write_text
from dataset_core.validation_external import ExternalValidationService
from providers.market_context import ContextResolver


@dataclass(frozen=True)
class BatchPlanItem:
    index: int
    requested_ticker: str
    resolved_ticker: str
    context: object | None = None


@dataclass
class BatchRuntime:
    context_resolver: ContextResolver
    provider_session: object | None
    auto_adjust: bool
    actions: bool
    factor_warnings: list[str]
    metadata_timeout: float | None


class BatchOrchestrator:
    def __init__(self, export_service: DatasetExportService | None = None) -> None:
        self.export_service = export_service or DatasetExportService()

    def _external_validation_service(
        self,
        config: ExternalValidationConfig,
    ) -> ExternalValidationService:
        adapters = []
        if config.reference_dir is not None:
            adapters.append(CSVReferenceAdapter(config.reference_dir))
        if config.manual_events_file is not None:
            adapters.append(ManualEventAdapter(config.manual_events_file))
        return ExternalValidationService(adapters=adapters)

    @staticmethod
    def _normalize_execution_mode(execution_mode: str) -> str:
        normalized = str(execution_mode or "concurrent").strip().lower()
        if normalized not in {"concurrent", "sequential"}:
            raise ValueError("execution_mode must be either 'concurrent' or 'sequential'.")
        return normalized

    def _build_runtime(self, request: DatasetRequest, export_service: DatasetExportService) -> BatchRuntime:
        metadata_timeout = request.provider.metadata_timeout or request.provider.timeout
        context_resolver = ContextResolver(metadata_timeout=metadata_timeout)
        create_session = getattr(export_service.acquisition_service, "create_session", None)
        provider_session = create_session(request) if callable(create_session) else None
        auto_adjust, actions, factor_warnings = resolve_provider_flags(
            auto_adjust=request.auto_adjust,
            actions=request.actions,
            requires_factor=request.requires_factor,
        )
        return BatchRuntime(
            context_resolver=context_resolver,
            provider_session=provider_session,
            auto_adjust=auto_adjust,
            actions=actions,
            factor_warnings=list(factor_warnings),
            metadata_timeout=metadata_timeout,
        )

    def _plan_batch(
        self,
        export_service: DatasetExportService,
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_logger,
    ) -> list[BatchPlanItem]:
        plans: list[BatchPlanItem] = []
        for index, ticker in enumerate(request.tickers, start=1):
            requested_ticker = str(ticker or "").strip().upper()
            try:
                context = export_service.resolve_context(
                    requested_ticker,
                    request,
                    context_resolver=runtime.context_resolver,
                )
                resolved_ticker = str(context.preferred_symbol or requested_ticker).upper()
            except Exception as exc:
                bind_runtime_logger(run_logger, ticker=requested_ticker, stage="plan_context").warning(
                    "Pre-planning context resolution failed and will fall back to per-ticker handling: %s",
                    exc,
                )
                context = None
                resolved_ticker = requested_ticker

            plans.append(
                BatchPlanItem(
                    index=index,
                    requested_ticker=requested_ticker,
                    resolved_ticker=resolved_ticker,
                    context=context,
                )
            )
        return plans

    @staticmethod
    def _unique_resolved_symbols(plans: list[BatchPlanItem]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for plan in plans:
            if plan.context is None:
                continue
            if plan.resolved_ticker in seen:
                continue
            seen.add(plan.resolved_ticker)
            ordered.append(plan.resolved_ticker)
        return ordered

    def _fetch_individually(
        self,
        export_service: DatasetExportService,
        symbols: list[str],
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_logger,
    ) -> dict[str, object]:
        results: dict[str, object] = {}
        for symbol in symbols:
            try:
                results[symbol] = export_service.acquisition_service.fetch(
                    symbol=symbol,
                    request=request,
                    auto_adjust=runtime.auto_adjust,
                    actions=runtime.actions,
                    session=runtime.provider_session,
                )
            except Exception as exc:
                bind_runtime_logger(run_logger, ticker=symbol, stage="batch_acquisition").warning(
                    "Prefetch acquisition failed and will fall back to per-ticker execution: %s",
                    exc,
                )
        return results

    def _acquire_batch(
        self,
        export_service: DatasetExportService,
        request: DatasetRequest,
        runtime: BatchRuntime,
        plans: list[BatchPlanItem],
        execution_mode: str,
        run_logger,
    ) -> dict[str, object]:
        resolved_symbols = self._unique_resolved_symbols(plans)
        if not resolved_symbols:
            return {}

        if execution_mode == "sequential":
            bind_runtime_logger(run_logger, stage="batch_acquisition").info(
                "Running sequential acquisition for %s resolved tickers.",
                len(resolved_symbols),
            )
            return self._fetch_individually(export_service, resolved_symbols, request, runtime, run_logger)

        bind_runtime_logger(run_logger, stage="batch_acquisition").info(
            "Running grouped acquisition for %s resolved tickers with a shared provider session.",
            len(resolved_symbols),
        )
        try:
            return export_service.acquisition_service.fetch_many(
                resolved_symbols,
                request,
                runtime.auto_adjust,
                runtime.actions,
                session=runtime.provider_session,
            )
        except Exception:
            bind_runtime_logger(run_logger, stage="batch_acquisition").warning(
                "Grouped acquisition failed; falling back to sequential per-ticker fetch.",
                exc_info=True,
            )
            return self._fetch_individually(export_service, resolved_symbols, request, runtime, run_logger)

    def run(
        self,
        request: DatasetRequest,
        progress_callback: Callable[[int, int, str], None] | None = None,
        *,
        execution_mode: str = "concurrent",
    ) -> BatchResult:
        normalized_execution_mode = self._normalize_execution_mode(execution_mode)
        export_service = self.export_service
        if request.external_validation.reference_dir or request.external_validation.manual_events_file:
            export_service = DatasetExportService(
                acquisition_service=export_service.acquisition_service,
                schema_builder=export_service.schema_builder,
                general_sanitizer=export_service.general_sanitizer,
                qlib_sanitizer=export_service.qlib_sanitizer,
                internal_validation=export_service.internal_validation,
                external_validation=self._external_validation_service(request.external_validation),
            )

        run_dirs = export_service.prepare_run_directories(request)
        run_logger_handle = configure_run_logger(run_dirs.run_id, run_dirs.workspace_root)
        bind_runtime_logger(run_logger_handle.logger, stage="run_start").info(
            "Starting batch export for %s tickers in execution_mode=%s.",
            len(request.tickers),
            normalized_execution_mode,
        )
        try:
            runtime = self._build_runtime(request, export_service)
            plans = self._plan_batch(export_service, request, runtime, run_logger_handle.logger)
            prefetched_results = self._acquire_batch(
                export_service=export_service,
                request=request,
                runtime=runtime,
                plans=plans,
                execution_mode=normalized_execution_mode,
                run_logger=run_logger_handle.logger,
            )

            results = []
            total = len(plans)
            for plan in plans:
                if progress_callback is not None:
                    progress_callback(plan.index, total, plan.requested_ticker)
                results.append(
                    export_service.export_ticker(
                        ticker=plan.requested_ticker,
                        request=request,
                        run_dirs=run_dirs,
                        prefetched_context=plan.context,
                        prefetched_fetch_result=prefetched_results.get(plan.resolved_ticker),
                        provider_session=runtime.provider_session,
                        context_resolver=runtime.context_resolver,
                        resolved_provider_flags=(
                            runtime.auto_adjust,
                            runtime.actions,
                            list(runtime.factor_warnings),
                        ),
                    )
                )

            batch_id = f"batch_{uuid4().hex[:12]}"
            manifest_json_path = run_dirs.output_root / "manifest_batch.json"
            manifest_txt_path = run_dirs.output_root / "manifest_batch.txt"
            batch_result = BatchResult(
                batch_id=batch_id,
                run_id=run_dirs.run_id,
                output_root=run_dirs.output_root,
                csv_dir=run_dirs.primary_csv_dir(request),
                meta_dir=run_dirs.meta_dir,
                report_dir=run_dirs.report_dir,
                manifest_json_path=manifest_json_path,
                manifest_txt_path=manifest_txt_path,
                run_log_path=run_dirs.run_log_path,
                results=results,
            )
            manifest = build_batch_manifest(request=request, batch_result=batch_result)
            write_json(manifest_json_path, manifest, temp_dir=run_dirs.temp_dir)
            write_text(manifest_txt_path, render_batch_manifest_text(manifest), temp_dir=run_dirs.temp_dir)
            bind_runtime_logger(run_logger_handle.logger, stage="batch_finish").info(
                "Completed batch export with status counts: success=%s warning=%s error=%s.",
                batch_result.status_counts.get("success", 0),
                batch_result.status_counts.get("warning", 0),
                batch_result.status_counts.get("error", 0),
            )
            return batch_result
        except Exception:
            bind_runtime_logger(run_logger_handle.logger, stage="batch_failure").exception("Batch export failed.")
            raise
        finally:
            try:
                removed_tmp_files = cleanup_orphan_temp_files(run_dirs.workspace_root)
                if removed_tmp_files:
                    bind_runtime_logger(run_logger_handle.logger, stage="tmp_cleanup").info(
                        "Removed %s orphan temporary file(s).",
                        len(removed_tmp_files),
                    )
            except Exception:
                bind_runtime_logger(run_logger_handle.logger, stage="tmp_cleanup").warning(
                    "Failed to cleanup orphan temporary files.",
                    exc_info=True,
                )
            close_run_logger(run_dirs.run_id)
