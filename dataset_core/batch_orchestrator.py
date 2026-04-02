from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import uuid4

from dataset_core.contracts import DatasetRequest
from dataset_core.external_sources.factory import build_external_validation_service
from dataset_core.export_service import DatasetExportService
from dataset_core.factor_policy import resolve_provider_flags
from dataset_core.logging_runtime import bind_runtime_logger, close_run_logger, configure_run_logger
from dataset_core.manifest_service import build_batch_manifest, render_batch_manifest_text
from dataset_core.result_models import BatchResult
from dataset_core.serialization import cleanup_orphan_temp_files, write_json, write_text
from dataset_core.settings import DEFAULT_METADATA_CANDIDATE_LIMIT, resolve_effective_cache_paths
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
    batch_max_workers: int
    batch_chunk_size: int


class BatchOrchestrator:
    def __init__(self, export_service: DatasetExportService | None = None) -> None:
        self.export_service = export_service or DatasetExportService()

    @staticmethod
    def _normalize_execution_mode(execution_mode: str) -> str:
        normalized = str(execution_mode or "concurrent").strip().lower()
        if normalized not in {"concurrent", "sequential"}:
            raise ValueError("execution_mode must be either 'concurrent' or 'sequential'.")
        return normalized

    @staticmethod
    def _effective_batch_workers(request: DatasetRequest, provider_session: object | None) -> int:
        configured_workers = request.provider.batch_max_workers or request.provider.max_workers
        if configured_workers is not None:
            return max(1, int(configured_workers))

        provider = getattr(provider_session, "provider", None)
        provider_workers = getattr(provider, "max_workers", None)
        if provider_workers is not None:
            return max(1, int(provider_workers))
        return 4

    @staticmethod
    def _effective_chunk_size(request: DatasetRequest, batch_max_workers: int) -> int:
        configured_chunk_size = request.provider.batch_chunk_size
        if configured_chunk_size is not None:
            return max(1, int(configured_chunk_size))
        return max(1, int(batch_max_workers) * 4)

    def _build_runtime(
        self,
        request: DatasetRequest,
        export_service: DatasetExportService,
        *,
        cache_namespace: str | None = None,
    ) -> BatchRuntime:
        metadata_timeout = request.provider.metadata_timeout or request.provider.timeout
        cache_paths = resolve_effective_cache_paths(request.output_dir, request.provider.cache_dir)
        create_session = getattr(export_service.acquisition_service, "create_session", None)
        provider_session = None
        if callable(create_session):
            try:
                provider_session = create_session(request, cache_namespace=cache_namespace)
            except TypeError:
                provider_session = create_session(request)
        batch_max_workers = self._effective_batch_workers(request, provider_session)
        batch_chunk_size = self._effective_chunk_size(request, batch_max_workers)
        context_resolver = ContextResolver(
            metadata_timeout=metadata_timeout,
            candidate_limit=request.provider.metadata_candidate_limit or DEFAULT_METADATA_CANDIDATE_LIMIT,
            cache_dir=cache_paths["market_context"],
            cache_ttl_seconds=request.provider.context_cache_ttl_seconds,
        )
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
            batch_max_workers=batch_max_workers,
            batch_chunk_size=batch_chunk_size,
        )

    def _plan_one(
        self,
        export_service: DatasetExportService,
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_logger,
        *,
        index: int,
        ticker: str,
    ) -> BatchPlanItem:
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

        return BatchPlanItem(
            index=index,
            requested_ticker=requested_ticker,
            resolved_ticker=resolved_ticker,
            context=context,
        )

    def _plan_batch(
        self,
        export_service: DatasetExportService,
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_logger,
        execution_mode: str,
    ) -> list[BatchPlanItem]:
        items = list(enumerate(request.tickers, start=1))
        if execution_mode == "sequential" or runtime.batch_max_workers <= 1 or len(items) <= 1:
            return [
                self._plan_one(
                    export_service,
                    request,
                    runtime,
                    run_logger,
                    index=index,
                    ticker=ticker,
                )
                for index, ticker in items
            ]

        bind_runtime_logger(run_logger, stage="plan_context").info(
            "Planning context concurrently for %s tickers with max_workers=%s.",
            len(items),
            runtime.batch_max_workers,
        )
        planned: dict[int, BatchPlanItem] = {}
        with ThreadPoolExecutor(max_workers=min(runtime.batch_max_workers, len(items))) as executor:
            futures = {
                executor.submit(
                    self._plan_one,
                    export_service,
                    request,
                    runtime,
                    run_logger,
                    index=index,
                    ticker=ticker,
                ): index
                for index, ticker in items
            }
            for future in as_completed(futures):
                plan = future.result()
                planned[plan.index] = plan
        return [planned[index] for index, _ in items]

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

    @staticmethod
    def _chunk_symbols(symbols: list[str], chunk_size: int) -> list[list[str]]:
        size = max(1, int(chunk_size))
        return [symbols[offset : offset + size] for offset in range(0, len(symbols), size)]

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

    def _fetch_chunk(
        self,
        export_service: DatasetExportService,
        symbols: list[str],
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_logger,
    ) -> dict[str, object]:
        try:
            return export_service.acquisition_service.fetch_many(
                symbols,
                request,
                runtime.auto_adjust,
                runtime.actions,
                session=runtime.provider_session,
            )
        except Exception:
            bind_runtime_logger(run_logger, stage="batch_acquisition").warning(
                "Grouped acquisition failed for chunk=%s; falling back to sequential fetch for that chunk.",
                ",".join(symbols),
                exc_info=True,
            )
            return self._fetch_individually(export_service, symbols, request, runtime, run_logger)

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

        chunks = self._chunk_symbols(resolved_symbols, runtime.batch_chunk_size)
        bind_runtime_logger(run_logger, stage="batch_acquisition").info(
            "Running grouped acquisition for %s resolved tickers with a shared provider session across %s chunk(s).",
            len(resolved_symbols),
            len(chunks),
        )
        if len(chunks) == 1 or runtime.batch_max_workers <= 1:
            return self._fetch_chunk(export_service, chunks[0], request, runtime, run_logger)

        prefetched_results: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=min(runtime.batch_max_workers, len(chunks))) as executor:
            futures = {
                executor.submit(
                    self._fetch_chunk,
                    export_service,
                    chunk,
                    request,
                    runtime,
                    run_logger,
                ): tuple(chunk)
                for chunk in chunks
            }
            for future in as_completed(futures):
                prefetched_results.update(future.result())
        return prefetched_results

    def _export_plan(
        self,
        export_service: DatasetExportService,
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_dirs,
        prefetched_results: dict[str, object],
        plan: BatchPlanItem,
    ):
        result = export_service.export_ticker(
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
        return plan.index, result

    def _finalize_batch(
        self,
        export_service: DatasetExportService,
        request: DatasetRequest,
        runtime: BatchRuntime,
        run_dirs,
        plans: list[BatchPlanItem],
        prefetched_results: dict[str, object],
        execution_mode: str,
        progress_callback: Callable[[int, int, str], None] | None,
        run_logger,
    ) -> list[object]:
        total = len(plans)
        if execution_mode == "sequential" or runtime.batch_max_workers <= 1 or total <= 1:
            results = []
            for plan in plans:
                if progress_callback is not None:
                    progress_callback(plan.index, total, plan.requested_ticker)
                _, result = self._export_plan(
                    export_service,
                    request,
                    runtime,
                    run_dirs,
                    prefetched_results,
                    plan,
                )
                results.append(result)
            return results

        bind_runtime_logger(run_logger, stage="batch_finalize").info(
            "Finalizing ticker exports concurrently for %s tickers with max_workers=%s.",
            total,
            runtime.batch_max_workers,
        )
        finalized: dict[int, object] = {}
        completed = 0
        with ThreadPoolExecutor(max_workers=min(runtime.batch_max_workers, total)) as executor:
            futures = {
                executor.submit(
                    self._export_plan,
                    export_service,
                    request,
                    runtime,
                    run_dirs,
                    prefetched_results,
                    plan,
                ): plan
                for plan in plans
            }
            for future in as_completed(futures):
                index, result = future.result()
                finalized[index] = result
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, result.requested_ticker)
        return [finalized[plan.index] for plan in plans]

    def run(
        self,
        request: DatasetRequest,
        progress_callback: Callable[[int, int, str], None] | None = None,
        *,
        execution_mode: str = "concurrent",
    ) -> BatchResult:
        normalized_execution_mode = self._normalize_execution_mode(execution_mode)
        export_service = self.export_service
        if request.external_validation.is_enabled():
            export_service = DatasetExportService(
                acquisition_service=export_service.acquisition_service,
                schema_builder=export_service.schema_builder,
                general_sanitizer=export_service.general_sanitizer,
                qlib_sanitizer=export_service.qlib_sanitizer,
                internal_validation=export_service.internal_validation,
                external_validation=build_external_validation_service(
                    request.external_validation,
                    output_root=request.output_dir,
                ),
            )

        run_dirs = export_service.prepare_run_directories(request)
        run_logger_handle = configure_run_logger(run_dirs.run_id, run_dirs.workspace_root)
        bind_runtime_logger(run_logger_handle.logger, stage="run_start").info(
            "Starting batch export for %s tickers in execution_mode=%s.",
            len(request.tickers),
            normalized_execution_mode,
        )
        try:
            runtime = self._build_runtime(request, export_service, cache_namespace=run_dirs.run_id)
            plans = self._plan_batch(
                export_service,
                request,
                runtime,
                run_logger_handle.logger,
                normalized_execution_mode,
            )
            prefetched_results = self._acquire_batch(
                export_service=export_service,
                request=request,
                runtime=runtime,
                plans=plans,
                execution_mode=normalized_execution_mode,
                run_logger=run_logger_handle.logger,
            )
            results = self._finalize_batch(
                export_service=export_service,
                request=request,
                runtime=runtime,
                run_dirs=run_dirs,
                plans=plans,
                prefetched_results=prefetched_results,
                execution_mode=normalized_execution_mode,
                progress_callback=progress_callback,
                run_logger=run_logger_handle.logger,
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
