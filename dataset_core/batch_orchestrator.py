from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from dataset_core.contracts import DatasetRequest, ExternalValidationConfig
from dataset_core.export_service import DatasetExportService
from dataset_core.logging_runtime import bind_runtime_logger, close_run_logger, configure_run_logger
from dataset_core.manifest_service import build_batch_manifest, render_batch_manifest_text
from dataset_core.reference_adapters import CSVReferenceAdapter, ManualEventAdapter
from dataset_core.result_models import BatchResult
from dataset_core.serialization import cleanup_orphan_temp_files, write_json, write_text
from dataset_core.validation_external import ExternalValidationService


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

    def run(
        self,
        request: DatasetRequest,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchResult:
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
            "Starting batch export for %s tickers.",
            len(request.tickers),
        )
        try:
            results = []
            total = len(request.tickers)

            for index, ticker in enumerate(request.tickers, start=1):
                if progress_callback is not None:
                    progress_callback(index, total, ticker)
                results.append(export_service.export_ticker(ticker=ticker, request=request, run_dirs=run_dirs))

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
