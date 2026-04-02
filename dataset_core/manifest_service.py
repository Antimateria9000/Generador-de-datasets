from __future__ import annotations

from dataset_core.contracts import DatasetRequest
from dataset_core.result_models import BatchResult, TickerResult
from dataset_core.settings import utc_now_iso


def build_ticker_manifest(result: TickerResult) -> dict[str, object]:
    payload = result.to_dict()
    payload["generated_at_utc"] = utc_now_iso()
    return payload


def build_batch_manifest(request: DatasetRequest, batch_result: BatchResult) -> dict[str, object]:
    return {
        "batch_id": batch_result.batch_id,
        "run_id": batch_result.run_id,
        "generated_at_utc": utc_now_iso(),
        "request": request.to_dict(),
        "ticker_count": request.batch_size,
        "output_root": str(batch_result.output_root.resolve()),
        "csv_dir": str(batch_result.csv_dir.resolve()),
        "meta_dir": str(batch_result.meta_dir.resolve()),
        "report_dir": str(batch_result.report_dir.resolve()),
        "manifest_json_path": str(batch_result.manifest_json_path.resolve()),
        "manifest_txt_path": str(batch_result.manifest_txt_path.resolve()),
        "run_log_path": None if batch_result.run_log_path is None else str(batch_result.run_log_path.resolve()),
        "status_counts": batch_result.status_counts,
        "validation_outcome_counts": batch_result.validation_outcome_counts,
        "results": [result.to_dict() for result in batch_result.results],
    }


def render_batch_manifest_text(manifest: dict[str, object]) -> str:
    counts = manifest.get("status_counts", {})
    validation_counts = manifest.get("validation_outcome_counts", {})
    lines = [
        f"Batch manifest: {manifest.get('batch_id')}",
        f"Run id: {manifest.get('run_id')}",
        f"Generated at: {manifest.get('generated_at_utc')}",
        f"Output root: {manifest.get('output_root')}",
        f"CSV dir: {manifest.get('csv_dir')}",
        f"Meta dir: {manifest.get('meta_dir')}",
        f"Report dir: {manifest.get('report_dir')}",
        f"Run log: {manifest.get('run_log_path')}",
        f"Success: {counts.get('success', 0)}",
        f"Warning: {counts.get('warning', 0)}",
        f"Error: {counts.get('error', 0)}",
        f"Validated success: {validation_counts.get('success_validated', 0)}",
        f"Partial validation: {validation_counts.get('success_partial_validation', 0)}",
        f"Validation failure: {validation_counts.get('failure', 0)}",
        "",
    ]

    for result in manifest.get("results", []):
        lines.append(
            f"- {result.get('ticker')}: status={result.get('status')} validation_outcome={result.get('validation_outcome')} qlib_compatible={result.get('qlib_compatible')}"
        )
        lines.append(f"  csv={result.get('csv_path')}")
        lines.append(f"  meta={result.get('meta_path')}")
        lines.append(f"  dq={result.get('dq_path')}")
        lines.append(f"  external={result.get('external_validation_json_path')}")
        if result.get("external_validation_status"):
            lines.append(
                "  external_status="
                f"{result.get('external_validation_status')} "
                f"(coverage={result.get('external_validation_coverage_status')} "
                f"comparison={result.get('external_validation_comparison_status')})"
            )
        if result.get("qlib_csv_path"):
            lines.append(f"  qlib_csv={result.get('qlib_csv_path')}")
        if result.get("qlib_report_path"):
            lines.append(f"  qlib_report={result.get('qlib_report_path')}")
        if result.get("status_reasons"):
            lines.append(f"  status_reasons={' | '.join(str(item) for item in result.get('status_reasons', []))}")
        if result.get("neutral_notes"):
            lines.append(f"  neutral_notes={' | '.join(str(item) for item in result.get('neutral_notes', []))}")
        warnings = result.get("warnings") or []
        errors = result.get("errors") or []
        if warnings:
            lines.append(f"  warnings={' | '.join(str(item) for item in warnings)}")
        if errors:
            lines.append(f"  errors={' | '.join(str(item) for item in errors)}")

    return "\n".join(lines).strip() + "\n"
