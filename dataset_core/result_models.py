from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _path_or_none(value: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    return str(value.resolve())


@dataclass
class ArtifactPaths:
    csv: Optional[Path] = None
    canonical_csv: Optional[Path] = None
    qlib_csv: Optional[Path] = None
    meta: Optional[Path] = None
    dq: Optional[Path] = None
    external_json: Optional[Path] = None
    external_txt: Optional[Path] = None
    manifest: Optional[Path] = None
    qlib_report: Optional[Path] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "csv_path": _path_or_none(self.csv),
            "canonical_csv_path": _path_or_none(self.canonical_csv),
            "qlib_csv_path": _path_or_none(self.qlib_csv),
            "meta_path": _path_or_none(self.meta),
            "dq_path": _path_or_none(self.dq),
            "external_validation_json_path": _path_or_none(self.external_json),
            "external_validation_txt_path": _path_or_none(self.external_txt),
            "manifest_path": _path_or_none(self.manifest),
            "qlib_report_path": _path_or_none(self.qlib_report),
        }


@dataclass
class TickerResult:
    ticker: str
    requested_ticker: str
    resolved_ticker: str
    status: str
    qlib_compatible: bool
    columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    internal_validation_status: Optional[str] = None
    external_validation_status: Optional[str] = None
    factor_policy: Optional[str] = None
    provider_symbol: Optional[str] = None
    provider_warnings: list[str] = field(default_factory=list)
    error_context: Optional[dict[str, object]] = None
    run_log_path: Optional[Path] = None
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "ticker": self.ticker,
            "requested_ticker": self.requested_ticker,
            "resolved_ticker": self.resolved_ticker,
            "status": self.status,
            "qlib_compatible": self.qlib_compatible,
            "columns": list(self.columns),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "internal_validation_status": self.internal_validation_status,
            "external_validation_status": self.external_validation_status,
            "factor_policy": self.factor_policy,
            "provider_symbol": self.provider_symbol,
            "provider_warnings": list(self.provider_warnings),
            "error_context": None if self.error_context is None else dict(self.error_context),
            "run_log_path": _path_or_none(self.run_log_path),
        }
        payload.update(self.artifacts.to_dict())
        return payload


@dataclass
class BatchResult:
    batch_id: str
    run_id: str
    output_root: Path
    csv_dir: Path
    meta_dir: Path
    report_dir: Path
    manifest_json_path: Path
    manifest_txt_path: Path
    run_log_path: Optional[Path] = None
    results: list[TickerResult] = field(default_factory=list)

    @property
    def status_counts(self) -> dict[str, int]:
        counts = {"success": 0, "warning": 0, "error": 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "output_root": str(self.output_root.resolve()),
            "csv_dir": str(self.csv_dir.resolve()),
            "meta_dir": str(self.meta_dir.resolve()),
            "report_dir": str(self.report_dir.resolve()),
            "manifest_json_path": str(self.manifest_json_path.resolve()),
            "manifest_txt_path": str(self.manifest_txt_path.resolve()),
            "run_log_path": _path_or_none(self.run_log_path),
            "status_counts": self.status_counts,
            "results": [result.to_dict() for result in self.results],
        }
