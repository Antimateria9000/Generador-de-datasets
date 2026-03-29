from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dataset_core.settings import ensure_workspace_tree

_RUN_COMPONENT_KEYS = ("runs", "exports", "manifests", "reports", "temp", "logs")


def _safe_timestamp(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _resolve_component_paths(workspace_root: Path, run_id: str) -> dict[str, Path]:
    workspace = ensure_workspace_tree(workspace_root)
    return {key: workspace[key] / run_id for key in _RUN_COMPONENT_KEYS}


def _iter_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return [candidate for candidate in path.rglob("*") if candidate.is_file()]


def _directory_size(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in _iter_files(path))


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(size_bytes, 0))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size_bytes}B"


@dataclass(frozen=True)
class WorkspaceRunRecord:
    run_id: str
    workspace_root: Path
    created_at_utc: str | None
    preset: str | None
    interval: str | None
    tickers: list[str] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    overall_status: str = "unknown"
    size_bytes: int = 0
    size_human: str = "0B"
    age_days: int | None = None
    orphaned: bool = False
    missing_components: list[str] = field(default_factory=list)
    component_paths: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "preset": self.preset,
            "interval": self.interval,
            "tickers": list(self.tickers),
            "ticker_summary": ", ".join(self.tickers),
            "status_counts": dict(self.status_counts),
            "overall_status": self.overall_status,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "age_days": self.age_days,
            "orphaned": self.orphaned,
            "missing_components": list(self.missing_components),
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path.resolve()),
            "runs_path": str(self.component_paths["runs"].resolve()),
            "exports_path": str(self.component_paths["exports"].resolve()),
            "reports_path": str(self.component_paths["reports"].resolve()),
            "manifests_path": str(self.component_paths["manifests"].resolve()),
            "temp_path": str(self.component_paths["temp"].resolve()),
            "logs_path": str(self.component_paths["logs"].resolve()),
        }


def list_workspace_runs(workspace_root: Path | None = None) -> list[WorkspaceRunRecord]:
    workspace = ensure_workspace_tree(workspace_root)
    root = workspace["workspace_root"]
    run_ids: set[str] = set()

    for key in _RUN_COMPONENT_KEYS:
        for candidate in workspace[key].iterdir():
            if candidate.name.startswith("."):
                continue
            run_ids.add(candidate.name)

    records: list[WorkspaceRunRecord] = []
    now_utc = pd.Timestamp.now(tz="UTC")
    for run_id in sorted(run_ids, reverse=True):
        component_paths = _resolve_component_paths(root, run_id)
        manifest_path = component_paths["runs"] / "manifest_batch.json"
        manifest = _read_json(manifest_path)
        generated_at = None
        preset = None
        interval = None
        tickers: list[str] = []
        status_counts: dict[str, int] = {}
        if manifest is not None:
            generated_at = manifest.get("generated_at_utc")
            request = manifest.get("request", {}) if isinstance(manifest.get("request"), dict) else {}
            preset = request.get("mode")
            interval = request.get("interval")
            tickers = [str(item) for item in request.get("tickers", []) if str(item).strip()]
            status_counts = {
                str(key): int(value)
                for key, value in (manifest.get("status_counts", {}) or {}).items()
                if str(key).strip()
            }

        missing_components = [key for key, path in component_paths.items() if not path.exists()]
        orphaned = manifest is None or bool(missing_components)
        size_bytes = sum(_directory_size(path) for path in component_paths.values() if path.exists())

        created_timestamp = _safe_timestamp(generated_at)
        if created_timestamp is None:
            fallback_path = next((path for path in component_paths.values() if path.exists()), None)
            if fallback_path is not None:
                created_timestamp = pd.Timestamp(fallback_path.stat().st_mtime, unit="s", tz="UTC")

        age_days = None
        if created_timestamp is not None:
            age_days = max(0, int((now_utc - created_timestamp).total_seconds() // 86400))

        overall_status = "unknown"
        if status_counts:
            if status_counts.get("error", 0) > 0:
                overall_status = "error"
            elif status_counts.get("warning", 0) > 0:
                overall_status = "warning"
            elif status_counts.get("success", 0) > 0:
                overall_status = "success"
        elif orphaned:
            overall_status = "orphan"

        records.append(
            WorkspaceRunRecord(
                run_id=run_id,
                workspace_root=root,
                created_at_utc=None if created_timestamp is None else created_timestamp.isoformat(),
                preset=preset,
                interval=interval,
                tickers=tickers,
                status_counts=status_counts,
                overall_status=overall_status,
                size_bytes=size_bytes,
                size_human=_format_bytes(size_bytes),
                age_days=age_days,
                orphaned=orphaned,
                missing_components=missing_components,
                component_paths=component_paths,
                manifest_path=manifest_path if manifest_path.exists() else None,
            )
        )

    return records


def filter_workspace_runs(
    runs: list[WorkspaceRunRecord],
    *,
    ticker: str | None = None,
    preset: str | None = None,
    interval: str | None = None,
    status: str | None = None,
    older_than_days: int | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    orphans_only: bool = False,
) -> list[WorkspaceRunRecord]:
    ticker_filter = str(ticker or "").strip().upper()
    preset_filter = str(preset or "").strip().lower()
    interval_filter = str(interval or "").strip().lower()
    status_filter = str(status or "").strip().lower()
    from_ts = _safe_timestamp(created_from)
    to_ts = _safe_timestamp(created_to)

    filtered: list[WorkspaceRunRecord] = []
    for record in runs:
        if ticker_filter and ticker_filter not in {ticker.upper() for ticker in record.tickers}:
            continue
        if preset_filter and str(record.preset or "").strip().lower() != preset_filter:
            continue
        if interval_filter and str(record.interval or "").strip().lower() != interval_filter:
            continue
        if status_filter and str(record.overall_status or "").strip().lower() != status_filter:
            continue
        if older_than_days is not None and (record.age_days is None or record.age_days < int(older_than_days)):
            continue
        if orphans_only and not record.orphaned:
            continue

        created_at = _safe_timestamp(record.created_at_utc)
        if from_ts is not None and (created_at is None or created_at < from_ts):
            continue
        if to_ts is not None and (created_at is None or created_at > to_ts):
            continue

        filtered.append(record)

    return filtered
