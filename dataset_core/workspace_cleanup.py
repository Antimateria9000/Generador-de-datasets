from __future__ import annotations

import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

from dataset_core.logging_runtime import close_run_logger
from dataset_core.settings import ensure_workspace_tree
from dataset_core.workspace_inventory import WorkspaceRunRecord, filter_workspace_runs, list_workspace_runs

_RUN_COMPONENT_KEYS = ("runs", "exports", "manifests", "reports", "temp", "logs")


def _handle_remove_error(func, path, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _assert_within_workspace(workspace_root: Path, target: Path) -> None:
    resolved_root = workspace_root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
        raise ValueError(f"Unsafe workspace cleanup target: {resolved_target}")


def _component_paths_for_run(workspace_root: Path, run_id: str) -> dict[str, Path]:
    workspace = ensure_workspace_tree(workspace_root)
    return {key: workspace[key] / run_id for key in _RUN_COMPONENT_KEYS}


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _unlink_with_retries(path: Path, retries: int = 20, delay_seconds: float = 0.1) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            pass
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)
        except OSError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def _rmdir_with_retries(path: Path, retries: int = 20, delay_seconds: float = 0.1) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            path.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        _unlink_with_retries(path)
        return

    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            _unlink_with_retries(child)
    _rmdir_with_retries(path)


@dataclass(frozen=True)
class CleanupResult:
    run_ids: list[str] = field(default_factory=list)
    removed_paths: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    bytes_reclaimed: int = 0
    dry_run: bool = False


def select_runs_for_cleanup(
    workspace_root: Path | None = None,
    *,
    run_ids: list[str] | None = None,
    older_than_days: int | None = None,
    orphans: bool = False,
    all_runs: bool = False,
) -> list[WorkspaceRunRecord]:
    inventory = list_workspace_runs(workspace_root)
    selected: dict[str, WorkspaceRunRecord] = {}

    normalized_run_ids = [str(run_id).strip() for run_id in (run_ids or []) if str(run_id).strip()]
    for record in inventory:
        if record.run_id in normalized_run_ids:
            selected[record.run_id] = record

    if older_than_days is not None:
        for record in filter_workspace_runs(inventory, older_than_days=older_than_days):
            selected[record.run_id] = record

    if orphans:
        for record in filter_workspace_runs(inventory, orphans_only=True):
            selected[record.run_id] = record

    if all_runs:
        for record in inventory:
            selected[record.run_id] = record

    return sorted(selected.values(), key=lambda item: item.run_id, reverse=True)


def cleanup_runs(
    workspace_root: Path | None = None,
    *,
    run_ids: list[str],
    dry_run: bool = False,
) -> CleanupResult:
    workspace = ensure_workspace_tree(workspace_root)
    root = workspace["workspace_root"]
    removed_paths: list[str] = []
    missing_paths: list[str] = []
    bytes_reclaimed = 0

    for run_id in [str(item).strip() for item in run_ids if str(item).strip()]:
        close_run_logger(run_id)
        component_paths = _component_paths_for_run(root, run_id)
        for path in component_paths.values():
            _assert_within_workspace(root, path)
            if not path.exists():
                missing_paths.append(str(path.resolve()))
                continue

            bytes_reclaimed += _path_size(path)
            removed_paths.append(str(path.resolve()))
            if dry_run:
                continue

            if path.is_dir():
                _remove_tree(path)
            else:
                _unlink_with_retries(path)

    return CleanupResult(
        run_ids=[str(item).strip() for item in run_ids if str(item).strip()],
        removed_paths=removed_paths,
        missing_paths=missing_paths,
        bytes_reclaimed=bytes_reclaimed,
        dry_run=dry_run,
    )
