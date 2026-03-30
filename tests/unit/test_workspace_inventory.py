from __future__ import annotations

from pathlib import Path

import pytest

from dataset_core.workspace_inventory import WorkspaceRunRecord, filter_workspace_runs, list_workspace_runs


def _component_paths(root: Path) -> dict[str, Path]:
    return {
        "runs": root / "runs",
        "exports": root / "exports",
        "manifests": root / "manifests",
        "reports": root / "reports",
        "temp": root / "temp",
        "logs": root / "logs",
    }


def test_filter_workspace_runs_created_to_is_inclusive_for_calendar_day(tmp_path):
    record = WorkspaceRunRecord(
        run_id="run-1",
        workspace_root=tmp_path,
        created_at_utc="2026-03-29T18:45:00+00:00",
        preset="base",
        interval="1d",
        tickers=["MSFT"],
        status_counts={"success": 1},
        overall_status="success",
        component_paths=_component_paths(tmp_path),
    )

    filtered = filter_workspace_runs(
        [record],
        created_to="2026-03-29T00:00:00",
    )

    assert filtered == [record]


def test_filter_workspace_runs_rejects_negative_older_than_days(tmp_path):
    record = WorkspaceRunRecord(
        run_id="run-1",
        workspace_root=tmp_path,
        created_at_utc="2026-03-29T18:45:00+00:00",
        preset="base",
        interval="1d",
        tickers=["MSFT"],
        status_counts={"success": 1},
        overall_status="success",
        component_paths=_component_paths(tmp_path),
    )

    with pytest.raises(ValueError, match="older_than_days must be >= 0"):
        filter_workspace_runs([record], older_than_days=-1)


def test_list_workspace_runs_does_not_materialize_missing_workspace(tmp_path):
    missing_root = tmp_path / "missing-workspace"

    records = list_workspace_runs(missing_root)

    assert records == []
    assert not missing_root.exists()
