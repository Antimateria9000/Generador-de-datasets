from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_core.settings import ensure_workspace_tree
from dataset_core.workspace_cleanup import cleanup_runs, select_runs_for_cleanup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Dataset Factory workspace selectively and safely."
    )
    parser.add_argument("--workspace-root", default=None, type=str, help="Workspace root override.")
    parser.add_argument("--run-id", action="append", default=[], help="Run identifier to remove. Repeatable.")
    parser.add_argument("--older-than-days", default=None, type=int, help="Remove runs older than N days.")
    parser.add_argument("--orphans", action="store_true", help="Remove orphaned or incomplete runs.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove every run under the workspace. Requires --confirm-all DELETE.",
    )
    parser.add_argument(
        "--confirm-all",
        default="",
        type=str,
        help="Explicit confirmation token for --all. Use exactly DELETE.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting it.")
    return parser


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(size_bytes, 0))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size_bytes}B"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = None if not args.workspace_root else Path(args.workspace_root).expanduser().resolve()
    workspace = ensure_workspace_tree(workspace_root)

    if args.all and args.confirm_all != "DELETE":
        raise SystemExit("--all requires --confirm-all DELETE")
    if args.older_than_days is not None and int(args.older_than_days) < 0:
        raise SystemExit("--older-than-days must be >= 0")

    selected_runs = select_runs_for_cleanup(
        workspace["workspace_root"],
        run_ids=args.run_id,
        older_than_days=args.older_than_days,
        orphans=bool(args.orphans),
        all_runs=bool(args.all),
    )
    if not selected_runs:
        print("No runs matched the cleanup criteria.")
        return 0

    result = cleanup_runs(
        workspace["workspace_root"],
        run_ids=[record.run_id for record in selected_runs],
        dry_run=bool(args.dry_run),
    )

    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {len(result.run_ids)} run(s) from {workspace['workspace_root']}")
    print(f"Bytes reclaimed: {_format_bytes(result.bytes_reclaimed)}")
    for run_id in result.run_ids:
        print(f"- {run_id}")
    if result.missing_paths:
        print("Missing paths:")
        for path in result.missing_paths:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
