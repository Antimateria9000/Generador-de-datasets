from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_core.settings import ensure_workspace_tree


def _handle_remove_error(func, path, exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _clean_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, onerror=_handle_remove_error)
        else:
            os.chmod(child, stat.S_IWRITE)
            child.unlink()


def main() -> int:
    workspace = ensure_workspace_tree()
    for key, path in workspace.items():
        if key == "workspace_root":
            continue
        _clean_directory(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
