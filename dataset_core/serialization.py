from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def _normalize_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _normalize_datetime(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return pd.Timestamp(value).isoformat()


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _normalize_float(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return _normalize_datetime(value)
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(column) for column in value.columns],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "name": None if value.name is None else str(value.name),
            "length": int(len(value)),
        }
    if isinstance(value, pd.Index):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [make_json_safe(item) for item in value]

    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return _normalize_float(float(value))
        if isinstance(value, np.ndarray):
            return [make_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.datetime64):
            return _normalize_datetime(pd.Timestamp(value))
        if isinstance(value, np.timedelta64):
            return str(pd.Timedelta(value))
        if isinstance(value, np.generic):
            return make_json_safe(value.item())
    except Exception:
        pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(str(source), str(destination))
        return
    except PermissionError:
        if source.parent == destination.parent:
            raise

    sibling_temp = destination.parent / f"{destination.name}.{uuid4().hex}.tmp"
    try:
        shutil.copyfile(source, sibling_temp)
        os.replace(str(sibling_temp), str(destination))
    finally:
        _safe_unlink(sibling_temp)
        _safe_unlink(source)


def _temp_path(path: Path, temp_dir: Path | None = None) -> Path:
    root = path.parent if temp_dir is None else Path(temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{path.name}.{uuid4().hex}.tmp"


def iter_orphan_temp_files(root: Path) -> list[Path]:
    workspace_root = Path(root).expanduser().resolve()
    if not workspace_root.exists():
        return []
    return [candidate for candidate in workspace_root.rglob("*.tmp") if candidate.is_file()]


def cleanup_orphan_temp_files(root: Path) -> list[Path]:
    removed: list[Path] = []
    for candidate in iter_orphan_temp_files(root):
        _safe_unlink(candidate)
        if not candidate.exists():
            removed.append(candidate)
    return removed


def write_json(path: Path, payload: dict[str, Any], temp_dir: Path | None = None) -> None:
    content = json.dumps(
        make_json_safe(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    )
    temp_path = _temp_path(path, temp_dir=temp_dir)
    temp_path.write_text(content, encoding="utf-8")
    try:
        _atomic_replace(temp_path, path)
    except PermissionError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        _safe_unlink(temp_path)


def write_text(path: Path, content: str, temp_dir: Path | None = None) -> None:
    temp_path = _temp_path(path, temp_dir=temp_dir)
    temp_path.write_text(str(content), encoding="utf-8")
    try:
        _atomic_replace(temp_path, path)
    except PermissionError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
        _safe_unlink(temp_path)


def write_csv(frame: pd.DataFrame, path: Path, temp_dir: Path | None = None) -> None:
    temp_path = _temp_path(path, temp_dir=temp_dir)
    frame.to_csv(temp_path, index=False, encoding="utf-8")
    try:
        _atomic_replace(temp_path, path)
    except PermissionError:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")
        _safe_unlink(temp_path)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
