from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.datetime64):
            return str(value)
    except Exception:
        pass

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
    shutil.copyfile(source, sibling_temp)
    os.replace(str(sibling_temp), str(destination))
    source.unlink(missing_ok=True)


def _temp_path(path: Path, temp_dir: Path | None = None) -> Path:
    root = path.parent if temp_dir is None else Path(temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{path.name}.{uuid4().hex}.tmp"


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def write_json(path: Path, payload: dict[str, Any], temp_dir: Path | None = None) -> None:
    content = json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=False)
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
