from __future__ import annotations

import re
from pathlib import Path

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PATH_SEPARATOR_RE = re.compile(r"[\\/]+")


def _extract_cross_platform_basename(raw_filename: str) -> str:
    parts = [segment.strip() for segment in _PATH_SEPARATOR_RE.split(str(raw_filename or "").strip()) if segment.strip()]
    if not parts:
        return ""
    return parts[-1]


def normalize_filename_override(filename: str) -> str:
    candidate = _extract_cross_platform_basename(filename)
    if not candidate or candidate in {".", ".."}:
        raise ValueError("filename_override must contain a valid basename.")

    stem = Path(candidate).stem.strip()
    safe_stem = _SAFE_FILENAME_RE.sub("_", stem).strip()
    safe_stem = safe_stem.lstrip(".")
    if not safe_stem:
        raise ValueError("filename_override does not contain a safe basename after sanitization.")

    return f"{safe_stem}.csv"


def assert_within_root(target: Path, root: Path) -> Path:
    resolved_root = Path(root).expanduser().resolve()
    resolved_target = Path(target).expanduser().resolve()
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise ValueError(f"Resolved path escapes root {resolved_root}: {resolved_target}")
    return resolved_target
