from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.settings import utc_now_token_microseconds

_SAFE_WITH_DOT_RE = re.compile(r"[^A-Z0-9._-]+")
_SAFE_NO_DOT_RE = re.compile(r"[^A-Z0-9_-]+")


def sanitize_symbol_for_csv(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise ValueError("Symbol cannot be empty for filename generation.")
    return _SAFE_WITH_DOT_RE.sub("_", value)


def artifact_stem(symbol: str) -> str:
    return _SAFE_NO_DOT_RE.sub("_", sanitize_symbol_for_csv(symbol))


def build_range_tag(time_range: TemporalRange) -> str:
    if time_range.mode == "exact_dates" and time_range.start and time_range.end:
        start_tag = time_range.start.date().isoformat()
        end_tag = time_range.end.date().isoformat()
        return f"{start_tag}_{end_tag}_end_exclusive"
    if time_range.years is None:
        return "range"
    return f"{int(time_range.years)}y"


def build_run_id() -> str:
    return f"{utc_now_token_microseconds()}_{uuid4().hex[:8]}"


def build_run_directory(request: DatasetRequest, run_id: str) -> Path:
    return request.output_dir / "runs" / run_id


def build_csv_filename(symbol: str, request: DatasetRequest, force_qlib_contract: bool = False) -> str:
    normalized = sanitize_symbol_for_csv(symbol)

    if request.mode == "qlib" or force_qlib_contract:
        return f"{normalized}.csv"

    if request.filename_override:
        filename = str(request.filename_override).strip()
        return filename if filename.lower().endswith(".csv") else f"{filename}.csv"

    return f"{normalized}_{request.interval}_{build_range_tag(request.time_range)}.csv"
