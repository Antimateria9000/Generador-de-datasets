from __future__ import annotations

import hashlib
import re
from pathlib import Path

from dataset_core.contracts import DatasetRequest, TemporalRange
from dataset_core.path_safety import assert_within_root, normalize_filename_override
from dataset_core.settings import utc_now_token, utc_now_token_microseconds

_SAFE_WITH_DOT_RE = re.compile(r"[^A-Z0-9._-]+")
_SAFE_NO_DOT_RE = re.compile(r"[^A-Z0-9_-]+")
_SAFE_RUN_TOKEN_RE = re.compile(r"[^a-z0-9-]+")
_RUN_TICKER_MAX_LEN = 28


def sanitize_symbol_for_csv(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise ValueError("Symbol cannot be empty for filename generation.")
    return _SAFE_WITH_DOT_RE.sub("_", value)


def artifact_stem(symbol: str) -> str:
    return _SAFE_NO_DOT_RE.sub("_", sanitize_symbol_for_csv(symbol))


def build_csv_output_path(
    root: Path,
    symbol: str,
    request: DatasetRequest,
    *,
    force_qlib_contract: bool = False,
) -> Path:
    filename = build_csv_filename(symbol, request, force_qlib_contract=force_qlib_contract)
    target = Path(root).expanduser().resolve() / filename
    assert_within_root(target, root)
    return target


def build_range_tag(time_range: TemporalRange) -> str:
    if time_range.mode == "exact_dates" and time_range.start and time_range.end:
        start_tag = time_range.start.date().isoformat()
        end_tag = time_range.end.date().isoformat()
        return f"{start_tag}_{end_tag}_end_exclusive"
    if time_range.years is None:
        return "range"
    return f"{int(time_range.years)}y"


def _sanitize_run_token(value: str) -> str:
    normalized = _SAFE_RUN_TOKEN_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return normalized or "run"


def summarize_tickers_for_run_id(tickers: list[str] | tuple[str, ...], max_length: int = _RUN_TICKER_MAX_LEN) -> str:
    normalized = [_sanitize_run_token(artifact_stem(symbol).lower()) for symbol in tickers if str(symbol).strip()]
    if not normalized:
        return "no-ticker"

    unique_tokens = list(dict.fromkeys(normalized))
    joined = "-".join(unique_tokens)
    if len(joined) <= max_length:
        return joined

    first = unique_tokens[0][: min(10, max_length)]
    if len(unique_tokens) == 1:
        return first

    suffix = f"plus{len(unique_tokens) - 1}"
    candidate = f"{first}-{suffix}"
    if len(candidate) <= max_length:
        return candidate
    return candidate[:max_length].rstrip("-")


def build_run_id(request: DatasetRequest | None = None) -> str:
    timestamp = utc_now_token()
    unique_seed = utc_now_token_microseconds()
    mode = _sanitize_run_token("run" if request is None else request.mode)
    interval = _sanitize_run_token("na" if request is None else request.interval)
    tickers = [] if request is None else list(request.tickers)
    ticker_summary = summarize_tickers_for_run_id(tickers)
    digest_seed = "|".join([timestamp, unique_seed, mode, interval, ",".join(tickers)])
    short_hash = hashlib.sha1(digest_seed.encode("utf-8")).hexdigest()[:6]
    return f"{timestamp}_{mode}_{interval}_{ticker_summary}_{short_hash}"


def build_run_directory(request: DatasetRequest, run_id: str) -> Path:
    return request.output_dir / "runs" / run_id


def build_csv_filename(symbol: str, request: DatasetRequest, force_qlib_contract: bool = False) -> str:
    normalized = sanitize_symbol_for_csv(symbol)

    if request.mode == "qlib" or force_qlib_contract:
        return f"{normalized}.csv"

    if request.filename_override:
        return normalize_filename_override(request.filename_override)

    return f"{normalized}_{request.interval}_{build_range_tag(request.time_range)}.csv"
