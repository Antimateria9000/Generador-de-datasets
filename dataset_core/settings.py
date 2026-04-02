from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Final
from uuid import uuid4

APP_NAME: Final[str] = "Dataset Factory"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT: Final[Path] = PROJECT_ROOT / "workspace"
WORKSPACE_RUNS_DIR: Final[Path] = WORKSPACE_ROOT / "runs"
WORKSPACE_EXPORTS_DIR: Final[Path] = WORKSPACE_ROOT / "exports"
WORKSPACE_MANIFESTS_DIR: Final[Path] = WORKSPACE_ROOT / "manifests"
WORKSPACE_REPORTS_DIR: Final[Path] = WORKSPACE_ROOT / "reports"
WORKSPACE_REFERENCES_DIR: Final[Path] = WORKSPACE_ROOT / "references"
WORKSPACE_CACHE_DIR: Final[Path] = WORKSPACE_ROOT / "cache"
WORKSPACE_TEMP_DIR: Final[Path] = WORKSPACE_ROOT / "temp"
WORKSPACE_LOGS_DIR: Final[Path] = WORKSPACE_ROOT / "logs"
WORKSPACE_AUDITS_DIR: Final[Path] = WORKSPACE_ROOT / "audits"
DEFAULT_OUTPUT_ROOT: Final[Path] = WORKSPACE_ROOT
DEFAULT_METADATA_CANDIDATE_LIMIT: Final[int] = 4
DEFAULT_CONTEXT_CACHE_TTL_SECONDS: Final[int] = 24 * 60 * 60
DEFAULT_EODHD_BASE_URL: Final[str] = "https://eodhd.com"
DEFAULT_EODHD_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_EODHD_CACHE_TTL_SECONDS: Final[int] = 24 * 60 * 60
DEFAULT_EODHD_MAX_RETRIES: Final[int] = 2
DEFAULT_EODHD_BACKOFF_SECONDS: Final[float] = 0.5
DEFAULT_EODHD_PRICE_LOOKBACK_DAYS: Final[int] = 365
DEFAULT_YFINANCE_CACHE_MODE: Final[str] = "shared"

SUPPORTED_INTERVALS: Final[tuple[str, ...]] = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
)

LISTING_PREFERENCES: Final[tuple[str, ...]] = (
    "exact_symbol",
    "home_market",
    "prefer_europe",
    "prefer_usa",
)

DQ_MODES: Final[tuple[str, ...]] = ("off", "report", "strict")
PRESET_NAMES: Final[tuple[str, ...]] = ("base", "extended", "qlib")

CORE_COLUMNS: Final[tuple[str, ...]] = ("date", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("adj_close", "dividends", "stock_splits", "factor")
QLIB_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "factor",
)
QLIB_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ()

REFERENCE_RELATIVE_TOLERANCE: Final[float] = 5e-3
REFERENCE_SAMPLE_POINTS: Final[int] = 10
YFINANCE_CACHE_MODES: Final[tuple[str, ...]] = ("shared", "process", "run", "off")

WORKSPACE_DIRECTORIES: Final[tuple[Path, ...]] = (
    WORKSPACE_ROOT,
    WORKSPACE_RUNS_DIR,
    WORKSPACE_EXPORTS_DIR,
    WORKSPACE_MANIFESTS_DIR,
    WORKSPACE_REPORTS_DIR,
    WORKSPACE_REFERENCES_DIR,
    WORKSPACE_CACHE_DIR,
    WORKSPACE_TEMP_DIR,
    WORKSPACE_LOGS_DIR,
    WORKSPACE_AUDITS_DIR,
)
REDACTED_SECRET: Final[str] = "***redacted***"

_SECRET_QUERY_PARAM_RE = re.compile(
    r"(?i)([?&](?:api_token|api_key|apikey|access_token|token)=)[^&#\s]+"
)
_SECRET_ENV_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)\b\s*=\s*)([^\r\n]+)"
)
_REGISTERED_SECRETS: set[str] = set()
_REGISTERED_SECRETS_LOCK = Lock()
_CACHE_NAMESPACE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_workspace_tree(base_root: Path | None = None) -> dict[str, Path]:
    workspace_root = Path(base_root or WORKSPACE_ROOT).expanduser().resolve()
    return {
        "workspace_root": workspace_root,
        "runs": workspace_root / "runs",
        "exports": workspace_root / "exports",
        "manifests": workspace_root / "manifests",
        "reports": workspace_root / "reports",
        "references": workspace_root / "references",
        "cache": workspace_root / "cache",
        "temp": workspace_root / "temp",
        "logs": workspace_root / "logs",
        "audits": workspace_root / "audits",
    }


def resolve_effective_cache_paths(
    base_root: Path | None = None,
    provider_cache_dir: Path | None = None,
) -> dict[str, Path]:
    workspace = resolve_workspace_tree(base_root)
    default_cache_root = workspace["cache"]

    if provider_cache_dir is None:
        cache_root = default_cache_root
        yfinance_cache = cache_root / "yfinance"
    else:
        yfinance_cache = Path(provider_cache_dir).expanduser().resolve()
        cache_root = yfinance_cache.parent if yfinance_cache.name.lower() == "yfinance" else yfinance_cache

    return {
        "cache_root": cache_root,
        "yfinance": yfinance_cache,
        "market_context": cache_root / "market_context",
    }


def normalize_yfinance_cache_mode(mode: str | None) -> str:
    normalized = str(mode or DEFAULT_YFINANCE_CACHE_MODE).strip().lower()
    if normalized not in YFINANCE_CACHE_MODES:
        allowed = ", ".join(YFINANCE_CACHE_MODES)
        raise ValueError(f"Unsupported yfinance cache mode: {mode!r}. Allowed values: {allowed}.")
    return normalized


def sanitize_cache_namespace(namespace: str | None, *, fallback_prefix: str = "session") -> str:
    normalized = _CACHE_NAMESPACE_RE.sub("-", str(namespace or "").strip()).strip("-.")
    if normalized:
        return normalized
    return f"{fallback_prefix}-{uuid4().hex[:12]}"


def resolve_yfinance_cache_dir(
    base_root: Path | None = None,
    provider_cache_dir: Path | None = None,
    *,
    cache_mode: str | None = None,
    cache_namespace: str | None = None,
) -> Path | None:
    normalized_mode = normalize_yfinance_cache_mode(cache_mode)
    if normalized_mode == "off":
        return None

    base_cache_dir = resolve_effective_cache_paths(base_root, provider_cache_dir)["yfinance"]
    if normalized_mode == "shared":
        return base_cache_dir
    if normalized_mode == "process":
        return base_cache_dir / f"process-{os.getpid()}"

    namespace = sanitize_cache_namespace(cache_namespace, fallback_prefix="run")
    return base_cache_dir / namespace


def ensure_workspace_tree(base_root: Path | None = None) -> dict[str, Path]:
    directories = resolve_workspace_tree(base_root)
    for path in directories.values():
        ensure_directory(path)
    return directories


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_token_microseconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def normalize_secret(value: str | None) -> str | None:
    normalized = None if value is None else str(value).strip()
    return normalized or None


def register_secret(secret: str | None) -> None:
    normalized = normalize_secret(secret)
    if normalized is None:
        return
    with _REGISTERED_SECRETS_LOCK:
        _REGISTERED_SECRETS.add(normalized)


def mask_secret(secret: str | None) -> str | None:
    normalized = normalize_secret(secret)
    if normalized is None:
        return None
    if len(normalized) <= 8:
        return REDACTED_SECRET
    return f"{normalized[:4]}...{normalized[-4:]}"


def sanitize_secret_text(value: str | None) -> str | None:
    if value is None:
        return None
    sanitized = str(value)
    if not sanitized:
        return sanitized

    sanitized = _SECRET_QUERY_PARAM_RE.sub(rf"\1{REDACTED_SECRET}", sanitized)
    sanitized = _SECRET_ENV_ASSIGNMENT_RE.sub(rf"\1{REDACTED_SECRET}", sanitized)

    with _REGISTERED_SECRETS_LOCK:
        registered = sorted(_REGISTERED_SECRETS, key=len, reverse=True)
    for secret in registered:
        sanitized = sanitized.replace(secret, REDACTED_SECRET)
    return sanitized


def _parse_env_value(raw_value: str) -> str:
    value = str(raw_value).strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    comment_index = value.find(" #")
    if comment_index >= 0:
        return value[:comment_index].rstrip()
    return value


def _load_env_with_python_dotenv(env_path: Path) -> bool:
    try:
        from dotenv import load_dotenv
    except Exception:
        return False

    load_dotenv(dotenv_path=env_path, override=False, encoding="utf-8")
    return True


def _load_env_without_python_dotenv(env_path: Path) -> None:
    content = env_path.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        env_name = str(name).strip()
        if not env_name:
            continue
        os.environ.setdefault(env_name, _parse_env_value(raw_value))


@lru_cache(maxsize=None)
def _load_local_env_cached(project_root: str) -> str | None:
    env_path = Path(project_root).expanduser().resolve() / ".env"
    if not env_path.is_file():
        return None
    if not _load_env_with_python_dotenv(env_path):
        _load_env_without_python_dotenv(env_path)
    return str(env_path)


def load_local_env(project_root: Path | None = None) -> Path | None:
    root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    loaded_path = _load_local_env_cached(str(root))
    return None if loaded_path is None else Path(loaded_path)


def reset_local_env_cache() -> None:
    _load_local_env_cached.cache_clear()


def get_env_secret(name: str, *, project_root: Path | None = None) -> str | None:
    load_local_env(project_root=project_root)
    secret = normalize_secret(os.getenv(name))
    register_secret(secret)
    return secret


def resolve_env_secret(
    name: str,
    manual_value: str | None = None,
    *,
    allow_env_fallback: bool = True,
    project_root: Path | None = None,
) -> str | None:
    manual_secret = normalize_secret(manual_value)
    if manual_secret is not None:
        register_secret(manual_secret)
        return manual_secret
    if not allow_env_fallback:
        return None
    return get_env_secret(name, project_root=project_root)


def get_default_eodhd_api_key(*, project_root: Path | None = None) -> str | None:
    return get_env_secret("EODHD_API_KEY", project_root=project_root)


def resolve_eodhd_api_key(
    manual_value: str | None = None,
    *,
    allow_env_fallback: bool = True,
    project_root: Path | None = None,
) -> str | None:
    return resolve_env_secret(
        "EODHD_API_KEY",
        manual_value,
        allow_env_fallback=allow_env_fallback,
        project_root=project_root,
    )
