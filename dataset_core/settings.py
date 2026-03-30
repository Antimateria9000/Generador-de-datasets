from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

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


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_workspace_tree(base_root: Path | None = None) -> dict[str, Path]:
    workspace_root = Path(base_root or WORKSPACE_ROOT).expanduser().resolve()
    directories = {
        "workspace_root": ensure_directory(workspace_root),
        "runs": ensure_directory(workspace_root / "runs"),
        "exports": ensure_directory(workspace_root / "exports"),
        "manifests": ensure_directory(workspace_root / "manifests"),
        "reports": ensure_directory(workspace_root / "reports"),
        "references": ensure_directory(workspace_root / "references"),
        "cache": ensure_directory(workspace_root / "cache"),
        "temp": ensure_directory(workspace_root / "temp"),
        "logs": ensure_directory(workspace_root / "logs"),
        "audits": ensure_directory(workspace_root / "audits"),
    }
    return directories


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_token_microseconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
