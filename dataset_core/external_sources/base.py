from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

ValidationScope = Literal["price", "event"]
SOURCE_METADATA_ATTR = "ab3_external_source_metadata"


class ExternalSourceError(Exception):
    """Base error for external validation source failures."""


class ExternalSourceNetworkError(ExternalSourceError):
    """Raised when the external source cannot be reached reliably."""


class ExternalSourceAuthError(ExternalSourceError):
    """Raised when the external source rejects credentials or authentication."""


class ExternalSourceRateLimitError(ExternalSourceError):
    """Raised when the external source rejects the request due to rate limiting."""


class ExternalSourceNotFoundError(ExternalSourceError):
    """Raised when a symbol cannot be resolved by the external source."""


class ExternalSourceCoverageError(ExternalSourceError):
    """Raised when a source lacks coverage for the requested symbol or range."""


class ExternalSourcePayloadError(ExternalSourceError):
    """Raised when the external source returns an invalid or unusable payload."""


@dataclass(frozen=True)
class ExternalSourceDescriptor:
    name: str
    scope: ValidationScope
    provider: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope,
            "provider": self.provider,
            "capabilities": list(self.capabilities),
            "details": dict(self.details),
        }


class ExternalReferenceSource(Protocol):
    def name(self) -> str:
        """Human-readable source name."""

    def validation_scope(self) -> ValidationScope:
        """Logical validation scope for the source."""


class ExternalPriceReferenceSource(ExternalReferenceSource, Protocol):
    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Load a normalized price reference frame for the requested symbol and range."""


class ExternalCorporateActionsReferenceSource(ExternalReferenceSource, Protocol):
    def fetch_events(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Load sparse corporate actions for the requested symbol and range."""


def attach_source_metadata(frame: pd.DataFrame, metadata: dict[str, object] | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    output = frame.copy()
    output.attrs[SOURCE_METADATA_ATTR] = dict(metadata or {})
    return output


def extract_source_metadata(frame: pd.DataFrame | None) -> dict[str, object]:
    if frame is None:
        return {}
    metadata = frame.attrs.get(SOURCE_METADATA_ATTR, {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _preserve_attrs(frame: pd.DataFrame, working: pd.DataFrame) -> pd.DataFrame:
    working.attrs = dict(getattr(frame, "attrs", {}))
    return working


def normalize_reference_timestamp(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).tz_convert("UTC").tz_localize(None)


def _normalize_reference_dates(values: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(values, utc=True, errors="coerce")
    return normalized.dt.tz_convert("UTC").dt.tz_localize(None)


def normalize_reference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    working.columns = [str(column).strip().lower() for column in working.columns]
    rename_map = {
        "adj close": "adj_close",
        "adjusted_close": "adj_close",
        "stock splits": "stock_splits",
    }
    working = working.rename(columns=rename_map)

    if "date" not in working.columns:
        first_column = str(working.columns[0])
        working = working.rename(columns={first_column: "date"})

    working["date"] = _normalize_reference_dates(working["date"])
    working = working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return _preserve_attrs(frame, working)


def normalize_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "dividends", "stock_splits"])

    working = normalize_reference_frame(frame)
    for column in ("dividends", "stock_splits"):
        if column not in working.columns:
            working[column] = 0.0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    if "symbol" in working.columns:
        working["symbol"] = working["symbol"].astype(str).str.upper()

    event_mask = (working["dividends"].abs() > 1e-12) | (working["stock_splits"].abs() > 1e-12)
    working = working[event_mask].copy()
    columns = ["date"]
    if "symbol" in working.columns:
        columns.append("symbol")
    columns.extend(["dividends", "stock_splits"])
    return _preserve_attrs(frame, working[columns].sort_values("date").reset_index(drop=True))


def filter_reference_frame(
    frame: pd.DataFrame,
    *,
    start: object | None,
    end: object | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    normalized = normalize_reference_frame(frame)
    if normalized.empty:
        return normalized

    start_ts = normalize_reference_timestamp(start) if start else None
    end_ts = normalize_reference_timestamp(end) if end else None
    if start_ts is not None:
        normalized = normalized[normalized["date"] >= start_ts]
    if end_ts is not None:
        normalized = normalized[normalized["date"] < end_ts]
    return _preserve_attrs(frame, normalized.reset_index(drop=True))


def filter_event_frame(
    frame: pd.DataFrame,
    *,
    start: object | None,
    end: object | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "dividends", "stock_splits"])

    normalized = normalize_event_frame(frame)
    if normalized.empty:
        return normalized

    start_ts = normalize_reference_timestamp(start) if start else None
    end_ts = normalize_reference_timestamp(end) if end else None
    if start_ts is not None:
        normalized = normalized[normalized["date"] >= start_ts]
    if end_ts is not None:
        normalized = normalized[normalized["date"] < end_ts]
    return _preserve_attrs(frame, normalized.reset_index(drop=True))


def adapter_validation_scope(adapter: object) -> ValidationScope:
    scope = getattr(adapter, "validation_scope", None)
    if callable(scope):
        value = str(scope()).strip().lower()
        if value in {"price", "event"}:
            return value  # type: ignore[return-value]
    return "price"
