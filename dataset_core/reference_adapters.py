from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd


class ReferenceAdapter(Protocol):
    def name(self) -> str:
        """Human-readable adapter name."""

    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Load a reference frame for the requested symbol and range."""


class PriceReferenceAdapter(ReferenceAdapter, Protocol):
    def validation_scope(self) -> Literal["price"]:
        """Price/calendar reference adapter."""


class EventReferenceAdapter(Protocol):
    def name(self) -> str:
        """Human-readable adapter name."""

    def validation_scope(self) -> Literal["event"]:
        """Event-only reference adapter."""

    def fetch_events(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Load sparse manual events for the requested symbol and range."""


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
        "stock splits": "stock_splits",
    }
    working = working.rename(columns=rename_map)

    if "date" not in working.columns:
        first_column = str(working.columns[0])
        working = working.rename(columns={first_column: "date"})

    working["date"] = _normalize_reference_dates(working["date"])
    working = working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return working


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
    return working[columns].sort_values("date").reset_index(drop=True)


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
    return normalized.reset_index(drop=True)


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
    return normalized.reset_index(drop=True)


def adapter_validation_scope(adapter: object) -> str:
    scope = getattr(adapter, "validation_scope", None)
    if callable(scope):
        value = str(scope()).strip().lower()
        if value in {"price", "event"}:
            return value
    return "price"


class CSVReferenceAdapter:
    def __init__(self, reference_dir: Path) -> None:
        self.reference_dir = Path(reference_dir).expanduser().resolve()

    def name(self) -> str:
        return "csv_reference"

    def validation_scope(self) -> Literal["price"]:
        return "price"

    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        candidates = [
            self.reference_dir / f"{symbol}.csv",
            self.reference_dir / f"{symbol.upper()}.csv",
            self.reference_dir / f"{symbol.replace('.', '_')}.csv",
            self.reference_dir / f"{symbol.upper().replace('.', '_')}.csv",
        ]
        existing = next((path for path in candidates if path.exists()), None)
        if existing is None:
            raise FileNotFoundError(f"No CSV reference found for {symbol} in {self.reference_dir}")

        frame = pd.read_csv(existing)
        return filter_reference_frame(frame, start=start, end=end)


class ManualEventAdapter:
    def __init__(self, events_file: Path) -> None:
        self.events_file = Path(events_file).expanduser().resolve()

    def name(self) -> str:
        return "manual_events"

    def validation_scope(self) -> Literal["event"]:
        return "event"

    def fetch_events(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        if not self.events_file.exists():
            raise FileNotFoundError(f"Manual events file not found: {self.events_file}")

        if self.events_file.suffix.lower() == ".json":
            payload = json.loads(self.events_file.read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload)
        else:
            frame = pd.read_csv(self.events_file)

        normalized = normalize_event_frame(frame)
        if "symbol" in normalized.columns:
            normalized = normalized[normalized["symbol"] == symbol.upper()]

        return filter_event_frame(normalized, start=start, end=end)

    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        return self.fetch_events(symbol, start, end)
