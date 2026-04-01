from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd

from dataset_core.external_sources.base import attach_source_metadata, filter_event_frame, normalize_event_frame


class ManualEventReferenceSource:
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

        filtered = filter_event_frame(normalized, start=start, end=end)
        return attach_source_metadata(
            filtered,
            {
                "provider": "manual_events",
                "source": "manual_events",
                "scope": "event",
                "path": str(self.events_file.resolve()),
            },
        )

    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        return self.fetch_events(symbol, start, end)
