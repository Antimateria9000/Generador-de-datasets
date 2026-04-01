from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from dataset_core.external_sources.base import attach_source_metadata, filter_reference_frame


class CSVReferenceSource:
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
        filtered = filter_reference_frame(frame, start=start, end=end)
        return attach_source_metadata(
            filtered,
            {
                "provider": "csv",
                "source": "csv_reference",
                "scope": "price",
                "path": str(existing.resolve()),
            },
        )
