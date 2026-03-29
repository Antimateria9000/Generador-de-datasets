from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dataset_core.settings import CORE_COLUMNS


class GeneralSanitizationError(ValueError):
    """Raised when the canonical dataset cannot be normalized safely."""


@dataclass(frozen=True)
class GeneralSanitizationResult:
    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    removed_rows: int = 0

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)


_RENAMES = {
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "stock splits": "stock_splits",
    "stocksplits": "stock_splits",
}

_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "factor",
)


class GeneralSanitizer:
    def sanitize(self, frame: pd.DataFrame, requested_extras: list[str] | tuple[str, ...]) -> GeneralSanitizationResult:
        if frame is None or frame.empty:
            raise GeneralSanitizationError("Provider returned an empty frame.")

        working = frame.copy()
        original_rows = len(working)
        warnings: list[str] = []

        if "date" not in working.columns:
            if working.index.name and str(working.index.name).lower() == "date":
                working = working.reset_index()
            elif isinstance(working.index, pd.DatetimeIndex):
                working = working.reset_index().rename(columns={working.index.name or "index": "date"})

        working.columns = [self._normalize_column_name(column) for column in working.columns]
        if "date" not in working.columns:
            raise GeneralSanitizationError("Provider frame does not contain the required 'date' column.")

        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        invalid_dates = int(working["date"].isna().sum())
        if invalid_dates:
            warnings.append(f"Removed {invalid_dates} rows with invalid dates.")
        working = working.dropna(subset=["date"]).sort_values("date")

        duplicate_count = int(working.duplicated(subset=["date"]).sum())
        if duplicate_count:
            warnings.append(f"Removed {duplicate_count} duplicate rows by date, keeping the last occurrence.")
            working = working.drop_duplicates(subset=["date"], keep="last")

        for column in _NUMERIC_COLUMNS:
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce")

        required_numeric = [column for column in CORE_COLUMNS if column != "date"]
        missing_required = [column for column in required_numeric if column not in working.columns]
        if missing_required:
            raise GeneralSanitizationError(
                f"Provider frame is missing required OHLCV columns: {missing_required}"
            )

        if "adj_close" not in working.columns and "adj_close" in requested_extras:
            working["adj_close"] = working["close"]
            warnings.append("adj_close was missing and was reconstructed from close.")

        for column, default_value in (("volume", 0.0), ("dividends", 0.0), ("stock_splits", 0.0)):
            if column in working.columns:
                working[column] = working[column].fillna(default_value)

        impossible_mask = self._build_impossible_mask(working)
        removed_impossible = int(impossible_mask.sum())
        if removed_impossible:
            warnings.append(f"Removed {removed_impossible} rows with impossible OHLCV geometry.")
            working = working.loc[~impossible_mask].copy()

        working = working.sort_values("date").reset_index(drop=True)
        if working.empty:
            raise GeneralSanitizationError("No valid rows remain after general sanitization.")

        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        removed_rows = original_rows - len(working)
        return GeneralSanitizationResult(frame=working, warnings=warnings, removed_rows=removed_rows)

    @staticmethod
    def _normalize_column_name(column: object) -> str:
        normalized = str(column).strip().lower().replace("-", "_")
        normalized = normalized.replace("/", "_").replace(" ", "_")
        return _RENAMES.get(normalized, normalized)

    @staticmethod
    def _build_impossible_mask(frame: pd.DataFrame) -> pd.Series:
        price_columns = ["open", "high", "low", "close"]
        numeric = frame[price_columns + ["volume"]].apply(pd.to_numeric, errors="coerce")

        high_too_low = numeric["high"] < numeric[price_columns].max(axis=1)
        low_too_high = numeric["low"] > numeric[price_columns].min(axis=1)
        negative_prices = (numeric[price_columns] < 0).any(axis=1)
        negative_volume = numeric["volume"] < 0
        missing_core = numeric[price_columns].isna().any(axis=1) | numeric["volume"].isna()

        return (high_too_low | low_too_high | negative_prices | negative_volume | missing_core).fillna(True)
