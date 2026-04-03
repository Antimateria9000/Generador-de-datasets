from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dataset_core.settings import CORE_COLUMNS

_GENERAL_STRUCTURED_WARNINGS_ATTR = "ab3_general_structured_warnings"
_GENERAL_COLUMN_PROVENANCE_ATTR = "ab3_general_column_provenance"


class GeneralSanitizationError(ValueError):
    """Raised when the canonical dataset cannot be normalized safely."""


@dataclass(frozen=True)
class GeneralSanitizationResult:
    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    removed_rows: int = 0
    structured_warnings: list[dict[str, object]] = field(default_factory=list)
    column_provenance: dict[str, dict[str, object]] = field(default_factory=dict)

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
        structured_warnings: list[dict[str, object]] = []
        requested_extra_set = {str(item).strip().lower() for item in requested_extras if str(item).strip()}
        materialized_adj_close = False

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

        # Policy choice: preserve requested adj_close compatibility with an explicit
        # empty column, never by fabricating provider-equivalent prices from close.
        if "adj_close" not in working.columns and "adj_close" in requested_extra_set:
            working["adj_close"] = pd.Series([pd.NA] * len(working), index=working.index, dtype="Float64")
            materialized_adj_close = True

        adj_close_provenance: dict[str, object] | None = None
        if "adj_close" in working.columns:
            adj_close = pd.to_numeric(working["adj_close"], errors="coerce")
            adj_close_has_real_values = bool(adj_close.notna().any())
            if adj_close_has_real_values:
                adj_close_provenance = {
                    "state": "provider",
                    "synthetic": False,
                    "materialized_empty_column": False,
                    "requested": "adj_close" in requested_extra_set,
                }
            else:
                adj_close_provenance = {
                    "state": "provider_missing",
                    "synthetic": False,
                    "materialized_empty_column": materialized_adj_close,
                    "requested": "adj_close" in requested_extra_set,
                }
                if "adj_close" in requested_extra_set:
                    message = (
                        "adj_close was requested but the provider did not supply usable adjusted-close values; "
                        "the column remains empty and is never treated as provider evidence."
                    )
                    warnings.append(message)
                    structured_warnings.append(
                        {
                            "code": "adj_close_unavailable",
                            "severity": "warning",
                            "column": "adj_close",
                            "state": "provider_missing",
                            "materialized_empty_column": materialized_adj_close,
                            "message": message,
                        }
                    )

        for column, default_value in (("volume", 0.0), ("dividends", 0.0), ("stock_splits", 0.0)):
            if column in working.columns:
                working[column] = working[column].fillna(default_value)

        rejection_reasons = self._build_row_rejection_reasons(working)
        rejected_mask = rejection_reasons != ""
        rejected_count = int(rejected_mask.sum())
        if rejected_count:
            reason_counts = rejection_reasons[rejected_mask].value_counts()
            if int(reason_counts.get("non_positive_ohlc", 0)):
                warnings.append(
                    f"Removed {int(reason_counts['non_positive_ohlc'])} rows with non-positive OHLC prices."
                )
            if int(reason_counts.get("impossible_ohlc_geometry", 0)):
                warnings.append(
                    f"Removed {int(reason_counts['impossible_ohlc_geometry'])} rows with impossible OHLC geometry."
                )
            if int(reason_counts.get("negative_volume", 0)):
                warnings.append(
                    f"Removed {int(reason_counts['negative_volume'])} rows with negative volume."
                )
            if int(reason_counts.get("missing_core_ohlcv", 0)):
                warnings.append(
                    f"Removed {int(reason_counts['missing_core_ohlcv'])} rows with missing core OHLCV values."
                )
            working = working.loc[~rejected_mask].copy()

        working = working.sort_values("date").reset_index(drop=True)
        if working.empty:
            raise GeneralSanitizationError("No valid rows remain after general sanitization.")

        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        removed_rows = original_rows - len(working)
        column_provenance = {}
        if adj_close_provenance is not None:
            column_provenance["adj_close"] = dict(adj_close_provenance)
        working.attrs[_GENERAL_STRUCTURED_WARNINGS_ATTR] = [dict(item) for item in structured_warnings]
        working.attrs[_GENERAL_COLUMN_PROVENANCE_ATTR] = dict(column_provenance)
        return GeneralSanitizationResult(
            frame=working,
            warnings=warnings,
            removed_rows=removed_rows,
            structured_warnings=structured_warnings,
            column_provenance=column_provenance,
        )

    @staticmethod
    def _normalize_column_name(column: object) -> str:
        normalized = str(column).strip().lower().replace("-", "_")
        normalized = normalized.replace("/", "_").replace(" ", "_")
        return _RENAMES.get(normalized, normalized)

    @staticmethod
    def _build_row_rejection_reasons(frame: pd.DataFrame) -> pd.Series:
        price_columns = ["open", "high", "low", "close"]
        numeric = frame[price_columns + ["volume"]].apply(pd.to_numeric, errors="coerce")

        missing_core = numeric[price_columns].isna().any(axis=1) | numeric["volume"].isna()
        non_positive_prices = (numeric[price_columns] <= 0).any(axis=1)
        negative_volume = numeric["volume"] < 0
        high_too_low = numeric["high"] < numeric[price_columns].max(axis=1)
        low_too_high = numeric["low"] > numeric[price_columns].min(axis=1)
        impossible_geometry = high_too_low | low_too_high

        reasons = pd.Series("", index=frame.index, dtype="object")
        reasons = reasons.mask(missing_core.fillna(True), "missing_core_ohlcv")
        reasons = reasons.mask(reasons.eq("") & non_positive_prices.fillna(False), "non_positive_ohlc")
        reasons = reasons.mask(reasons.eq("") & negative_volume.fillna(False), "negative_volume")
        reasons = reasons.mask(reasons.eq("") & impossible_geometry.fillna(False), "impossible_ohlc_geometry")
        return reasons
