from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dataset_core.settings import QLIB_REQUIRED_COLUMNS


@dataclass(frozen=True)
class QlibContractResult:
    compatible: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_qlib_frame(frame: pd.DataFrame) -> QlibContractResult:
    reasons: list[str] = []
    warnings: list[str] = []

    if frame is None or frame.empty:
        reasons.append("Qlib dataset is empty.")
        return QlibContractResult(compatible=False, reasons=reasons, warnings=warnings)

    columns = list(frame.columns)
    required = list(QLIB_REQUIRED_COLUMNS)
    missing = [column for column in required if column not in columns]
    if missing:
        reasons.append(f"Missing required Qlib columns: {missing}.")

    if columns[: len(required)] != required:
        reasons.append("Qlib columns are not in canonical order.")

    working = frame.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
    if working["date"].isna().any():
        reasons.append("date column contains invalid values.")
    if not working["date"].is_monotonic_increasing:
        reasons.append("date column is not sorted ascending.")
    if working["date"].duplicated().any():
        reasons.append("date column contains duplicates.")

    for column in ("open", "high", "low", "close", "volume", "factor"):
        if column not in working.columns:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        if numeric.isna().any():
            reasons.append(f"{column} contains non-numeric or missing values.")
            continue
        if column == "factor" and (numeric <= 0).any():
            reasons.append("factor must be strictly positive.")
        if column == "volume" and (numeric < 0).any():
            reasons.append("volume cannot be negative.")

    if not reasons and any(column not in required for column in columns):
        warnings.append("Qlib export contains additional diagnostic columns beyond the minimum contract.")

    return QlibContractResult(compatible=not reasons, reasons=reasons, warnings=warnings)
