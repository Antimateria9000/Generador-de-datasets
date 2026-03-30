from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dataset_core.settings import QLIB_OPTIONAL_COLUMNS, QLIB_REQUIRED_COLUMNS

_PRICE_JUMP_WARNING_THRESHOLD = 0.35
_PRICE_JUMP_ERROR_THRESHOLD = 0.65
_FACTOR_STEP_THRESHOLD = 1.5
_CONTINUITY_REL_TOLERANCE = 0.25
_DOUBLE_ADJUST_REL_TOLERANCE = 0.20
_ADJ_CLOSE_WARNING_TOLERANCE = 5e-3
_ADJ_CLOSE_ERROR_TOLERANCE = 2e-2


@dataclass(frozen=True)
class QlibContractResult:
    compatible: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)


def _record_check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    severity: str,
    message: str,
    **details: object,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "message": message,
            "details": details,
        }
    )


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _relative_match(observed: float, expected: float, tolerance: float) -> bool:
    if not np.isfinite(observed) or not np.isfinite(expected) or expected <= 0:
        return False
    return abs(observed - expected) / expected <= tolerance


def validate_qlib_frame(
    frame: pd.DataFrame,
    reference_adj_close: pd.Series | None = None,
) -> QlibContractResult:
    reasons: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, object]] = []
    metrics: dict[str, object] = {}

    if frame is None or frame.empty:
        reasons.append("Qlib dataset is empty.")
        _record_check(checks, "non_empty_frame", False, "blocking", "Qlib dataset is empty.")
        return QlibContractResult(
            compatible=False,
            reasons=reasons,
            warnings=warnings,
            checks=checks,
            metrics=metrics,
        )

    columns = list(frame.columns)
    required = list(QLIB_REQUIRED_COLUMNS)
    missing = [column for column in required if column not in columns]
    if missing:
        reasons.append(f"Missing required Qlib columns: {missing}.")
    _record_check(
        checks,
        "required_columns",
        passed=not missing,
        severity="blocking" if missing else "info",
        message="Validated required Qlib columns.",
        missing_columns=missing,
    )

    canonical_order = columns[: len(required)] == required
    if not canonical_order:
        reasons.append("Qlib columns are not in canonical order.")
    _record_check(
        checks,
        "column_order",
        passed=canonical_order,
        severity="blocking" if not canonical_order else "info",
        message="Validated canonical Qlib column order.",
        observed_prefix=columns[: len(required)],
    )

    working = frame.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")

    invalid_dates = int(working["date"].isna().sum())
    duplicate_dates = int(working["date"].duplicated().sum())
    sorted_dates = bool(working["date"].is_monotonic_increasing)
    if invalid_dates:
        reasons.append("date column contains invalid values.")
    if not sorted_dates:
        reasons.append("date column is not sorted ascending.")
    if duplicate_dates:
        reasons.append("date column contains duplicates.")
    _record_check(
        checks,
        "date_integrity",
        passed=invalid_dates == 0 and sorted_dates and duplicate_dates == 0,
        severity="blocking" if invalid_dates or not sorted_dates or duplicate_dates else "info",
        message="Validated Qlib date ordering and uniqueness.",
        invalid_dates=invalid_dates,
        duplicate_dates=duplicate_dates,
        sorted=sorted_dates,
    )

    numeric_columns = ("open", "high", "low", "close", "volume", "factor")
    numeric_data: dict[str, pd.Series] = {}
    for column in numeric_columns:
        if column not in working.columns:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        numeric_data[column] = numeric
        invalid_numeric = int(numeric.isna().sum())
        if invalid_numeric:
            reasons.append(f"{column} contains non-numeric or missing values.")
        if column == "factor" and (numeric <= 0).any():
            reasons.append("factor must be strictly positive.")
        if column == "volume" and (numeric < 0).any():
            reasons.append("volume cannot be negative.")
        if column in {"open", "high", "low", "close"} and (numeric <= 0).any():
            reasons.append(f"{column} must be strictly positive for Qlib compatibility.")
        _record_check(
            checks,
            f"{column}_numeric",
            passed=invalid_numeric == 0
            and not (column == "factor" and (numeric <= 0).any())
            and not (column == "volume" and (numeric < 0).any())
            and not (column in {"open", "high", "low", "close"} and (numeric <= 0).any()),
            severity="blocking"
            if invalid_numeric
            or (column == "factor" and (numeric <= 0).any())
            or (column == "volume" and (numeric < 0).any())
            or (column in {"open", "high", "low", "close"} and (numeric <= 0).any())
            else "info",
            message=f"Validated numeric integrity for {column}.",
            invalid_rows=invalid_numeric,
        )

    if {"open", "high", "low", "close"} <= numeric_data.keys():
        open_values = numeric_data["open"]
        high_values = numeric_data["high"]
        low_values = numeric_data["low"]
        close_values = numeric_data["close"]
        bad_geometry = (high_values < pd.concat([open_values, close_values, low_values], axis=1).max(axis=1)) | (
            low_values > pd.concat([open_values, close_values, high_values], axis=1).min(axis=1)
        )
        bad_geometry_count = int(bad_geometry.fillna(False).sum())
        if bad_geometry_count:
            reasons.append("OHLC rows contain impossible high/low geometry.")
        _record_check(
            checks,
            "ohlc_geometry",
            passed=bad_geometry_count == 0,
            severity="blocking" if bad_geometry_count else "info",
            message="Validated intrarow OHLC geometry.",
            invalid_rows=bad_geometry_count,
        )

    if {"close", "factor"} <= numeric_data.keys() and not reasons:
        close_values = numeric_data["close"].astype(float)
        factor_values = numeric_data["factor"].astype(float)
        raw_close = close_values / factor_values
        invalid_raw_close = int((~np.isfinite(raw_close) | (raw_close <= 0)).sum())
        if invalid_raw_close:
            reasons.append("close/factor produced invalid reconstructed raw prices.")
        _record_check(
            checks,
            "close_factor_coherence",
            passed=invalid_raw_close == 0,
            severity="blocking" if invalid_raw_close else "info",
            message="Validated reconstructed raw close from close/factor.",
            invalid_rows=invalid_raw_close,
        )

        close_ratio = (close_values.shift(1) / close_values).replace([np.inf, -np.inf], np.nan)
        factor_ratio = (factor_values.shift(1) / factor_values).replace([np.inf, -np.inf], np.nan)
        raw_ratio = (raw_close.shift(1) / raw_close).replace([np.inf, -np.inf], np.nan)

        close_abs_return = close_values.pct_change().abs()
        factor_step_mask = (factor_ratio > _FACTOR_STEP_THRESHOLD) | (factor_ratio < (1.0 / _FACTOR_STEP_THRESHOLD))
        moderate_price_jumps = close_abs_return > _PRICE_JUMP_WARNING_THRESHOLD
        extreme_price_jumps = close_abs_return > _PRICE_JUMP_ERROR_THRESHOLD

        unexpected_extreme_jumps = extreme_price_jumps & ~factor_step_mask.fillna(False)
        if int(unexpected_extreme_jumps.fillna(False).sum()):
            reasons.append("Adjusted close contains extreme discontinuities without a matching factor step.")
        elif int(moderate_price_jumps.fillna(False).sum()):
            warnings.append("Adjusted close contains unusually large moves; review the Qlib report for diagnostics.")

        max_close_abs_return = close_abs_return.max(skipna=True)
        metrics["max_close_abs_return"] = round(
            0.0 if pd.isna(max_close_abs_return) else float(max_close_abs_return),
            6,
        )
        metrics["factor_step_events"] = int(factor_step_mask.fillna(False).sum())
        metrics["unexpected_extreme_price_jumps"] = int(unexpected_extreme_jumps.fillna(False).sum())

        double_adjust_events: list[dict[str, object]] = []
        ambiguous_factor_events: list[dict[str, object]] = []
        for index in range(1, len(working)):
            observed_factor_ratio = float(factor_ratio.iloc[index]) if pd.notna(factor_ratio.iloc[index]) else np.nan
            if not np.isfinite(observed_factor_ratio):
                continue
            if not ((observed_factor_ratio > _FACTOR_STEP_THRESHOLD) or (observed_factor_ratio < (1.0 / _FACTOR_STEP_THRESHOLD))):
                continue

            observed_close_ratio = float(close_ratio.iloc[index]) if pd.notna(close_ratio.iloc[index]) else np.nan
            observed_raw_ratio = float(raw_ratio.iloc[index]) if pd.notna(raw_ratio.iloc[index]) else np.nan
            event_payload = {
                "date": working.loc[index, "date"].strftime("%Y-%m-%d"),
                "factor_ratio": round(observed_factor_ratio, 8),
                "close_ratio": None if not np.isfinite(observed_close_ratio) else round(observed_close_ratio, 8),
                "raw_ratio": None if not np.isfinite(observed_raw_ratio) else round(observed_raw_ratio, 8),
            }
            if (
                np.isfinite(observed_close_ratio)
                and np.isfinite(observed_raw_ratio)
                and _relative_match(observed_close_ratio, observed_factor_ratio, _DOUBLE_ADJUST_REL_TOLERANCE)
                and _relative_match(observed_raw_ratio, 1.0, _DOUBLE_ADJUST_REL_TOLERANCE)
            ):
                double_adjust_events.append(event_payload)
                continue

            expected_raw_ratio = 1.0 / observed_factor_ratio if observed_factor_ratio != 0 else np.nan
            adjusted_continuous = np.isfinite(observed_close_ratio) and _relative_match(
                observed_close_ratio,
                1.0,
                _CONTINUITY_REL_TOLERANCE,
            )
            raw_consistent = np.isfinite(observed_raw_ratio) and _relative_match(
                observed_raw_ratio,
                expected_raw_ratio,
                _CONTINUITY_REL_TOLERANCE,
            )
            if not adjusted_continuous and not raw_consistent:
                ambiguous_factor_events.append(event_payload)

        if double_adjust_events:
            reasons.append(
                "Detected patterns compatible with double adjustment around factor step changes."
            )
        if ambiguous_factor_events:
            warnings.append(
                "Some factor step changes are semantically ambiguous; inspect the event diagnostics in qlib_report."
            )

        _record_check(
            checks,
            "factor_step_semantics",
            passed=not double_adjust_events,
            severity="blocking" if double_adjust_events else "warning" if ambiguous_factor_events else "info",
            message="Validated close/factor behaviour around step changes.",
            double_adjust_events=double_adjust_events[:5],
            ambiguous_events=ambiguous_factor_events[:5],
        )

    if reference_adj_close is not None and "close" in numeric_data:
        aligned_adj = pd.to_numeric(pd.Series(reference_adj_close), errors="coerce").reset_index(drop=True)
        aligned_close = pd.to_numeric(numeric_data["close"], errors="coerce").reset_index(drop=True)
        if len(aligned_adj) != len(aligned_close):
            warnings.append("adj_close reference was available but could not be aligned by row count.")
            _record_check(
                checks,
                "adj_close_reference_alignment",
                False,
                "warning",
                "adj_close reference length does not match the Qlib frame.",
                reference_rows=len(aligned_adj),
                frame_rows=len(aligned_close),
            )
        else:
            base = aligned_adj.abs().replace(0, pd.NA)
            rel_diff = ((aligned_close - aligned_adj).abs() / base).fillna(0.0)
            max_rel_diff = float(rel_diff.max()) if not rel_diff.empty else 0.0
            metrics["adj_close_max_relative_diff"] = round(max_rel_diff, 8)
            if max_rel_diff > _ADJ_CLOSE_ERROR_TOLERANCE:
                reasons.append("Adjusted close diverges materially from the provided adj_close reference.")
            elif max_rel_diff > _ADJ_CLOSE_WARNING_TOLERANCE:
                warnings.append("Adjusted close differs slightly from the provided adj_close reference.")
            _record_check(
                checks,
                "adj_close_reference_consistency",
                passed=max_rel_diff <= _ADJ_CLOSE_ERROR_TOLERANCE,
                severity="blocking" if max_rel_diff > _ADJ_CLOSE_ERROR_TOLERANCE else "warning" if max_rel_diff > _ADJ_CLOSE_WARNING_TOLERANCE else "info",
                message="Compared adjusted close against the source adj_close reference.",
                max_relative_diff=round(max_rel_diff, 8),
            )

    allowed_optional = set(QLIB_OPTIONAL_COLUMNS)
    unexpected_extra_columns = [column for column in columns if column not in required and column not in allowed_optional]
    if unexpected_extra_columns:
        reasons.append(
            "Qlib export contains forbidden columns outside the closed contract: "
            + ", ".join(unexpected_extra_columns)
            + "."
        )
    _record_check(
        checks,
        "closed_contract_columns",
        passed=not unexpected_extra_columns,
        severity="blocking" if unexpected_extra_columns else "info",
        message="Validated that the closed Qlib contract does not emit forbidden columns.",
        unexpected_columns=unexpected_extra_columns,
    )

    return QlibContractResult(
        compatible=not reasons,
        reasons=_unique(reasons),
        warnings=_unique(warnings),
        checks=checks,
        metrics=metrics,
    )
