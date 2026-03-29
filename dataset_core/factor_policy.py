from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class FactorPolicyError(ValueError):
    """Raised when the Qlib factor policy cannot be applied safely."""


@dataclass(frozen=True)
class FactorApplicationResult:
    frame: pd.DataFrame
    factor_policy: str
    warnings: list[str] = field(default_factory=list)
    qlib_reasons: list[str] = field(default_factory=list)


def resolve_provider_flags(
    auto_adjust: bool,
    actions: bool,
    requires_factor: bool,
) -> tuple[bool, bool, list[str]]:
    resolved_auto_adjust = bool(auto_adjust)
    resolved_actions = bool(actions)
    warnings: list[str] = []

    if requires_factor and resolved_auto_adjust:
        resolved_auto_adjust = False
        warnings.append(
            "Factor requested: provider auto_adjust was forced to False to preserve raw OHLC."
        )
    if requires_factor and not resolved_actions:
        resolved_actions = True
        warnings.append(
            "Factor requested: provider actions were forced to True to retrieve split events."
        )

    return resolved_auto_adjust, resolved_actions, warnings


def compute_split_factor(frame: pd.DataFrame, split_column: str = "stock_splits") -> pd.Series:
    if split_column not in frame.columns:
        raise FactorPolicyError(
            f"Factor requested but the provider frame does not include {split_column!r}."
        )

    ordered = frame.copy()
    if "date" not in ordered.columns:
        raise FactorPolicyError("Factor policy requires a 'date' column.")

    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered = ordered.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if ordered.empty:
        raise FactorPolicyError("Factor policy cannot be applied to an empty frame.")

    splits = pd.to_numeric(ordered[split_column], errors="coerce").fillna(0.0)
    factors = np.ones(len(ordered), dtype=float)
    running_factor = 1.0

    for position in range(len(ordered) - 1, -1, -1):
        factors[position] = running_factor
        split_value = float(splits.iloc[position])
        if split_value > 0 and not np.isclose(split_value, 1.0):
            running_factor = running_factor / split_value

    return pd.Series(factors, index=ordered.index, name="factor")


def apply_factor_policy(frame: pd.DataFrame, adjust_ohlcv: bool) -> FactorApplicationResult:
    working = frame.copy()
    if "date" not in working.columns:
        raise FactorPolicyError("Schema builder requires a 'date' column.")

    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    factor = compute_split_factor(working)
    working["factor"] = factor

    qlib_reasons: list[str] = []
    for column in ("open", "high", "low", "close", "volume"):
        if column not in working.columns:
            qlib_reasons.append(f"Missing required column for factor policy: {column}.")

    if qlib_reasons:
        raise FactorPolicyError(" ".join(qlib_reasons))

    if adjust_ohlcv:
        for column in ("open", "high", "low", "close"):
            working[column] = pd.to_numeric(working[column], errors="coerce") * working["factor"]

        raw_volume = pd.to_numeric(working["volume"], errors="coerce")
        working["volume"] = np.where(
            working["factor"].to_numpy() != 0,
            raw_volume / working["factor"],
            np.nan,
        )
        policy_name = "qlib_split_adjusted_from_raw_ohlcv"
    else:
        policy_name = "split_factor_only_from_raw_ohlcv"

    return FactorApplicationResult(frame=working, factor_policy=policy_name, qlib_reasons=qlib_reasons)
