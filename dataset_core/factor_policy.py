from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_SPLIT_REL_TOLERANCE = 0.20
_CONTINUITY_REL_TOLERANCE = 0.25


class FactorPolicyError(ValueError):
    """Raised when the Qlib factor policy cannot be applied safely."""


@dataclass(frozen=True)
class FactorApplicationResult:
    frame: pd.DataFrame
    factor_policy: str
    factor_source: str
    warnings: list[str] = field(default_factory=list)
    qlib_reasons: list[str] = field(default_factory=list)
    semantic_checks: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class _FactorCandidate:
    factor: pd.Series | None
    factor_policy: str
    factor_source: str
    warnings: list[str] = field(default_factory=list)
    qlib_reasons: list[str] = field(default_factory=list)
    semantic_checks: list[dict[str, object]] = field(default_factory=list)
    reference_adj_close: pd.Series | None = None

    @property
    def usable(self) -> bool:
        return self.factor is not None and not self.qlib_reasons


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
            "Factor requested: provider auto_adjust was forced to False to preserve provider-side semantics."
        )
    if requires_factor and not resolved_actions:
        resolved_actions = True
        warnings.append(
            "Factor requested: provider actions were forced to True to retrieve corporate-action metadata."
        )

    return resolved_auto_adjust, resolved_actions, warnings


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


def _sorted_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise FactorPolicyError("Factor policy cannot be applied to an empty frame.")

    working = frame.copy()
    working.attrs.update(getattr(frame, "attrs", {}))
    if "date" not in working.columns:
        raise FactorPolicyError("Factor policy requires a 'date' column.")

    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if working.empty:
        raise FactorPolicyError("Factor policy cannot be applied after removing invalid dates.")

    for column in ("open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    return working


def compute_split_factor(frame: pd.DataFrame, split_column: str = "stock_splits") -> pd.Series:
    ordered = _sorted_numeric_frame(frame)
    if split_column not in ordered.columns:
        raise FactorPolicyError(
            f"Factor requested but the provider frame does not include {split_column!r}."
        )

    splits = pd.to_numeric(ordered[split_column], errors="coerce").fillna(0.0)
    factors = np.ones(len(ordered), dtype=float)
    running_factor = 1.0

    for position in range(len(ordered) - 1, -1, -1):
        factors[position] = running_factor
        split_value = float(splits.iloc[position])
        if split_value > 0 and not np.isclose(split_value, 1.0):
            running_factor = running_factor / split_value

    return pd.Series(factors, index=ordered.index, name="factor")


def _iter_split_events(frame: pd.DataFrame) -> list[dict[str, object]]:
    if "stock_splits" not in frame.columns:
        return []

    splits = pd.to_numeric(frame["stock_splits"], errors="coerce").fillna(0.0)
    events: list[dict[str, object]] = []
    for index, split_value in enumerate(splits.tolist()):
        if split_value <= 0 or np.isclose(split_value, 1.0):
            continue
        events.append(
            {
                "index": index,
                "date": None if pd.isna(frame.loc[index, "date"]) else pd.Timestamp(frame.loc[index, "date"]).strftime("%Y-%m-%d"),
                "split": float(split_value),
            }
        )
    return events


def _classify_split_event(
    frame: pd.DataFrame,
    index: int,
    split_value: float,
    factor: pd.Series | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "index": index,
        "split": float(split_value),
        "date": pd.Timestamp(frame.loc[index, "date"]).strftime("%Y-%m-%d"),
        "close_state": "insufficient_context",
        "factor_state": "not_evaluated",
    }
    if index <= 0:
        return event

    previous_close = float(frame.loc[index - 1, "close"])
    current_close = float(frame.loc[index, "close"])
    if previous_close > 0 and current_close > 0:
        close_jump = previous_close / current_close
        event["close_jump"] = round(close_jump, 8)
        if _relative_match(close_jump, split_value, _SPLIT_REL_TOLERANCE):
            event["close_state"] = "raw_close_geometry"
        elif _relative_match(close_jump, 1.0, _CONTINUITY_REL_TOLERANCE):
            event["close_state"] = "already_split_adjusted"
        else:
            event["close_state"] = "ambiguous"

    if factor is None or index <= 0:
        return event

    previous_factor = float(factor.iloc[index - 1])
    current_factor = float(factor.iloc[index])
    if previous_factor > 0 and current_factor > 0:
        factor_jump = previous_factor / current_factor
        event["factor_jump"] = round(factor_jump, 8)
        if _relative_match(factor_jump, 1.0 / split_value, _SPLIT_REL_TOLERANCE):
            event["factor_state"] = "raw_close_geometry"
        elif _relative_match(factor_jump, 1.0, _CONTINUITY_REL_TOLERANCE):
            event["factor_state"] = "already_split_adjusted"
        else:
            event["factor_state"] = "ambiguous"

    return event


def _adj_close_candidate(working: pd.DataFrame) -> _FactorCandidate:
    warnings: list[str] = []
    reasons: list[str] = []
    checks: list[dict[str, object]] = []

    if "adj_close" not in working.columns:
        reason = "adj_close is missing, so the primary Yahoo/Qlib factor path is unavailable."
        _record_check(checks, "adj_close_available", False, "blocking", reason)
        return _FactorCandidate(
            factor=None,
            factor_policy="factor_unavailable",
            factor_source="adj_close_ratio",
            qlib_reasons=[reason],
            semantic_checks=checks,
        )

    close = pd.to_numeric(working["close"], errors="coerce")
    adj_close = pd.to_numeric(working["adj_close"], errors="coerce")
    usable_mask = adj_close.notna()

    if not usable_mask.any():
        reason = "adj_close is present but empty, so the primary Yahoo/Qlib factor path is unavailable."
        _record_check(checks, "adj_close_available", False, "blocking", reason)
        return _FactorCandidate(
            factor=None,
            factor_policy="factor_unavailable",
            factor_source="adj_close_ratio",
            qlib_reasons=[reason],
            semantic_checks=checks,
        )

    missing_adj_close = int(adj_close.isna().sum())
    missing_close = int((close.isna() | (close <= 0)).sum())
    if missing_adj_close:
        reasons.append("adj_close contains missing values and cannot safely drive factor derivation.")
    if missing_close:
        reasons.append("close contains missing or non-positive values, so adj_close/close cannot be computed safely.")

    factor = adj_close / close
    invalid_factor = int(((~np.isfinite(factor)) | (factor <= 0)).sum())
    if invalid_factor:
        reasons.append("adj_close/close produced non-finite or non-positive factor values.")

    _record_check(
        checks,
        "adj_close_ratio_integrity",
        passed=not reasons,
        severity="blocking" if reasons else "info",
        message="Primary factor path derived from adj_close/close.",
        missing_adj_close_rows=missing_adj_close,
        invalid_close_rows=missing_close,
        invalid_factor_rows=int(invalid_factor),
    )

    split_events = _iter_split_events(working)
    inconsistent_events: list[str] = []
    ambiguous_events: list[str] = []
    for event in split_events:
        classification = _classify_split_event(
            working,
            index=int(event["index"]),
            split_value=float(event["split"]),
            factor=factor,
        )
        close_state = str(classification["close_state"])
        factor_state = str(classification["factor_state"])
        if close_state in {"raw_close_geometry", "already_split_adjusted"} and factor_state in {
            "raw_close_geometry",
            "already_split_adjusted",
        } and close_state != factor_state:
            inconsistent_events.append(
                f"{classification['date']} split={classification['split']} close_state={close_state} factor_state={factor_state}"
            )
        elif "ambiguous" in {close_state, factor_state}:
            ambiguous_events.append(
                f"{classification['date']} split={classification['split']} close_state={close_state} factor_state={factor_state}"
            )

    if inconsistent_events:
        reasons.append(
            "Corporate-action semantics are inconsistent between close geometry and adj_close/close: "
            + "; ".join(inconsistent_events[:5])
        )
    if ambiguous_events:
        warnings.append(
            "Some split events could not be classified cleanly from adj_close/close semantics: "
            + "; ".join(ambiguous_events[:5])
        )

    _record_check(
        checks,
        "split_semantics_consistency",
        passed=not inconsistent_events,
        severity="blocking" if inconsistent_events else "warning" if ambiguous_events else "info",
        message="Compared split geometry against adj_close/close behaviour.",
        split_event_count=len(split_events),
        inconsistent_events=inconsistent_events[:5],
        ambiguous_events=ambiguous_events[:5],
    )

    return _FactorCandidate(
        factor=None if reasons else factor.astype(float),
        factor_policy="qlib_adjusted_from_adj_close_ratio",
        factor_source="adj_close_ratio",
        warnings=_unique(warnings),
        qlib_reasons=_unique(reasons),
        semantic_checks=checks,
        reference_adj_close=adj_close.astype(float),
    )


def _split_fallback_candidate(working: pd.DataFrame) -> _FactorCandidate:
    warnings: list[str] = []
    reasons: list[str] = []
    checks: list[dict[str, object]] = []

    if "stock_splits" not in working.columns:
        reason = "stock_splits is missing, so no controlled fallback factor source is available."
        _record_check(checks, "split_fallback_available", False, "blocking", reason)
        return _FactorCandidate(
            factor=None,
            factor_policy="factor_unavailable",
            factor_source="stock_splits_fallback",
            qlib_reasons=[reason],
            semantic_checks=checks,
        )

    close = pd.to_numeric(working["close"], errors="coerce")
    invalid_close = int((close.isna() | (close <= 0)).sum())
    if invalid_close:
        reasons.append("close contains missing or non-positive values, so split fallback is unsafe.")

    dividends = (
        pd.to_numeric(working["dividends"], errors="coerce").fillna(0.0)
        if "dividends" in working.columns
        else pd.Series(0.0, index=working.index)
    )
    dividend_rows = int((dividends.abs() > 1e-12).sum())
    if dividend_rows:
        reasons.append("Dividends are present but adj_close is unavailable, so split fallback cannot recover a safe Qlib factor.")

    split_events = _iter_split_events(working)
    already_adjusted_events: list[str] = []
    ambiguous_events: list[str] = []
    incomplete_events: list[str] = []
    for event in split_events:
        classification = _classify_split_event(
            working,
            index=int(event["index"]),
            split_value=float(event["split"]),
        )
        state = str(classification["close_state"])
        if state == "already_split_adjusted":
            already_adjusted_events.append(f"{classification['date']} split={classification['split']}")
        elif state == "ambiguous":
            ambiguous_events.append(
                f"{classification['date']} split={classification['split']} close_jump={classification.get('close_jump')}"
            )
        elif state == "insufficient_context":
            incomplete_events.append(f"{classification['date']} split={classification['split']}")

    if already_adjusted_events:
        reasons.append(
            "stock_splits fallback would double-adjust already adjusted prices around split events: "
            + "; ".join(already_adjusted_events[:5])
        )
    if ambiguous_events:
        reasons.append(
            "stock_splits fallback could not reconcile split geometry for all events: "
            + "; ".join(ambiguous_events[:5])
        )
    if incomplete_events:
        warnings.append(
            "Some split events had insufficient neighbouring rows for a full semantic check: "
            + "; ".join(incomplete_events[:5])
        )

    factor = None
    if not reasons:
        factor = compute_split_factor(working)
        if (factor <= 0).any():
            reasons.append("stock_splits fallback produced a non-positive factor.")
            factor = None

    _record_check(
        checks,
        "split_fallback_semantics",
        passed=not reasons,
        severity="blocking" if reasons else "warning" if warnings else "info",
        message="Controlled fallback based on stock_splits after semantic checks.",
        split_event_count=len(split_events),
        dividend_rows=dividend_rows,
        invalid_close_rows=invalid_close,
        already_adjusted_events=already_adjusted_events[:5],
        ambiguous_events=ambiguous_events[:5],
        incomplete_events=incomplete_events[:5],
    )

    if not split_events and not reasons:
        warnings.append("adj_close was unavailable; split fallback defaulted to factor=1 because no split events were present.")

    return _FactorCandidate(
        factor=None if reasons else factor,
        factor_policy="qlib_adjusted_from_split_fallback",
        factor_source="stock_splits_fallback",
        warnings=_unique(warnings),
        qlib_reasons=_unique(reasons),
        semantic_checks=checks,
    )


def _select_factor_candidate(working: pd.DataFrame) -> _FactorCandidate:
    primary = _adj_close_candidate(working)
    if primary.usable:
        return primary

    fallback = _split_fallback_candidate(working)
    if fallback.usable:
        primary_rejection = (
            []
            if not primary.qlib_reasons
            else [f"Primary factor path rejected: {' | '.join(primary.qlib_reasons)}"]
        )
        fallback_warnings = _unique(
            primary.warnings
            + primary_rejection
            + [
                "Primary factor path adj_close/close was unavailable or unsafe; stock_splits fallback was activated after semantic checks."
            ]
            + fallback.warnings
        )
        return _FactorCandidate(
            factor=fallback.factor,
            factor_policy=fallback.factor_policy,
            factor_source=fallback.factor_source,
            warnings=fallback_warnings,
            qlib_reasons=[],
            semantic_checks=primary.semantic_checks + fallback.semantic_checks,
            reference_adj_close=fallback.reference_adj_close,
        )

    reasons = _unique(primary.qlib_reasons + fallback.qlib_reasons)
    warnings = _unique(primary.warnings + fallback.warnings)
    raise FactorPolicyError(" | ".join(reasons + warnings))


def apply_factor_policy(frame: pd.DataFrame, adjust_ohlcv: bool) -> FactorApplicationResult:
    working = _sorted_numeric_frame(frame)
    qlib_reasons = [
        f"Missing required column for factor policy: {column}."
        for column in ("open", "high", "low", "close", "volume")
        if column not in working.columns
    ]
    if qlib_reasons:
        raise FactorPolicyError(" ".join(qlib_reasons))

    candidate = _select_factor_candidate(working)
    factor = pd.to_numeric(candidate.factor, errors="coerce")
    if factor.isna().any() or (factor <= 0).any():
        raise FactorPolicyError("Selected factor source produced invalid factor values.")

    working["factor"] = factor.astype(float)

    if adjust_ohlcv:
        for column in ("open", "high", "low", "close"):
            working[column] = pd.to_numeric(working[column], errors="coerce") * working["factor"]

        raw_volume = pd.to_numeric(working["volume"], errors="coerce")
        working["volume"] = np.where(
            working["factor"].to_numpy() != 0,
            raw_volume / working["factor"],
            np.nan,
        )
        if candidate.reference_adj_close is not None:
            working["close"] = candidate.reference_adj_close.to_numpy()
        policy_name = candidate.factor_policy
    else:
        policy_name = candidate.factor_policy.replace("qlib_adjusted", "factor_only")

    return FactorApplicationResult(
        frame=working,
        factor_policy=policy_name,
        factor_source=candidate.factor_source,
        warnings=_unique(candidate.warnings),
        qlib_reasons=list(qlib_reasons),
        semantic_checks=list(candidate.semantic_checks),
    )
