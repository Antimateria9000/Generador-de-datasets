from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dataset_core.factor_policy import FactorPolicyError, apply_factor_policy
from dataset_core.qlib_contract import QlibContractResult, validate_qlib_frame


class QlibSanitizationError(ValueError):
    """Raised when the dataset cannot be converted into a Qlib-ready artifact."""


@dataclass(frozen=True)
class QlibSanitizationResult:
    frame: pd.DataFrame
    factor_policy: str
    factor_source: str
    technical_report: dict[str, object]
    warnings: list[str] = field(default_factory=list)
    contract: QlibContractResult = field(default_factory=lambda: QlibContractResult(False, ["not_validated"]))


class QlibSanitizer:
    def sanitize(self, frame: pd.DataFrame, include_adj_close: bool = False) -> QlibSanitizationResult:
        if include_adj_close:
            raise QlibSanitizationError("The closed Qlib contract does not allow adj_close in the emitted artifact.")

        try:
            factor_result = apply_factor_policy(frame, adjust_ohlcv=True)
        except FactorPolicyError as exc:
            raise QlibSanitizationError(str(exc)) from exc

        qlib_frame = factor_result.frame.copy()
        output_columns = ["date", "open", "high", "low", "close", "volume", "factor"]

        missing = [column for column in output_columns if column not in qlib_frame.columns]
        if missing:
            raise QlibSanitizationError(f"Qlib sanitization could not emit required columns: {missing}")

        qlib_frame = qlib_frame[output_columns].copy()
        qlib_frame["date"] = pd.to_datetime(qlib_frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        reference_adj_close = None
        if factor_result.factor_source == "adj_close_ratio" and "adj_close" in factor_result.frame.columns:
            reference_adj_close = pd.to_numeric(factor_result.frame["adj_close"], errors="coerce")

        contract = validate_qlib_frame(qlib_frame, reference_adj_close=reference_adj_close)
        warnings = list(factor_result.warnings) + list(contract.warnings)
        technical_report = {
            "factor_policy": factor_result.factor_policy,
            "factor_source": factor_result.factor_source,
            "factor_semantic_checks": list(factor_result.semantic_checks),
            "qlib_compatible": contract.compatible,
            "columns_emitted": list(qlib_frame.columns),
            "warnings": warnings,
            "reasons": list(contract.reasons),
            "contract_checks": list(contract.checks),
            "metrics": dict(contract.metrics),
        }
        if not contract.compatible:
            raise QlibSanitizationError(" | ".join(contract.reasons))

        return QlibSanitizationResult(
            frame=qlib_frame,
            factor_policy=factor_result.factor_policy,
            factor_source=factor_result.factor_source,
            technical_report=technical_report,
            warnings=warnings,
            contract=contract,
        )
