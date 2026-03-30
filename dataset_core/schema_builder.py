from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dataset_core.factor_policy import FactorPolicyError, apply_factor_policy
from dataset_core.presets import ResolvedPreset, resolve_preset
from dataset_core.qlib_contract import validate_qlib_frame


class SchemaBuildError(ValueError):
    """Raised when a dataset cannot be shaped into a valid output schema."""


@dataclass(frozen=True)
class SchemaBuildResult:
    frame: pd.DataFrame
    resolved_preset: ResolvedPreset
    warnings: list[str] = field(default_factory=list)
    factor_policy: str | None = None
    factor_source: str | None = None
    qlib_compatible: bool = False
    qlib_reasons: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)


class DatasetSchemaBuilder:
    def build(self, frame: pd.DataFrame, mode: str, extras: list[str]) -> SchemaBuildResult:
        if frame is None or frame.empty:
            raise SchemaBuildError("The sanitized frame is empty.")

        working = frame.copy()
        try:
            resolved_preset = resolve_preset(mode, extras)
        except ValueError as exc:
            raise SchemaBuildError(str(exc)) from exc
        warnings = [
            f"Extra ignored for preset {resolved_preset.preset.name}: {item}"
            for item in resolved_preset.ignored_extras
        ]
        factor_policy = None
        factor_source = None
        qlib_reasons: list[str] = []

        if resolved_preset.preset.name != "qlib" and "factor" in resolved_preset.selected_extras and "factor" not in working.columns:
            try:
                factor_result = apply_factor_policy(working, adjust_ohlcv=False)
            except FactorPolicyError as exc:
                raise SchemaBuildError(str(exc)) from exc
            working = factor_result.frame
            factor_policy = factor_result.factor_policy
            factor_source = factor_result.factor_source
            warnings.extend(factor_result.warnings)

        final_columns = list(resolved_preset.output_columns)
        missing = [column for column in final_columns if column not in working.columns]
        if missing:
            raise SchemaBuildError(f"Output schema requested missing columns: {missing}")

        output = working[final_columns].copy()
        output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        qlib_compatible = False
        if resolved_preset.preset.name == "qlib":
            contract = validate_qlib_frame(output)
            qlib_compatible = contract.compatible
            qlib_reasons.extend(contract.reasons)

        return SchemaBuildResult(
            frame=output,
            resolved_preset=resolved_preset,
            warnings=warnings,
            factor_policy=factor_policy,
            factor_source=factor_source,
            qlib_compatible=qlib_compatible,
            qlib_reasons=qlib_reasons,
        )
