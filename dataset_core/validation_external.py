from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from dataset_core.reference_adapters import ReferenceAdapter, normalize_reference_frame
from dataset_core.settings import REFERENCE_RELATIVE_TOLERANCE, REFERENCE_SAMPLE_POINTS


@dataclass(frozen=True)
class ExternalValidationResult:
    report: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


class ExternalValidationService:
    def __init__(self, adapters: Iterable[ReferenceAdapter] | None = None) -> None:
        self.adapters = list(adapters or [])

    def validate(
        self,
        frame: pd.DataFrame,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> ExternalValidationResult:
        dataset = normalize_reference_frame(frame)
        adapter_reports: list[dict[str, object]] = []

        if dataset.empty:
            return ExternalValidationResult(
                report={
                    "status": "not_validated",
                    "score": None,
                    "reason": "Dataset is empty; external validation was skipped.",
                    "adapter_reports": adapter_reports,
                }
            )

        validated_scores: list[float] = []
        has_failure = False

        for adapter in self.adapters:
            adapter_name = adapter.name()
            try:
                reference = normalize_reference_frame(adapter.fetch_reference(symbol, start, end))
            except FileNotFoundError as exc:
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "not_validated",
                        "reason": str(exc),
                        "score": None,
                    }
                )
                continue
            except Exception as exc:
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "not_validated",
                        "reason": f"Adapter error: {exc}",
                        "score": None,
                    }
                )
                continue

            if reference.empty:
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "not_validated",
                        "reason": "Reference adapter returned no rows for the requested range.",
                        "score": None,
                    }
                )
                continue

            merged = dataset.merge(reference, on="date", how="inner", suffixes=("_dataset", "_reference"))
            dataset_dates = set(dataset["date"])
            reference_dates = set(reference["date"])
            missing_dataset_dates = len(dataset_dates - reference_dates)
            missing_reference_dates = len(reference_dates - dataset_dates)
            gap_count = missing_dataset_dates + missing_reference_dates

            numeric_base = ["open", "high", "low", "close", "volume", "adj_close"]
            relative_differences: dict[str, float] = {}
            for column in numeric_base:
                dataset_column = f"{column}_dataset"
                reference_column = f"{column}_reference"
                if dataset_column not in merged.columns or reference_column not in merged.columns:
                    continue

                base = pd.to_numeric(merged[reference_column], errors="coerce").abs().replace(0, pd.NA)
                delta = (
                    pd.to_numeric(merged[dataset_column], errors="coerce")
                    - pd.to_numeric(merged[reference_column], errors="coerce")
                ).abs()
                relative = (delta / base).fillna(0.0)
                relative_differences[column] = float(relative.max()) if not relative.empty else 0.0

            split_mismatch_count = 0
            if "stock_splits_dataset" in merged.columns and "stock_splits_reference" in merged.columns:
                split_mismatch_count = int(
                    (
                        pd.to_numeric(merged["stock_splits_dataset"], errors="coerce").fillna(0.0).round(8)
                        != pd.to_numeric(merged["stock_splits_reference"], errors="coerce").fillna(0.0).round(8)
                    ).sum()
                )

            max_relative_diff = max(relative_differences.values(), default=0.0)
            overlap = len(merged)
            gap_ratio = 0.0 if max(len(dataset), len(reference)) == 0 else gap_count / max(len(dataset), len(reference))
            score = max(0.0, 100.0 - (gap_ratio * 60.0) - (max_relative_diff * 1000.0) - (split_mismatch_count * 10.0))
            validated_scores.append(score)

            sample_dates = merged["date"].head(REFERENCE_SAMPLE_POINTS).dt.strftime("%Y-%m-%d").tolist()
            status = "passed"
            reasons: list[str] = []
            if overlap == 0:
                status = "failed"
                reasons.append("No overlapping dates between dataset and reference.")
            if gap_ratio > 0.10:
                status = "failed"
                reasons.append(f"Calendar gap ratio too high: {gap_ratio:.2%}.")
            if max_relative_diff > REFERENCE_RELATIVE_TOLERANCE:
                status = "failed"
                reasons.append(
                    f"Relative OHLCV difference exceeds tolerance ({REFERENCE_RELATIVE_TOLERANCE:.4f})."
                )
            if split_mismatch_count > 0:
                status = "failed"
                reasons.append(f"Split mismatch count: {split_mismatch_count}.")

            if status == "failed":
                has_failure = True

            adapter_reports.append(
                {
                    "adapter": adapter_name,
                    "status": status,
                    "reason": None if not reasons else " | ".join(reasons),
                    "score": round(score, 3),
                    "overlap_rows": overlap,
                    "gap_count": gap_count,
                    "gap_ratio": round(gap_ratio, 6),
                    "max_relative_diff": round(max_relative_diff, 8),
                    "split_mismatch_count": split_mismatch_count,
                    "sample_dates": sample_dates,
                }
            )

        overall_status = "not_validated"
        overall_reason = "No external reference was available."
        overall_score = None
        if validated_scores:
            overall_score = round(sum(validated_scores) / len(validated_scores), 3)
            if has_failure:
                overall_status = "failed"
                overall_reason = "At least one external adapter reported blocking differences."
            else:
                overall_status = "passed"
                overall_reason = None

        return ExternalValidationResult(
            report={
                "status": overall_status,
                "score": overall_score,
                "reason": overall_reason,
                "adapter_reports": adapter_reports,
            }
        )

    @staticmethod
    def render_text(report: dict[str, object], symbol: str) -> str:
        lines = [
            f"External validation report for {symbol}",
            f"status: {report.get('status')}",
            f"score: {report.get('score')}",
            f"reason: {report.get('reason')}",
            "",
        ]

        for adapter_report in report.get("adapter_reports", []):
            lines.append(f"- adapter: {adapter_report.get('adapter')}")
            lines.append(f"  status: {adapter_report.get('status')}")
            lines.append(f"  score: {adapter_report.get('score')}")
            lines.append(f"  reason: {adapter_report.get('reason')}")

        return "\n".join(lines).strip() + "\n"
