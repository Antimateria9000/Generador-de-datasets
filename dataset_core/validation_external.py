from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from dataset_core.reference_adapters import (
    EventReferenceAdapter,
    PriceReferenceAdapter,
    ReferenceAdapter,
    adapter_validation_scope,
    normalize_event_frame,
    normalize_reference_frame,
)
from dataset_core.settings import REFERENCE_RELATIVE_TOLERANCE, REFERENCE_SAMPLE_POINTS


@dataclass(frozen=True)
class ExternalValidationResult:
    report: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


class ExternalValidationService:
    def __init__(
        self,
        adapters: Iterable[ReferenceAdapter] | None = None,
        *,
        price_adapters: Iterable[PriceReferenceAdapter] | None = None,
        event_adapters: Iterable[EventReferenceAdapter] | None = None,
    ) -> None:
        self.price_adapters = list(price_adapters or [])
        self.event_adapters = list(event_adapters or [])

        for adapter in adapters or []:
            if adapter_validation_scope(adapter) == "event":
                self.event_adapters.append(adapter)  # type: ignore[arg-type]
            else:
                self.price_adapters.append(adapter)  # type: ignore[arg-type]

    def validate(
        self,
        frame: pd.DataFrame,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> ExternalValidationResult:
        dataset = normalize_reference_frame(frame)
        dataset_events = normalize_event_frame(frame)
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
        has_adapter_error = False
        has_validation_error = False

        for adapter in self.price_adapters:
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
                has_adapter_error = True
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "adapter_error",
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

            try:
                comparison_report = self._build_adapter_report(
                    adapter_name=adapter_name,
                    dataset=dataset,
                    reference=reference,
                )
            except Exception as exc:
                has_validation_error = True
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "validation_error",
                        "reason": f"Validation error: {exc}",
                        "score": None,
                    }
                )
                continue

            if comparison_report["status"] == "failed":
                has_failure = True
            else:
                validated_scores.append(float(comparison_report["score"]))
            adapter_reports.append(comparison_report)

        for adapter in self.event_adapters:
            adapter_name = adapter.name()
            try:
                reference_events = normalize_event_frame(adapter.fetch_events(symbol, start, end))
            except FileNotFoundError as exc:
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "not_validated",
                        "reason": str(exc),
                        "score": None,
                        "scope": "event",
                    }
                )
                continue
            except Exception as exc:
                has_adapter_error = True
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "adapter_error",
                        "reason": f"Adapter error: {exc}",
                        "score": None,
                        "scope": "event",
                    }
                )
                continue

            if reference_events.empty:
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "not_validated",
                        "reason": "Event reference adapter returned no manual events for the requested range.",
                        "score": None,
                        "scope": "event",
                    }
                )
                continue

            try:
                comparison_report = self._build_event_adapter_report(
                    adapter_name=adapter_name,
                    dataset_events=dataset_events,
                    reference_events=reference_events,
                )
            except Exception as exc:
                has_validation_error = True
                adapter_reports.append(
                    {
                        "adapter": adapter_name,
                        "status": "validation_error",
                        "reason": f"Validation error: {exc}",
                        "score": None,
                        "scope": "event",
                    }
                )
                continue

            if comparison_report["status"] == "failed":
                has_failure = True
            else:
                validated_scores.append(float(comparison_report["score"]))
            adapter_reports.append(comparison_report)

        overall_status = "not_validated"
        overall_reason = "No external reference was available."
        overall_score = None
        if validated_scores:
            overall_score = round(sum(validated_scores) / len(validated_scores), 3)
        if has_failure:
            overall_status = "failed"
            overall_reason = "At least one external adapter reported blocking differences."
        elif has_validation_error:
            overall_status = "validation_error"
            overall_reason = "The external validation pipeline failed after loading a reference."
        elif has_adapter_error:
            overall_status = "adapter_error"
            overall_reason = "At least one external adapter failed while loading a reference."
        elif validated_scores:
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
    def _relative_difference_report(
        dataset_values: pd.Series,
        reference_values: pd.Series,
    ) -> tuple[float, int]:
        dataset_numeric = pd.to_numeric(dataset_values, errors="coerce")
        reference_numeric = pd.to_numeric(reference_values, errors="coerce")
        delta = (dataset_numeric - reference_numeric).abs()
        comparable_base = reference_numeric.abs() > 1e-12

        relative = pd.Series(0.0, index=delta.index, dtype="float64")
        if comparable_base.any():
            relative.loc[comparable_base] = (
                delta.loc[comparable_base] / reference_numeric.loc[comparable_base].abs()
            ).fillna(0.0)

        zero_reference_mismatch_mask = (~comparable_base) & (delta > 1e-12)
        zero_reference_mismatch_count = int(zero_reference_mismatch_mask.sum())
        if zero_reference_mismatch_count:
            relative.loc[zero_reference_mismatch_mask] = 1.0

        return (float(relative.max()) if not relative.empty else 0.0, zero_reference_mismatch_count)

    def _build_adapter_report(
        self,
        *,
        adapter_name: str,
        dataset: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> dict[str, object]:
        merged = dataset.merge(reference, on="date", how="inner", suffixes=("_dataset", "_reference"))
        dataset_dates = set(dataset["date"])
        reference_dates = set(reference["date"])
        missing_dataset_dates = len(dataset_dates - reference_dates)
        missing_reference_dates = len(reference_dates - dataset_dates)
        gap_count = missing_dataset_dates + missing_reference_dates

        numeric_base = ["open", "high", "low", "close", "volume", "adj_close"]
        relative_differences: dict[str, float] = {}
        zero_reference_mismatches: dict[str, int] = {}
        zero_reference_mismatch_count = 0
        for column in numeric_base:
            dataset_column = f"{column}_dataset"
            reference_column = f"{column}_reference"
            if dataset_column not in merged.columns or reference_column not in merged.columns:
                continue

            max_relative_diff, zero_mismatch_count = self._relative_difference_report(
                merged[dataset_column],
                merged[reference_column],
            )
            relative_differences[column] = max_relative_diff
            if zero_mismatch_count:
                zero_reference_mismatches[column] = zero_mismatch_count
            zero_reference_mismatch_count += zero_mismatch_count

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
        denominator = max(len(dataset), len(reference))
        gap_ratio = 0.0 if denominator == 0 else gap_count / denominator
        score = max(
            0.0,
            100.0
            - (gap_ratio * 60.0)
            - (max_relative_diff * 1000.0)
            - (split_mismatch_count * 10.0)
            - (zero_reference_mismatch_count * 10.0),
        )

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
        if zero_reference_mismatch_count > 0:
            status = "failed"
            reasons.append(
                "Reference rows with zero baseline diverge from the dataset in numeric columns: "
                + ", ".join(f"{column}={count}" for column, count in zero_reference_mismatches.items())
                + "."
            )
        if split_mismatch_count > 0:
            status = "failed"
            reasons.append(f"Split mismatch count: {split_mismatch_count}.")

        return {
            "adapter": adapter_name,
            "status": status,
            "reason": None if not reasons else " | ".join(reasons),
            "score": round(score, 3),
            "scope": "price",
            "overlap_rows": overlap,
            "gap_count": gap_count,
            "gap_ratio": round(gap_ratio, 6),
            "max_relative_diff": round(max_relative_diff, 8),
            "zero_reference_mismatch_count": zero_reference_mismatch_count,
            "zero_reference_mismatches": zero_reference_mismatches,
            "split_mismatch_count": split_mismatch_count,
            "sample_dates": sample_dates,
        }

    @staticmethod
    def _build_event_adapter_report(
        *,
        adapter_name: str,
        dataset_events: pd.DataFrame,
        reference_events: pd.DataFrame,
    ) -> dict[str, object]:
        if reference_events.empty:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": "No manual events were provided.",
                "score": None,
                "scope": "event",
            }

        if dataset_events.empty:
            dataset_lookup = pd.DataFrame(columns=["date", "dividends", "stock_splits"]).set_index("date")
        else:
            dataset_lookup = dataset_events.copy().set_index("date")

        mismatches: list[str] = []
        matched_events = 0
        checked_events = 0
        checked_columns: set[str] = set()

        for row in reference_events.itertuples(index=False):
            date_value = pd.Timestamp(row.date)
            dataset_row = dataset_lookup.loc[date_value] if date_value in dataset_lookup.index else None

            for column in ("dividends", "stock_splits"):
                reference_value = float(getattr(row, column, 0.0) or 0.0)
                if abs(reference_value) <= 1e-12:
                    continue
                checked_events += 1
                checked_columns.add(column)

                dataset_value = 0.0
                if dataset_row is not None:
                    if isinstance(dataset_row, pd.DataFrame):
                        dataset_value = float(pd.to_numeric(dataset_row[column], errors="coerce").fillna(0.0).iloc[-1])
                    else:
                        dataset_value = float(pd.to_numeric(pd.Series([dataset_row[column]]), errors="coerce").fillna(0.0).iloc[0])

                if abs(dataset_value - reference_value) <= 1e-8:
                    matched_events += 1
                else:
                    mismatches.append(
                        f"{column} mismatch at {date_value.date().isoformat()}: dataset={dataset_value} reference={reference_value}"
                    )

        score = 100.0 if checked_events == 0 else max(0.0, 100.0 * (matched_events / checked_events))
        return {
            "adapter": adapter_name,
            "status": "passed" if not mismatches else "failed",
            "reason": None if not mismatches else " | ".join(mismatches),
            "score": round(score, 3),
            "scope": "event",
            "checked_event_count": checked_events,
            "matched_event_count": matched_events,
            "mismatch_count": len(mismatches),
            "checked_columns": sorted(checked_columns),
            "sample_dates": reference_events["date"].head(REFERENCE_SAMPLE_POINTS).dt.strftime("%Y-%m-%d").tolist(),
        }

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
