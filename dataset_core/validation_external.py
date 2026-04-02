from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import pandas as pd

from dataset_core.external_sources.base import (
    ExternalSourceAuthError,
    ExternalSourceCoverageError,
    ExternalSourceNetworkError,
    ExternalSourceNotFoundError,
    ExternalSourcePayloadError,
    ExternalSourceRateLimitError,
    extract_source_metadata,
)
from dataset_core.reference_adapters import (
    EventReferenceAdapter,
    PriceReferenceAdapter,
    ReferenceAdapter,
    adapter_validation_scope,
    normalize_event_frame,
    normalize_reference_frame,
)
from dataset_core.settings import REFERENCE_RELATIVE_TOLERANCE, REFERENCE_SAMPLE_POINTS

CoverageStatus = Literal["full", "partial", "none"]
ComparisonStatus = Literal["passed", "failed", "not_validated", "adapter_error", "validation_error"]

_PRICE_BLOCKING_COLUMNS = ("open", "high", "low", "close")
_PRICE_ADVISORY_COLUMNS = ("adj_close", "volume")
_PARTIAL_PRICE_MIN_OVERLAP_ROWS = 60
_PARTIAL_PRICE_END_TOLERANCE_DAYS = 7
_SYSTEMATIC_DIVIDEND_SCALE_MIN_RECORDS = 2
_SYSTEMATIC_DIVIDEND_SCALE_STABILITY_STD = 1e-4
_SYSTEMATIC_DIVIDEND_SCALE_IDENTITY_TOL = 0.01

_PARTIAL_KIND_COVERAGE_LIMITED = "coverage_limited"
_PARTIAL_KIND_METHODOLOGY_CAVEAT = "methodology_caveat"
_PARTIAL_KIND_ADVISORY_ZERO_BASELINE = "advisory_zero_baseline_mismatch"


@dataclass(frozen=True)
class ExternalValidationResult:
    report: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


def _unique_non_empty(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


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
        if dataset.empty:
            return ExternalValidationResult(
                {
                    "status": "not_validated",
                    "coverage_status": "none",
                    "comparison_status": "not_validated",
                    "partial_validation_kinds": [],
                    "score": None,
                    "reason": "Dataset is empty; external validation was skipped.",
                    "adapter_reports": [],
                }
            )

        adapter_reports: list[dict[str, object]] = []
        validated_scores: list[float] = []

        for adapter in self.price_adapters:
            report, score_bucket = self._validate_price_adapter(adapter, dataset, symbol, start, end)
            adapter_reports.append(report)
            if str(report.get("comparison_status")) == "passed" and score_bucket is not None:
                validated_scores.append(score_bucket)

        for adapter in self.event_adapters:
            report, score_bucket = self._validate_event_adapter(adapter, dataset_events, symbol, start, end)
            adapter_reports.append(report)
            if str(report.get("comparison_status")) == "passed" and score_bucket is not None:
                validated_scores.append(score_bucket)

        comparison_status = self._aggregate_comparison_status(adapter_reports)
        coverage_status = self._aggregate_coverage_status(adapter_reports)
        partial_kinds = self._aggregate_partial_validation_kinds(adapter_reports)
        overall_status = self._compose_status(
            comparison_status=comparison_status,
            coverage_status=coverage_status,
            partial_validation_kinds=partial_kinds,
        )
        overall_score = None if not validated_scores else round(sum(validated_scores) / len(validated_scores), 3)

        return ExternalValidationResult(
            {
                "status": overall_status,
                "coverage_status": coverage_status,
                "comparison_status": comparison_status,
                "partial_validation_kinds": partial_kinds,
                "score": overall_score,
                "reason": self._build_overall_reason(
                    comparison_status=comparison_status,
                    coverage_status=coverage_status,
                    partial_validation_kinds=partial_kinds,
                ),
                "adapter_reports": adapter_reports,
            }
        )

    def _validate_price_adapter(
        self,
        adapter,
        dataset: pd.DataFrame,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> tuple[dict[str, object], float | None]:
        adapter_name = adapter.name()
        try:
            raw_reference = adapter.fetch_reference(symbol, start, end)
            reference = normalize_reference_frame(raw_reference)
            source_metadata = extract_source_metadata(reference)
        except FileNotFoundError as exc:
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                coverage_status="none",
                comparison_status="not_validated",
                coverage_reason="Reference file is missing.",
                comparison_reason=str(exc),
            ), None
        except ExternalSourceNotFoundError as exc:
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                coverage_status="none",
                comparison_status="not_validated",
                coverage_reason="The external source could not resolve the requested symbol.",
                comparison_reason=str(exc),
                error_kind=self._adapter_error_kind(exc),
            ), None
        except ExternalSourceCoverageError as exc:
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                coverage_status="partial",
                comparison_status="not_validated",
                coverage_reason="The external source reported limited plan or range coverage.",
                comparison_reason=str(exc),
                error_kind=self._adapter_error_kind(exc),
            ), None
        except (ExternalSourceAuthError, ExternalSourceNetworkError, ExternalSourceRateLimitError, ExternalSourcePayloadError) as exc:
            return self._build_error_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                comparison_status="adapter_error",
                comparison_reason=f"Adapter error: {exc}",
                error_kind=self._adapter_error_kind(exc),
            ), None
        except Exception as exc:
            return self._build_error_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                comparison_status="adapter_error",
                comparison_reason=f"Adapter error: {exc}",
            ), None

        if reference.empty:
            coverage_status, coverage_reason = self._empty_reference_coverage(source_metadata)
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                coverage_status=coverage_status,
                comparison_status="not_validated",
                coverage_reason=coverage_reason,
                comparison_reason="Reference adapter returned no rows for the requested range.",
                source_metadata=source_metadata,
            ), None

        try:
            report = self._build_price_adapter_report(
                adapter_name=adapter_name,
                dataset=dataset,
                reference=reference,
                source_metadata=source_metadata,
                requested_start=start,
                requested_end=end,
            )
        except Exception as exc:
            return self._build_error_adapter_report(
                adapter_name=adapter_name,
                scope="price",
                comparison_status="validation_error",
                comparison_reason=f"Validation error: {exc}",
            ), None

        report["source_metadata"] = source_metadata
        return report, None if report.get("score") is None else float(report["score"])

    def _validate_event_adapter(
        self,
        adapter,
        dataset_events: pd.DataFrame,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> tuple[dict[str, object], float | None]:
        adapter_name = adapter.name()
        try:
            raw_reference = adapter.fetch_events(symbol, start, end)
            reference = normalize_event_frame(raw_reference)
            source_metadata = extract_source_metadata(reference)
        except FileNotFoundError as exc:
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                coverage_status="none",
                comparison_status="not_validated",
                coverage_reason="Event reference file is missing.",
                comparison_reason=str(exc),
            ), None
        except ExternalSourceNotFoundError as exc:
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                coverage_status="none",
                comparison_status="not_validated",
                coverage_reason="The external source could not resolve the requested symbol.",
                comparison_reason=str(exc),
                error_kind=self._adapter_error_kind(exc),
            ), None
        except ExternalSourceCoverageError as exc:
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                coverage_status="partial",
                comparison_status="not_validated",
                coverage_reason="The external source reported limited coverage for corporate actions.",
                comparison_reason=str(exc),
                error_kind=self._adapter_error_kind(exc),
            ), None
        except (ExternalSourceAuthError, ExternalSourceNetworkError, ExternalSourceRateLimitError, ExternalSourcePayloadError) as exc:
            return self._build_error_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                comparison_status="adapter_error",
                comparison_reason=f"Adapter error: {exc}",
                error_kind=self._adapter_error_kind(exc),
            ), None
        except Exception as exc:
            return self._build_error_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                comparison_status="adapter_error",
                comparison_reason=f"Adapter error: {exc}",
            ), None

        if reference.empty:
            coverage_status, coverage_reason = self._empty_reference_coverage(source_metadata)
            return self._build_non_comparable_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                coverage_status=coverage_status,
                comparison_status="not_validated",
                coverage_reason=coverage_reason,
                comparison_reason="Event reference adapter returned no manual events for the requested range.",
                source_metadata=source_metadata,
            ), None

        try:
            report = self._build_event_adapter_report(
                adapter_name=adapter_name,
                dataset_events=dataset_events,
                reference_events=reference,
                source_metadata=source_metadata,
            )
        except Exception as exc:
            return self._build_error_adapter_report(
                adapter_name=adapter_name,
                scope="event",
                comparison_status="validation_error",
                comparison_reason=f"Validation error: {exc}",
            ), None

        report["source_metadata"] = source_metadata
        return report, None if report.get("score") is None else float(report["score"])

    @staticmethod
    def _build_non_comparable_adapter_report(
        *,
        adapter_name: str,
        scope: Literal["price", "event"],
        coverage_status: CoverageStatus,
        comparison_status: ComparisonStatus,
        coverage_reason: str | None,
        comparison_reason: str | None,
        source_metadata: dict[str, object] | None = None,
        error_kind: str | None = None,
    ) -> dict[str, object]:
        report = {
            "adapter": adapter_name,
            "status": comparison_status,
            "coverage_status": coverage_status,
            "comparison_status": comparison_status,
            "coverage_reason": coverage_reason,
            "comparison_reason": comparison_reason,
            "partial_validation_kinds": [],
            "score": None,
            "scope": scope,
            "reason": ExternalValidationService._join_reason_parts(coverage_reason, comparison_reason),
        }
        if source_metadata:
            report["source_metadata"] = dict(source_metadata)
        if error_kind:
            report["error_kind"] = error_kind
        return report

    @staticmethod
    def _build_error_adapter_report(
        *,
        adapter_name: str,
        scope: Literal["price", "event"],
        comparison_status: ComparisonStatus,
        comparison_reason: str,
        error_kind: str | None = None,
    ) -> dict[str, object]:
        report = {
            "adapter": adapter_name,
            "status": comparison_status,
            "coverage_status": "none",
            "comparison_status": comparison_status,
            "coverage_reason": None,
            "comparison_reason": comparison_reason,
            "partial_validation_kinds": [],
            "score": None,
            "scope": scope,
            "reason": comparison_reason,
        }
        if error_kind:
            report["error_kind"] = error_kind
        return report

    @staticmethod
    def _join_reason_parts(*parts: object) -> str | None:
        items = _unique_non_empty(parts)
        return None if not items else " | ".join(items)

    @staticmethod
    def _aggregate_partial_validation_kinds(adapter_reports: list[dict[str, object]]) -> list[str]:
        kinds: list[str] = []
        for report in adapter_reports:
            kinds.extend(str(item) for item in report.get("partial_validation_kinds", []) if str(item).strip())
        return _unique_non_empty(kinds)

    @staticmethod
    def _aggregate_comparison_status(adapter_reports: list[dict[str, object]]) -> ComparisonStatus:
        statuses = [str(report.get("comparison_status") or "").strip().lower() for report in adapter_reports]
        if any(status == "failed" for status in statuses):
            return "failed"
        if any(status == "validation_error" for status in statuses):
            return "validation_error"
        if any(status == "adapter_error" for status in statuses):
            return "adapter_error"
        if any(status == "passed" for status in statuses):
            return "passed"
        return "not_validated"

    @staticmethod
    def _aggregate_coverage_status(adapter_reports: list[dict[str, object]]) -> CoverageStatus:
        if not adapter_reports:
            return "none"
        coverage_statuses = [str(report.get("coverage_status") or "").strip().lower() for report in adapter_reports]
        if all(status == "none" for status in coverage_statuses):
            return "none"
        clean_full = all(
            str(report.get("coverage_status")) == "full"
            and str(report.get("comparison_status")) == "passed"
            and not report.get("partial_validation_kinds")
            for report in adapter_reports
        )
        if clean_full:
            return "full"
        return "partial"

    @staticmethod
    def _compose_status(
        *,
        comparison_status: ComparisonStatus,
        coverage_status: CoverageStatus,
        partial_validation_kinds: list[str],
    ) -> str:
        if comparison_status != "passed":
            return comparison_status
        if coverage_status == "full" and not partial_validation_kinds:
            return "passed"
        return "passed_partial"

    @staticmethod
    def _build_overall_reason(
        *,
        comparison_status: ComparisonStatus,
        coverage_status: CoverageStatus,
        partial_validation_kinds: list[str],
    ) -> str | None:
        if comparison_status == "failed":
            return "At least one external adapter reported blocking differences inside the validated overlap."
        if comparison_status == "validation_error":
            return "The external validation pipeline failed after loading a reference."
        if comparison_status == "adapter_error":
            return "At least one external adapter failed while loading a reference."
        if comparison_status == "not_validated":
            if coverage_status == "partial":
                return "External references only offered partial coverage and produced no comparable overlap."
            return "No external reference was available."

        notes: list[str] = []
        if coverage_status == "partial":
            notes.append("External validation passed on the provider-covered or otherwise validated overlap only.")
        if _PARTIAL_KIND_METHODOLOGY_CAVEAT in partial_validation_kinds:
            notes.append("At least one adapter reported a provider methodology caveat.")
        if _PARTIAL_KIND_ADVISORY_ZERO_BASELINE in partial_validation_kinds:
            notes.append("At least one advisory numeric column diverged against a zero reference baseline.")
        return None if not notes else " ".join(_unique_non_empty(notes))

    @staticmethod
    def _adapter_error_kind(exc: Exception) -> str:
        if isinstance(exc, ExternalSourceAuthError):
            return "auth_error"
        if isinstance(exc, ExternalSourceRateLimitError):
            return "rate_limited"
        if isinstance(exc, ExternalSourceNotFoundError):
            return "symbol_not_found"
        if isinstance(exc, ExternalSourceCoverageError):
            return "coverage_insufficient"
        if isinstance(exc, ExternalSourcePayloadError):
            return "provider_schema_error"
        if isinstance(exc, ExternalSourceNetworkError):
            return "transport_error"
        return type(exc).__name__

    @staticmethod
    def _relative_difference_report(dataset_values: pd.Series, reference_values: pd.Series) -> tuple[float, int]:
        dataset_numeric = pd.to_numeric(dataset_values, errors="coerce")
        reference_numeric = pd.to_numeric(reference_values, errors="coerce")
        delta = (dataset_numeric - reference_numeric).abs()
        comparable_base = reference_numeric.abs() > 1e-12
        relative = pd.Series(0.0, index=delta.index, dtype="float64")
        if comparable_base.any():
            relative.loc[comparable_base] = (
                delta.loc[comparable_base] / reference_numeric.loc[comparable_base].abs()
            ).fillna(0.0)
        zero_reference_mask = (~comparable_base) & (delta > 1e-12)
        zero_reference_count = int(zero_reference_mask.sum())
        if zero_reference_count:
            relative.loc[zero_reference_mask] = 1.0
        return (float(relative.max()) if not relative.empty else 0.0, zero_reference_count)

    @staticmethod
    def _normalize_timestamp(value: object) -> pd.Timestamp | None:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return None
        return pd.Timestamp(timestamp).tz_convert("UTC").tz_localize(None)

    def _resolve_price_coverage(
        self,
        *,
        dataset: pd.DataFrame,
        reference: pd.DataFrame,
        source_metadata: dict[str, object] | None,
        requested_start: str | None,
        requested_end: str | None,
    ) -> dict[str, object]:
        metadata = dict(source_metadata or {})
        effective_start = self._normalize_timestamp(metadata.get("effective_start"))
        effective_end = self._normalize_timestamp(metadata.get("effective_end"))
        requested_start_ts = self._normalize_timestamp(requested_start)
        requested_end_ts = self._normalize_timestamp(requested_end)
        provider_limited = bool(metadata.get("partial_coverage"))
        if not provider_limited and requested_start_ts is not None and effective_start is not None:
            provider_limited = bool(effective_start > requested_start_ts)
        if not provider_limited and requested_end_ts is not None and effective_end is not None:
            provider_limited = bool(
                effective_end < (requested_end_ts - pd.Timedelta(days=_PARTIAL_PRICE_END_TOLERANCE_DAYS))
            )

        if provider_limited:
            if effective_start is None and not reference.empty:
                effective_start = pd.Timestamp(reference["date"].min())
            if effective_end is None and not reference.empty:
                effective_end = pd.Timestamp(reference["date"].max())
            dataset_window = dataset
            if effective_start is not None:
                dataset_window = dataset_window[dataset_window["date"] >= effective_start]
            if effective_end is not None:
                dataset_window = dataset_window[dataset_window["date"] < effective_end]
            dataset_window = dataset_window.reset_index(drop=True)
            uncovered_prefix_count = 0 if effective_start is None else int((dataset["date"] < effective_start).sum())
            uncovered_suffix_count = 0 if effective_end is None else int((dataset["date"] >= effective_end).sum())
            coverage_status: CoverageStatus = "partial"
            coverage_reason = "Reference coverage is limited to the provider-declared validation window."
        else:
            dataset_window = dataset.reset_index(drop=True)
            uncovered_prefix_count = 0
            uncovered_suffix_count = 0
            coverage_status = "full" if not reference.empty else "none"
            coverage_reason = None if coverage_status == "full" else "Reference coverage is unavailable for the requested range."

        partial_recent_suffix = (
            provider_limited
            and uncovered_prefix_count > 0
            and uncovered_suffix_count <= _PARTIAL_PRICE_END_TOLERANCE_DAYS
            and len(dataset_window) >= _PARTIAL_PRICE_MIN_OVERLAP_ROWS
        )
        return {
            "coverage_status": coverage_status,
            "coverage_reason": coverage_reason,
            "provider_limited": provider_limited,
            "partial_recent_suffix": partial_recent_suffix,
            "dataset_window": dataset_window,
            "dataset_rows": int(len(dataset)),
            "dataset_window_rows": int(len(dataset_window)),
            "reference_rows": int(len(reference)),
            "effective_start": None if effective_start is None else effective_start.isoformat(),
            "effective_end": None if effective_end is None else effective_end.isoformat(),
            "uncovered_prefix_count": uncovered_prefix_count,
            "uncovered_suffix_count": uncovered_suffix_count,
        }

    @staticmethod
    def _build_price_advisory_notes(advisory_differences: dict[str, float]) -> list[str]:
        notes: list[str] = []
        if advisory_differences.get("adj_close", 0.0) > REFERENCE_RELATIVE_TOLERANCE:
            notes.append("Adjusted-close values differ materially across providers and were treated as advisory only.")
        if advisory_differences.get("volume", 0.0) > REFERENCE_RELATIVE_TOLERANCE:
            notes.append("Volume differs materially across providers and was treated as advisory only.")
        return notes

    @staticmethod
    def _detect_systematic_dividend_scale_mismatch(
        mismatch_records: list[dict[str, object]],
        *,
        matched_events: int,
        date_aligned_events: int,
    ) -> dict[str, object] | None:
        dividend_records = [
            record
            for record in mismatch_records
            if record.get("column") == "dividends"
            and abs(float(record.get("dataset_value", 0.0) or 0.0)) > 1e-12
            and abs(float(record.get("reference_value", 0.0) or 0.0)) > 1e-12
        ]
        mismatch_count = len(dividend_records)
        if mismatch_count < _SYSTEMATIC_DIVIDEND_SCALE_MIN_RECORDS or mismatch_count != len(mismatch_records):
            return None
        if mismatch_count == 2 and (date_aligned_events != mismatch_count or matched_events != 0):
            return None
        ratios = pd.Series(
            [float(record["reference_value"]) / float(record["dataset_value"]) for record in dividend_records],
            dtype="float64",
        )
        rounded = ratios.round(4)
        dominant_ratio = float(rounded.mode().iloc[0])
        dominant_mask = rounded == dominant_ratio
        dominant_count = int(dominant_mask.sum())
        minimum_dominant_count = max(_SYSTEMATIC_DIVIDEND_SCALE_MIN_RECORDS, math.ceil(mismatch_count * 0.8))
        if dominant_count < minimum_dominant_count:
            return None
        if (
            float(ratios[dominant_mask].std(ddof=0)) > _SYSTEMATIC_DIVIDEND_SCALE_STABILITY_STD
            or abs(dominant_ratio - 1.0) <= _SYSTEMATIC_DIVIDEND_SCALE_IDENTITY_TOL
        ):
            return None
        return {
            "dominant_ratio": dominant_ratio,
            "matched_events": matched_events,
            "date_aligned_events": date_aligned_events,
            "mismatch_count": mismatch_count,
        }

    @staticmethod
    def _empty_reference_coverage(source_metadata: dict[str, object] | None) -> tuple[CoverageStatus, str | None]:
        metadata = dict(source_metadata or {})
        if bool(metadata.get("partial_coverage")):
            return "partial", "The source declared partial coverage, but no comparable rows remained after normalization/filtering."
        return "none", None

    def _build_price_adapter_report(
        self,
        *,
        adapter_name: str,
        dataset: pd.DataFrame,
        reference: pd.DataFrame,
        source_metadata: dict[str, object] | None = None,
        requested_start: str | None = None,
        requested_end: str | None = None,
    ) -> dict[str, object]:
        coverage = self._resolve_price_coverage(
            dataset=dataset,
            reference=reference,
            source_metadata=source_metadata,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        dataset_window = coverage["dataset_window"]
        merged = dataset_window.merge(reference, on="date", how="inner", suffixes=("_dataset", "_reference"))
        dataset_dates = set(dataset_window["date"])
        reference_dates = set(reference["date"])
        gap_count = len(dataset_dates - reference_dates) + len(reference_dates - dataset_dates)

        blocking_differences: dict[str, float] = {}
        advisory_differences: dict[str, float] = {}
        zero_reference_mismatches: dict[str, int] = {}
        blocking_zero_reference_count = 0
        advisory_zero_reference_count = 0
        for column in (*_PRICE_BLOCKING_COLUMNS, *_PRICE_ADVISORY_COLUMNS):
            left = f"{column}_dataset"
            right = f"{column}_reference"
            if left not in merged.columns or right not in merged.columns:
                continue
            max_relative_diff, zero_count = self._relative_difference_report(merged[left], merged[right])
            if column in _PRICE_BLOCKING_COLUMNS:
                blocking_differences[column] = max_relative_diff
                blocking_zero_reference_count += zero_count
            else:
                advisory_differences[column] = max_relative_diff
                advisory_zero_reference_count += zero_count
            if zero_count:
                zero_reference_mismatches[column] = zero_count

        split_mismatch_count = 0
        if "stock_splits_dataset" in merged.columns and "stock_splits_reference" in merged.columns:
            split_mismatch_count = int(
                (
                    pd.to_numeric(merged["stock_splits_dataset"], errors="coerce").fillna(0.0).round(8)
                    != pd.to_numeric(merged["stock_splits_reference"], errors="coerce").fillna(0.0).round(8)
                ).sum()
            )

        overlap = len(merged)
        gap_ratio = 0.0 if max(len(dataset_window), len(reference)) == 0 else gap_count / max(len(dataset_window), len(reference))
        max_relative_diff = max(blocking_differences.values(), default=0.0)
        max_advisory_relative_diff = max(advisory_differences.values(), default=0.0)
        score = max(
            0.0,
            100.0
            - (gap_ratio * 60.0)
            - (max_relative_diff * 1000.0)
            - (split_mismatch_count * 10.0)
            - (blocking_zero_reference_count * 10.0),
        )

        coverage_status = coverage["coverage_status"]
        coverage_reason = coverage["coverage_reason"]
        comparison_status: ComparisonStatus = "passed"
        comparison_notes: list[str] = []
        partial_validation_kinds: list[str] = []

        if coverage["provider_limited"]:
            partial_validation_kinds.append(_PARTIAL_KIND_COVERAGE_LIMITED)

        if overlap == 0:
            if dataset_window.empty:
                comparison_status = "not_validated"
                comparison_notes.append("No overlapping dates were available inside the provider-covered validation window.")
            else:
                comparison_status = "failed"
                comparison_notes.append("No overlapping dates were available inside the effective validation window.")
        if comparison_status != "not_validated" and gap_ratio > 0.10:
            comparison_status = "failed"
            comparison_notes.append(f"Calendar gap ratio too high inside the validated overlap: {gap_ratio:.2%}.")
        if max_relative_diff > REFERENCE_RELATIVE_TOLERANCE:
            comparison_status = "failed"
            comparison_notes.append(
                f"Relative OHLC difference exceeds tolerance ({REFERENCE_RELATIVE_TOLERANCE:.4f}) inside the validated overlap."
            )
        if blocking_zero_reference_count > 0:
            comparison_status = "failed"
            comparison_notes.append(
                "Reference rows with zero baseline diverge from the dataset in blocking numeric columns: "
                + ", ".join(
                    f"{column}={count}"
                    for column, count in zero_reference_mismatches.items()
                    if column in _PRICE_BLOCKING_COLUMNS
                )
                + "."
            )
        if split_mismatch_count > 0:
            comparison_status = "failed"
            comparison_notes.append(f"Split mismatch count inside the validated overlap: {split_mismatch_count}.")

        advisory_notes = self._build_price_advisory_notes(advisory_differences)
        if advisory_notes:
            partial_validation_kinds.append(_PARTIAL_KIND_METHODOLOGY_CAVEAT)
            if comparison_status == "passed":
                comparison_notes.extend(advisory_notes)

        if advisory_zero_reference_count > 0:
            partial_validation_kinds.append(_PARTIAL_KIND_ADVISORY_ZERO_BASELINE)
            if comparison_status == "passed":
                comparison_notes.append(
                    "Advisory numeric columns diverged against zero reference baselines: "
                    + ", ".join(
                        f"{column}={count}"
                        for column, count in zero_reference_mismatches.items()
                        if column in _PRICE_ADVISORY_COLUMNS
                    )
                    + "."
                )

        if comparison_status == "not_validated" and coverage["provider_limited"]:
            comparison_notes.append(
                "The external source exposes only a partial recent window for this request, and no usable overlap was available."
            )

        partial_validation_kinds = _unique_non_empty(partial_validation_kinds)
        comparison_reason = self._join_reason_parts(*comparison_notes)
        status = self._compose_status(
            comparison_status=comparison_status,
            coverage_status=coverage_status,
            partial_validation_kinds=partial_validation_kinds,
        )

        return {
            "adapter": adapter_name,
            "status": status,
            "coverage_status": coverage_status,
            "comparison_status": comparison_status,
            "coverage_reason": coverage_reason,
            "comparison_reason": comparison_reason,
            "partial_validation_kinds": partial_validation_kinds,
            "reason": self._join_reason_parts(coverage_reason, comparison_reason),
            "score": None if comparison_status == "not_validated" else round(score, 3),
            "scope": "price",
            "overlap_rows": overlap,
            "gap_count": gap_count,
            "gap_ratio": round(gap_ratio, 6),
            "max_relative_diff": round(max_relative_diff, 8),
            "max_advisory_relative_diff": round(max_advisory_relative_diff, 8),
            "blocking_relative_differences": {key: round(value, 8) for key, value in blocking_differences.items()},
            "advisory_relative_differences": {key: round(value, 8) for key, value in advisory_differences.items()},
            "advisory_notes": advisory_notes,
            "zero_reference_mismatch_count": blocking_zero_reference_count + advisory_zero_reference_count,
            "zero_reference_mismatches": zero_reference_mismatches,
            "split_mismatch_count": split_mismatch_count,
            "sample_dates": (
                merged["date"].head(REFERENCE_SAMPLE_POINTS).dt.strftime("%Y-%m-%d").tolist()
                if not merged.empty
                else reference["date"].head(REFERENCE_SAMPLE_POINTS).dt.strftime("%Y-%m-%d").tolist()
            ),
            "coverage": {key: value for key, value in coverage.items() if key != "dataset_window"},
        }

    @staticmethod
    def _build_event_adapter_report(
        *,
        adapter_name: str,
        dataset_events: pd.DataFrame,
        reference_events: pd.DataFrame,
        source_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metadata = dict(source_metadata or {})
        coverage_status: CoverageStatus = "partial" if bool(metadata.get("partial_coverage")) else "full"
        coverage_notes = metadata.get("coverage_notes")
        coverage_reason = None
        if coverage_status == "partial":
            if isinstance(coverage_notes, list):
                coverage_reason = " | ".join(str(item).strip() for item in coverage_notes if str(item).strip()) or (
                    "Corporate action coverage is partial for this adapter."
                )
            elif coverage_notes:
                coverage_reason = str(coverage_notes).strip()
            else:
                coverage_reason = "Corporate action coverage is partial for this adapter."

        dataset_lookup = (
            pd.DataFrame(columns=["date", "dividends", "stock_splits"]).set_index("date")
            if dataset_events.empty
            else dataset_events.copy().set_index("date")
        )
        mismatches: list[str] = []
        mismatch_records: list[dict[str, object]] = []
        matched_events = 0
        checked_events = 0
        checked_columns: set[str] = set()
        date_aligned_events = 0

        for row in reference_events.itertuples(index=False):
            date_value = pd.Timestamp(row.date)
            dataset_row = dataset_lookup.loc[date_value] if date_value in dataset_lookup.index else None
            if dataset_row is not None:
                date_aligned_events += 1
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
                        dataset_value = float(
                            pd.to_numeric(pd.Series([dataset_row[column]]), errors="coerce").fillna(0.0).iloc[0]
                        )
                if abs(dataset_value - reference_value) <= 1e-8:
                    matched_events += 1
                else:
                    mismatch_records.append(
                        {
                            "column": column,
                            "date": date_value.date().isoformat(),
                            "dataset_value": dataset_value,
                            "reference_value": reference_value,
                        }
                    )
                    mismatches.append(
                        f"{column} mismatch at {date_value.date().isoformat()}: dataset={dataset_value} reference={reference_value}"
                    )

        partial_validation_kinds: list[str] = []
        if coverage_status == "partial":
            partial_validation_kinds.append(_PARTIAL_KIND_COVERAGE_LIMITED)

        if checked_events == 0:
            comparison_status: ComparisonStatus = "not_validated"
            comparison_reason = "The reference events did not expose any non-zero corporate action values to compare."
            score: float | None = None
            scale_mismatch = None
        else:
            score = max(0.0, 100.0 * (matched_events / checked_events))
            scale_mismatch = ExternalValidationService._detect_systematic_dividend_scale_mismatch(
                mismatch_records,
                matched_events=matched_events,
                date_aligned_events=date_aligned_events,
            )
            comparison_status = "passed" if not mismatches else "failed"
            comparison_reason = None if not mismatches else " | ".join(mismatches)
            if scale_mismatch is not None:
                comparison_status = "passed"
                score = 100.0
                partial_validation_kinds.append(_PARTIAL_KIND_METHODOLOGY_CAVEAT)
                comparison_reason = (
                    "Dividend ex-dates align, but amounts differ by a stable scale factor "
                    f"(~{scale_mismatch['dominant_ratio']:.4f}x); provider adjustment methodology likely differs, so event validation is partial."
                )

        partial_validation_kinds = _unique_non_empty(partial_validation_kinds)
        status = ExternalValidationService._compose_status(
            comparison_status=comparison_status,
            coverage_status=coverage_status,
            partial_validation_kinds=partial_validation_kinds,
        )

        return {
            "adapter": adapter_name,
            "status": status,
            "coverage_status": coverage_status,
            "comparison_status": comparison_status,
            "coverage_reason": coverage_reason,
            "comparison_reason": comparison_reason,
            "partial_validation_kinds": partial_validation_kinds,
            "reason": ExternalValidationService._join_reason_parts(coverage_reason, comparison_reason),
            "score": None if score is None else round(score, 3),
            "scope": "event",
            "checked_event_count": checked_events,
            "matched_event_count": matched_events,
            "date_aligned_event_count": date_aligned_events,
            "mismatch_count": len(mismatches),
            "checked_columns": sorted(checked_columns),
            "sample_dates": reference_events["date"].head(REFERENCE_SAMPLE_POINTS).dt.strftime("%Y-%m-%d").tolist(),
            "systematic_dividend_scale_factor": None if scale_mismatch is None else round(float(scale_mismatch["dominant_ratio"]), 6),
        }

    @staticmethod
    def render_text(report: dict[str, object], symbol: str) -> str:
        lines = [
            f"External validation report for {symbol}",
            f"status: {report.get('status')}",
            f"coverage_status: {report.get('coverage_status')}",
            f"comparison_status: {report.get('comparison_status')}",
            f"score: {report.get('score')}",
            f"reason: {report.get('reason')}",
            "",
        ]
        for adapter_report in report.get("adapter_reports", []):
            lines.append(f"- adapter: {adapter_report.get('adapter')}")
            lines.append(f"  status: {adapter_report.get('status')}")
            lines.append(f"  coverage_status: {adapter_report.get('coverage_status')}")
            lines.append(f"  comparison_status: {adapter_report.get('comparison_status')}")
            lines.append(f"  score: {adapter_report.get('score')}")
            lines.append(f"  reason: {adapter_report.get('reason')}")
            if adapter_report.get("coverage_reason"):
                lines.append(f"  coverage_reason: {adapter_report.get('coverage_reason')}")
            if adapter_report.get("comparison_reason"):
                lines.append(f"  comparison_reason: {adapter_report.get('comparison_reason')}")
        return "\n".join(lines).strip() + "\n"
