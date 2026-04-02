from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd

from dataset_core.external_sources import (
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

_PRICE_BLOCKING_COLUMNS = ("open", "high", "low", "close")
_PRICE_ADVISORY_COLUMNS = ("adj_close", "volume")
_PARTIAL_PRICE_MIN_OVERLAP_ROWS = 60
_PARTIAL_PRICE_END_TOLERANCE_DAYS = 7
_SYSTEMATIC_DIVIDEND_SCALE_MIN_RECORDS = 2
_SYSTEMATIC_DIVIDEND_SCALE_STABILITY_STD = 1e-4
_SYSTEMATIC_DIVIDEND_SCALE_IDENTITY_TOL = 0.01


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
        if dataset.empty:
            return ExternalValidationResult(
                {
                    "status": "not_validated",
                    "score": None,
                    "reason": "Dataset is empty; external validation was skipped.",
                    "adapter_reports": [],
                }
            )

        adapter_reports: list[dict[str, object]] = []
        validated_scores: list[float] = []
        has_failure = False
        has_adapter_error = False
        has_validation_error = False
        has_partial_validation = False

        for adapter in self.price_adapters:
            report, score_bucket = self._validate_price_adapter(adapter, dataset, symbol, start, end)
            adapter_reports.append(report)
            status = str(report.get("status") or "")
            if status == "failed":
                has_failure = True
            elif status == "adapter_error":
                has_adapter_error = True
            elif status == "validation_error":
                has_validation_error = True
            elif status == "passed":
                validated_scores.append(score_bucket)
            elif status == "passed_partial":
                has_partial_validation = True
                validated_scores.append(score_bucket)
            else:
                has_partial_validation = True

        for adapter in self.event_adapters:
            report, score_bucket = self._validate_event_adapter(adapter, dataset_events, symbol, start, end)
            adapter_reports.append(report)
            status = str(report.get("status") or "")
            if status == "failed":
                has_failure = True
            elif status == "adapter_error":
                has_adapter_error = True
            elif status == "validation_error":
                has_validation_error = True
            elif status == "passed":
                validated_scores.append(score_bucket)
            elif status == "passed_partial":
                has_partial_validation = True
                validated_scores.append(score_bucket)
            else:
                has_partial_validation = True

        overall_status = "not_validated"
        overall_reason = "No external reference was available."
        overall_score = None if not validated_scores else round(sum(validated_scores) / len(validated_scores), 3)
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
            overall_status = "passed_partial" if has_partial_validation else "passed"
            overall_reason = (
                "External validation completed with partial provider coverage or methodology caveats."
                if has_partial_validation
                else None
            )

        return ExternalValidationResult(
            {
                "status": overall_status,
                "score": overall_score,
                "reason": overall_reason,
                "adapter_reports": adapter_reports,
            }
        )

    def _validate_price_adapter(self, adapter, dataset: pd.DataFrame, symbol: str, start: str | None, end: str | None):
        adapter_name = adapter.name()
        try:
            raw_reference = adapter.fetch_reference(symbol, start, end)
            reference = normalize_reference_frame(raw_reference)
            source_metadata = extract_source_metadata(reference)
        except FileNotFoundError as exc:
            return {"adapter": adapter_name, "status": "not_validated", "reason": str(exc), "score": None, "scope": "price"}, 0.0
        except ExternalSourceNotFoundError as exc:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": str(exc),
                "score": None,
                "scope": "price",
                "error_kind": self._adapter_error_kind(exc),
            }, 0.0
        except ExternalSourceCoverageError as exc:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": str(exc),
                "score": None,
                "scope": "price",
                "error_kind": self._adapter_error_kind(exc),
            }, 0.0
        except (ExternalSourceAuthError, ExternalSourceNetworkError, ExternalSourceRateLimitError, ExternalSourcePayloadError) as exc:
            return {
                "adapter": adapter_name,
                "status": "adapter_error",
                "reason": f"Adapter error: {exc}",
                "score": None,
                "scope": "price",
                "error_kind": self._adapter_error_kind(exc),
            }, 0.0
        except Exception as exc:
            return {
                "adapter": adapter_name,
                "status": "adapter_error",
                "reason": f"Adapter error: {exc}",
                "score": None,
                "scope": "price",
            }, 0.0

        if reference.empty:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": "Reference adapter returned no rows for the requested range.",
                "score": None,
                "scope": "price",
                "source_metadata": source_metadata,
            }, 0.0

        try:
            report = self._build_adapter_report(
                adapter_name=adapter_name,
                dataset=dataset,
                reference=reference,
                source_metadata=source_metadata,
                requested_start=start,
                requested_end=end,
            )
        except Exception as exc:
            return {
                "adapter": adapter_name,
                "status": "validation_error",
                "reason": f"Validation error: {exc}",
                "score": None,
                "scope": "price",
            }, 0.0

        report["source_metadata"] = source_metadata
        return report, float(report["score"])

    def _validate_event_adapter(self, adapter, dataset_events: pd.DataFrame, symbol: str, start: str | None, end: str | None):
        adapter_name = adapter.name()
        try:
            raw_reference = adapter.fetch_events(symbol, start, end)
            reference = normalize_event_frame(raw_reference)
            source_metadata = extract_source_metadata(reference)
        except FileNotFoundError as exc:
            return {"adapter": adapter_name, "status": "not_validated", "reason": str(exc), "score": None, "scope": "event"}, 0.0
        except ExternalSourceNotFoundError as exc:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": str(exc),
                "score": None,
                "scope": "event",
                "error_kind": self._adapter_error_kind(exc),
            }, 0.0
        except ExternalSourceCoverageError as exc:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": str(exc),
                "score": None,
                "scope": "event",
                "error_kind": self._adapter_error_kind(exc),
            }, 0.0
        except (ExternalSourceAuthError, ExternalSourceNetworkError, ExternalSourceRateLimitError, ExternalSourcePayloadError) as exc:
            return {
                "adapter": adapter_name,
                "status": "adapter_error",
                "reason": f"Adapter error: {exc}",
                "score": None,
                "scope": "event",
                "error_kind": self._adapter_error_kind(exc),
            }, 0.0
        except Exception as exc:
            return {
                "adapter": adapter_name,
                "status": "adapter_error",
                "reason": f"Adapter error: {exc}",
                "score": None,
                "scope": "event",
            }, 0.0

        if reference.empty:
            return {
                "adapter": adapter_name,
                "status": "not_validated",
                "reason": "Event reference adapter returned no manual events for the requested range.",
                "score": None,
                "scope": "event",
                "source_metadata": source_metadata,
            }, 0.0

        try:
            report = self._build_event_adapter_report(
                adapter_name=adapter_name,
                dataset_events=dataset_events,
                reference_events=reference,
            )
        except Exception as exc:
            return {
                "adapter": adapter_name,
                "status": "validation_error",
                "reason": f"Validation error: {exc}",
                "score": None,
                "scope": "event",
            }, 0.0

        report["source_metadata"] = source_metadata
        return report, float(report["score"])

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
        partial_coverage = bool(metadata.get("partial_coverage"))
        if not partial_coverage and requested_start_ts is not None and effective_start is not None:
            partial_coverage = bool(effective_start > requested_start_ts)
        if not partial_coverage and requested_end_ts is not None and effective_end is not None:
            partial_coverage = bool(
                effective_end < (requested_end_ts - pd.Timedelta(days=_PARTIAL_PRICE_END_TOLERANCE_DAYS))
            )

        if partial_coverage:
            if effective_start is None and not reference.empty:
                effective_start = pd.Timestamp(reference["date"].min())
            if effective_end is None and not reference.empty:
                effective_end = pd.Timestamp(reference["date"].max())
            dataset_window = dataset
            if effective_start is not None:
                dataset_window = dataset_window[dataset_window["date"] >= effective_start]
            if effective_end is not None:
                dataset_window = dataset_window[dataset_window["date"] <= effective_end]
            dataset_window = dataset_window.reset_index(drop=True)
            uncovered_prefix_count = 0 if effective_start is None else int((dataset["date"] < effective_start).sum())
            uncovered_suffix_count = 0 if effective_end is None else int((dataset["date"] > effective_end).sum())
        else:
            dataset_window = dataset.reset_index(drop=True)
            uncovered_prefix_count = 0
            uncovered_suffix_count = 0

        partial_recent_suffix = (
            partial_coverage
            and uncovered_prefix_count > 0
            and uncovered_suffix_count <= _PARTIAL_PRICE_END_TOLERANCE_DAYS
            and len(dataset_window) >= _PARTIAL_PRICE_MIN_OVERLAP_ROWS
        )
        return {
            "partial_coverage": partial_coverage,
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

    def _build_adapter_report(
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
            100.0 - (gap_ratio * 60.0) - (max_relative_diff * 1000.0) - (split_mismatch_count * 10.0) - (blocking_zero_reference_count * 10.0),
        )

        status = "passed"
        reasons: list[str] = []
        if overlap == 0:
            status = "not_validated"
            reasons.append("No overlapping dates were available inside the provider-covered validation window.")
        if gap_ratio > 0.10:
            status = "failed"
            reasons.append(f"Calendar gap ratio too high: {gap_ratio:.2%}.")
        if max_relative_diff > REFERENCE_RELATIVE_TOLERANCE:
            status = "failed"
            reasons.append(f"Relative OHLC difference exceeds tolerance ({REFERENCE_RELATIVE_TOLERANCE:.4f}).")
        if blocking_zero_reference_count > 0:
            status = "failed"
            reasons.append(
                "Reference rows with zero baseline diverge from the dataset in numeric columns: "
                + ", ".join(f"{column}={count}" for column, count in zero_reference_mismatches.items())
                + "."
            )
        if split_mismatch_count > 0:
            status = "failed"
            reasons.append(f"Split mismatch count: {split_mismatch_count}.")
        if status == "passed" and coverage["partial_recent_suffix"]:
            status = "passed_partial"
            reasons.append("Reference coverage is limited to the recent window available from the external source; validated over that overlapping window only.")
        elif status == "not_validated" and coverage["partial_coverage"]:
            reasons.append("The external source exposes only a partial recent window for this request, and no usable overlap was available.")

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
            "max_advisory_relative_diff": round(max_advisory_relative_diff, 8),
            "blocking_relative_differences": {key: round(value, 8) for key, value in blocking_differences.items()},
            "advisory_relative_differences": {key: round(value, 8) for key, value in advisory_differences.items()},
            "advisory_notes": self._build_price_advisory_notes(advisory_differences),
            "zero_reference_mismatch_count": blocking_zero_reference_count,
            "zero_reference_mismatches": zero_reference_mismatches,
            "split_mismatch_count": split_mismatch_count,
            "sample_dates": merged["date"].head(REFERENCE_SAMPLE_POINTS).dt.strftime("%Y-%m-%d").tolist(),
            "coverage": {key: value for key, value in coverage.items() if key != "dataset_window"},
        }

    @staticmethod
    def _build_event_adapter_report(
        *,
        adapter_name: str,
        dataset_events: pd.DataFrame,
        reference_events: pd.DataFrame,
    ) -> dict[str, object]:
        if reference_events.empty:
            return {"adapter": adapter_name, "status": "not_validated", "reason": "No manual events were provided.", "score": None, "scope": "event"}

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
                        dataset_value = float(pd.to_numeric(pd.Series([dataset_row[column]]), errors="coerce").fillna(0.0).iloc[0])
                if abs(dataset_value - reference_value) <= 1e-8:
                    matched_events += 1
                else:
                    mismatch_records.append(
                        {"column": column, "date": date_value.date().isoformat(), "dataset_value": dataset_value, "reference_value": reference_value}
                    )
                    mismatches.append(f"{column} mismatch at {date_value.date().isoformat()}: dataset={dataset_value} reference={reference_value}")

        score = 100.0 if checked_events == 0 else max(0.0, 100.0 * (matched_events / checked_events))
        scale_mismatch = ExternalValidationService._detect_systematic_dividend_scale_mismatch(
            mismatch_records,
            matched_events=matched_events,
            date_aligned_events=date_aligned_events,
        )
        status = "passed" if not mismatches else "failed"
        reason = None if not mismatches else " | ".join(mismatches)
        if scale_mismatch is not None:
            status = "passed_partial"
            score = 100.0
            reason = (
                "Dividend ex-dates align, but amounts differ by a stable scale factor "
                f"(~{scale_mismatch['dominant_ratio']:.4f}x); provider adjustment methodology likely differs, so event validation is partial."
            )

        return {
            "adapter": adapter_name,
            "status": status,
            "reason": reason,
            "score": round(score, 3),
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
