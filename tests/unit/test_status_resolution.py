from __future__ import annotations

from dataset_core.status_resolution import resolve_ticker_status


def test_not_validated_external_status_degrades_to_warning():
    result = resolve_ticker_status(
        warnings=[],
        internal_validation_status="skipped",
        external_validation_status="not_validated",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "warning"
    assert result.validation_outcome == "success_partial_validation"
    assert "External validation did not validate the dataset." in result.reasons


def test_failed_external_status_degrades_to_warning():
    result = resolve_ticker_status(
        warnings=[],
        internal_validation_status="passed",
        external_validation_status="failed",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "error"
    assert result.validation_outcome == "failure"
    assert result.reasons == ["External validation reported blocking differences."]


def test_internal_unsupported_degrades_to_warning():
    result = resolve_ticker_status(
        warnings=[],
        neutral_notes=[],
        internal_validation_status="unsupported",
        internal_validation_reason="Internal DQ safe mode is guaranteed only for interval=1d.",
        external_validation_status="not_validated",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "warning"
    assert result.validation_outcome == "success_partial_validation"
    assert "Internal DQ safe mode is guaranteed only for interval=1d." in result.reasons


def test_adapter_error_external_status_is_material():
    result = resolve_ticker_status(
        warnings=[],
        internal_validation_status="passed_with_warnings",
        external_validation_status="adapter_error",
        external_validation_reason="Adapter error: boom",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "error"
    assert result.validation_outcome == "failure"
    assert "Adapter error: boom" in result.reasons
    assert "Internal validation passed with warnings." in result.reasons


def test_passed_partial_external_status_describes_provider_covered_overlap():
    result = resolve_ticker_status(
        warnings=[],
        internal_validation_status="passed",
        external_validation_status="passed_partial",
        external_validation_reason="External validation passed on the provider-covered or otherwise validated overlap only.",
        external_validation_coverage_status="partial",
        external_validation_comparison_status="passed",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "warning"
    assert result.validation_outcome == "success_partial_validation"
    assert any("provider-covered overlap only" in reason.lower() for reason in result.reasons)
