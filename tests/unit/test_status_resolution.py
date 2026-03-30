from __future__ import annotations

from dataset_core.status_resolution import resolve_ticker_status


def test_not_validated_external_status_is_neutral():
    result = resolve_ticker_status(
        warnings=[],
        internal_validation_status="skipped",
        external_validation_status="not_validated",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "success"
    assert result.reasons == []
    assert "External validation did not run." in result.neutral_notes


def test_failed_external_status_degrades_to_warning():
    result = resolve_ticker_status(
        warnings=[],
        internal_validation_status="passed",
        external_validation_status="failed",
        qlib_requested=False,
        qlib_compatible=False,
        qlib_errors=[],
    )

    assert result.status == "warning"
    assert result.reasons == ["External validation reported blocking differences."]


def test_internal_unsupported_and_passed_with_warnings_are_neutral():
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

    assert result.status == "success"
    assert result.reasons == []
    assert "Internal DQ safe mode is guaranteed only for interval=1d." in result.neutral_notes


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

    assert result.status == "warning"
    assert result.reasons == ["Adapter error: boom"]
    assert "Internal validation passed with non-blocking warnings." in result.neutral_notes
