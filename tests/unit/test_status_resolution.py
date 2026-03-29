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
    assert result.reasons == ["External validation status is failed."]
