from __future__ import annotations

from dataset_core.batch_orchestrator import BatchOrchestrator
from dataset_core.contracts import DatasetRequest, ExternalValidationConfig, TemporalRange
from dataset_core.export_service import DatasetExportService
from tests.fixtures.sample_data import DummyAcquisitionService, make_provider_frame


def test_external_validation_reports_pass_when_reference_matches(tmp_path, patch_market_context, reference_dir):
    frame = make_provider_frame("MSFT")
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
        external_validation=ExternalValidationConfig(reference_dir=reference_dir),
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]

    assert result.external_validation_status == "passed"
    assert result.artifacts.external_json.exists()
    assert result.artifacts.external_txt.exists()


def test_external_validation_reports_not_validated_without_reference(tmp_path, patch_market_context):
    frame = make_provider_frame("MSFT")
    export_service = DatasetExportService(acquisition_service=DummyAcquisitionService({"MSFT": frame}))
    orchestrator = BatchOrchestrator(export_service=export_service)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        dq_mode="off",
    )

    batch_result = orchestrator.run(request)
    result = batch_result.results[0]

    assert result.external_validation_status == "not_validated"
    assert result.status == "success"
    assert "External validation did not run." in result.neutral_notes
