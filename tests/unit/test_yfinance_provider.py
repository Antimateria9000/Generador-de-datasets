from __future__ import annotations

from dataset_core.acquisition import AcquisitionService
from dataset_core.contracts import DatasetRequest, TemporalRange
from tests.fixtures.sample_data import make_fetch_result, make_provider_frame


def test_acquisition_service_passes_effective_workspace_cache_dir(tmp_path):
    captured_kwargs = {}

    class _Provider:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def get_history_bundle(self, **kwargs):
            return make_fetch_result("MSFT", make_provider_frame("MSFT"))

    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
    )

    AcquisitionService(provider_factory=_Provider).fetch(
        symbol="MSFT",
        request=request,
        auto_adjust=False,
        actions=True,
    )

    assert captured_kwargs["cache_dir"] == (tmp_path / "cache" / "yfinance").resolve()
