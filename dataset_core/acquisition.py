from __future__ import annotations

from providers.yfinance_provider import FetchResult, YFinanceProvider

from dataset_core.contracts import DatasetRequest


class AcquisitionService:
    def __init__(self, provider_factory: type[YFinanceProvider] = YFinanceProvider) -> None:
        self.provider_factory = provider_factory

    def fetch(
        self,
        symbol: str,
        request: DatasetRequest,
        auto_adjust: bool,
        actions: bool,
    ) -> FetchResult:
        provider = self.provider_factory(**request.provider.to_kwargs())
        result = provider.get_history_bundle(
            symbols=symbol,
            start=request.time_range.start,
            end=request.time_range.end,
            interval=request.interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        if not isinstance(result, FetchResult):
            raise RuntimeError("Single ticker export expected a FetchResult instance.")
        return result
