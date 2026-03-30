from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

from providers.yfinance_provider import FetchResult, YFinanceProvider

from dataset_core.contracts import DatasetRequest


@dataclass
class ProviderSession:
    provider: YFinanceProvider
    cache_dir: Path
    bundle_cache: Dict[Tuple[object, ...], Dict[str, FetchResult]] = field(default_factory=dict)
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "provider_instances": 1,
            "fetch_calls": 0,
            "fetch_many_calls": 0,
            "bundle_cache_hits": 0,
            "bundle_cache_misses": 0,
        }
    )


class AcquisitionService:
    def __init__(self, provider_factory: type[YFinanceProvider] = YFinanceProvider) -> None:
        self.provider_factory = provider_factory
        self.last_session: ProviderSession | None = None

    @staticmethod
    def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in symbols:
            symbol = str(item or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        return normalized

    @staticmethod
    def _bundle_cache_key(
        symbols: list[str],
        request: DatasetRequest,
        auto_adjust: bool,
        actions: bool,
    ) -> tuple[object, ...]:
        return (
            tuple(symbols),
            request.interval,
            request.time_range.start_iso,
            request.time_range.end_iso,
            bool(auto_adjust),
            bool(actions),
        )

    @staticmethod
    def _is_failure_result(result: FetchResult) -> bool:
        metadata = getattr(result, "metadata", None)
        warnings = list(getattr(metadata, "warnings", []) or [])
        attempts = list(getattr(metadata, "attempts", []) or [])
        backend_used = getattr(metadata, "backend_used", None)
        is_empty = getattr(result, "data", None) is None or result.data.empty
        attempted_and_failed = bool(attempts) and all(not getattr(item, "success", False) for item in attempts)
        return bool(
            is_empty
            and backend_used is None
            and attempted_and_failed
            and any("Symbol failed during batch retrieval:" in str(warning) for warning in warnings)
        )

    def _provider_kwargs(self, request: DatasetRequest) -> dict[str, object]:
        provider_kwargs = request.provider.to_kwargs()
        provider_kwargs.setdefault("cache_dir", request.output_dir / "cache" / "yfinance")
        return provider_kwargs

    def create_session(self, request: DatasetRequest) -> ProviderSession:
        provider_kwargs = self._provider_kwargs(request)
        provider = self.provider_factory(**provider_kwargs)
        session = ProviderSession(
            provider=provider,
            cache_dir=Path(provider_kwargs["cache_dir"]).expanduser().resolve(),
        )
        self.last_session = session
        return session

    def fetch_many(
        self,
        symbols: Iterable[str],
        request: DatasetRequest,
        auto_adjust: bool,
        actions: bool,
        *,
        session: ProviderSession | None = None,
    ) -> dict[str, FetchResult]:
        normalized_symbols = self._normalize_symbols(symbols)
        if not normalized_symbols:
            return {}

        active_session = session or self.create_session(request)
        cache_key = self._bundle_cache_key(normalized_symbols, request, auto_adjust, actions)
        cached_bundle = active_session.bundle_cache.get(cache_key)
        if cached_bundle is not None:
            active_session.metrics["bundle_cache_hits"] += 1
            return {symbol: cached_bundle[symbol] for symbol in normalized_symbols}

        active_session.metrics["bundle_cache_misses"] += 1
        active_session.metrics["fetch_many_calls"] += 1
        result = active_session.provider.get_history_bundle(
            symbols=normalized_symbols,
            start=request.time_range.start,
            end=request.time_range.end,
            interval=request.interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        if isinstance(result, FetchResult):
            bundle = {normalized_symbols[0]: result}
        else:
            bundle = {
                symbol: result[symbol]
                for symbol in normalized_symbols
                if symbol in result and not self._is_failure_result(result[symbol])
            }

        if bundle:
            active_session.bundle_cache[cache_key] = dict(bundle)
        return bundle

    def fetch(
        self,
        symbol: str,
        request: DatasetRequest,
        auto_adjust: bool,
        actions: bool,
        *,
        session: ProviderSession | None = None,
    ) -> FetchResult:
        active_session = session or self.create_session(request)
        active_session.metrics["fetch_calls"] += 1
        result = self.fetch_many(
            [symbol],
            request,
            auto_adjust,
            actions,
            session=active_session,
        )
        normalized_symbol = str(symbol or "").strip().upper()
        if normalized_symbol not in result:
            raise RuntimeError("Single ticker export expected a FetchResult instance.")
        return result[normalized_symbol]
