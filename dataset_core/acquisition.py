from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Tuple
from uuid import uuid4

from dataset_core.settings import resolve_yfinance_cache_dir
from providers.yfinance_provider import FetchResult, FetchState, YFinanceProvider

from dataset_core.contracts import DatasetRequest


@dataclass
class ProviderSession:
    provider: YFinanceProvider
    cache_dir: Path | None
    cache_namespace: str | None = None
    bundle_cache: Dict[Tuple[object, ...], Dict[str, FetchResult]] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock, repr=False)
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
    def _classify_fetch_result(result: FetchResult) -> str:
        metadata = getattr(result, "metadata", None)
        fetch_state = getattr(metadata, "fetch_state", None)
        if isinstance(fetch_state, FetchState):
            return fetch_state.value
        normalized_state = str(fetch_state or "").strip().lower()
        if normalized_state in {item.value for item in FetchState}:
            return normalized_state

        attempts = list(getattr(metadata, "attempts", []) or [])
        backend_used = getattr(metadata, "backend_used", None)
        is_empty = getattr(result, "data", None) is None or result.data.empty
        attempted_and_failed = bool(attempts) and all(not getattr(item, "success", False) for item in attempts)
        if is_empty and backend_used in {None, "n/a"} and attempted_and_failed:
            return FetchState.FAILED.value
        if is_empty:
            return FetchState.EMPTY.value
        return FetchState.SUCCESS.value

    @classmethod
    def _is_failure_result(cls, result: FetchResult) -> bool:
        return cls._classify_fetch_result(result) == FetchState.FAILED.value

    def _provider_kwargs(
        self,
        request: DatasetRequest,
        *,
        cache_namespace: str | None = None,
    ) -> tuple[dict[str, object], str | None]:
        provider_kwargs = request.provider.to_runtime_kwargs()
        resolved_namespace = cache_namespace
        if request.provider.cache_mode == "run" and not resolved_namespace:
            resolved_namespace = f"session-{uuid4().hex[:12]}"
        provider_kwargs["cache_dir"] = resolve_yfinance_cache_dir(
            request.output_dir,
            request.provider.cache_dir,
            cache_mode=request.provider.cache_mode,
            cache_namespace=resolved_namespace,
        )
        provider_kwargs["cache_mode"] = request.provider.cache_mode
        return provider_kwargs, resolved_namespace

    def create_session(
        self,
        request: DatasetRequest,
        *,
        cache_namespace: str | None = None,
    ) -> ProviderSession:
        provider_kwargs, resolved_namespace = self._provider_kwargs(request, cache_namespace=cache_namespace)
        provider = self.provider_factory(**provider_kwargs)
        session = ProviderSession(
            provider=provider,
            cache_dir=None
            if provider_kwargs["cache_dir"] is None
            else Path(provider_kwargs["cache_dir"]).expanduser().resolve(),
            cache_namespace=resolved_namespace,
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
        with active_session.lock:
            cached_bundle = active_session.bundle_cache.get(cache_key)
        if cached_bundle is not None:
            with active_session.lock:
                active_session.metrics["bundle_cache_hits"] += 1
            return {symbol: cached_bundle[symbol] for symbol in normalized_symbols}

        with active_session.lock:
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
            with active_session.lock:
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
        with active_session.lock:
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
