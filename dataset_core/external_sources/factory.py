from __future__ import annotations

from pathlib import Path

from dataset_core.contracts import ExternalValidationConfig
from dataset_core.external_sources.csv_source import CSVReferenceSource
from dataset_core.external_sources.eodhd import (
    EODHDClient,
    EODHDCorporateActionsReferenceSource,
    EODHDPriceReferenceSource,
    EODHDSymbolResolver,
)
from dataset_core.external_sources.manual_events import ManualEventReferenceSource
from dataset_core.settings import resolve_workspace_tree
from dataset_core.validation_external import ExternalValidationService


def _default_eodhd_cache_dir(output_root: Path | None) -> Path:
    workspace = resolve_workspace_tree(output_root)
    return workspace["cache"] / "external_validation" / "eodhd"


def build_external_validation_service(
    config: ExternalValidationConfig,
    *,
    output_root: Path | None = None,
) -> ExternalValidationService:
    if not config.is_enabled():
        return ExternalValidationService()

    provider = config.resolved_provider()
    price_adapters = []
    event_adapters = []

    if provider == "csv":
        if config.reference_dir is not None:
            price_adapters.append(CSVReferenceSource(config.reference_dir))
        if config.manual_events_file is not None:
            event_adapters.append(ManualEventReferenceSource(config.manual_events_file))
        return ExternalValidationService(price_adapters=price_adapters, event_adapters=event_adapters)

    if provider == "eodhd":
        if not config.eodhd.api_key:
            raise ValueError("external_validation.provider='eodhd' requires external_validation.eodhd.api_key.")
        cache_dir = None
        if config.eodhd.use_cache:
            cache_dir = config.eodhd.cache_dir or _default_eodhd_cache_dir(output_root)
        client = EODHDClient(
            api_key=config.eodhd.api_key,
            base_url=config.eodhd.base_url,
            timeout_seconds=config.eodhd.timeout_seconds,
            use_cache=config.eodhd.use_cache,
            cache_dir=cache_dir,
            cache_ttl_seconds=config.eodhd.cache_ttl_seconds,
            max_retries=config.eodhd.max_retries,
            backoff_seconds=config.eodhd.backoff_seconds,
        )
        symbol_resolver = EODHDSymbolResolver(
            exchange_hint=config.eodhd.exchange_hint,
            symbol_map_file=config.eodhd.symbol_map_file,
        )
        return ExternalValidationService(
            price_adapters=[
                EODHDPriceReferenceSource(
                    client,
                    symbol_resolver=symbol_resolver,
                    price_lookback_days=config.eodhd.price_lookback_days,
                )
            ],
            event_adapters=[
                EODHDCorporateActionsReferenceSource(
                    client,
                    allow_partial_coverage=config.eodhd.allow_partial_coverage,
                    symbol_resolver=symbol_resolver,
                )
            ],
        )

    raise ValueError(f"Unsupported external validation provider: {provider!r}")
