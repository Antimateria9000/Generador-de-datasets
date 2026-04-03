from __future__ import annotations

from pathlib import Path

from dataset_core.contracts import EODHDExternalValidationConfig, ExternalValidationConfig
from dataset_core.settings import (
    DEFAULT_EODHD_BACKOFF_SECONDS,
    DEFAULT_EODHD_BASE_URL,
    DEFAULT_EODHD_CACHE_TTL_SECONDS,
    DEFAULT_EODHD_MAX_RETRIES,
    DEFAULT_EODHD_PRICE_LOOKBACK_DAYS,
    DEFAULT_EODHD_TIMEOUT_SECONDS,
    is_external_validation_runtime_enabled,
)


def _coalesce_if_none(value, default):
    return default if value is None else value


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _normalize_provider_name(provider: str | None) -> str | None:
    text = str(provider or "").strip()
    return None if not text else text


def build_external_validation_config(
    *,
    enabled: bool | None,
    provider: str | None,
    reference_dir: str | Path | None,
    manual_events_file: str | Path | None,
    eodhd_api_key: str | None,
    eodhd_base_url: str | None,
    eodhd_timeout_seconds: float | None,
    eodhd_use_cache: bool,
    eodhd_cache_dir: str | Path | None,
    eodhd_cache_ttl_seconds: int | None,
    eodhd_allow_partial_coverage: bool,
    eodhd_max_retries: int | None,
    eodhd_backoff_seconds: float | None,
    eodhd_price_lookback_days: int | None,
) -> ExternalValidationConfig:
    if not is_external_validation_runtime_enabled():
        return ExternalValidationConfig()

    return ExternalValidationConfig(
        enabled=enabled,
        provider=_normalize_provider_name(provider),
        reference_dir=_resolve_optional_path(reference_dir),
        manual_events_file=_resolve_optional_path(manual_events_file),
        eodhd=EODHDExternalValidationConfig(
            api_key=eodhd_api_key,
            base_url=str(_coalesce_if_none(eodhd_base_url, DEFAULT_EODHD_BASE_URL)).strip(),
            timeout_seconds=float(_coalesce_if_none(eodhd_timeout_seconds, DEFAULT_EODHD_TIMEOUT_SECONDS)),
            use_cache=bool(eodhd_use_cache),
            cache_dir=_resolve_optional_path(eodhd_cache_dir),
            cache_ttl_seconds=int(_coalesce_if_none(eodhd_cache_ttl_seconds, DEFAULT_EODHD_CACHE_TTL_SECONDS)),
            allow_partial_coverage=bool(eodhd_allow_partial_coverage),
            max_retries=int(_coalesce_if_none(eodhd_max_retries, DEFAULT_EODHD_MAX_RETRIES)),
            backoff_seconds=float(_coalesce_if_none(eodhd_backoff_seconds, DEFAULT_EODHD_BACKOFF_SECONDS)),
            price_lookback_days=int(_coalesce_if_none(eodhd_price_lookback_days, DEFAULT_EODHD_PRICE_LOOKBACK_DAYS)),
        ),
    )
