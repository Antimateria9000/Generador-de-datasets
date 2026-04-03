from __future__ import annotations

import pytest

from dataset_core.contracts import RequestContractError
from dataset_core.external_validation_runtime import build_external_validation_config


def test_external_validation_runtime_builder_preserves_explicit_zero_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dataset_core.external_validation_runtime.is_external_validation_runtime_enabled",
        lambda: True,
    )

    config = build_external_validation_config(
        enabled=True,
        provider="eodhd",
        reference_dir=None,
        manual_events_file=None,
        eodhd_api_key="secret",
        eodhd_base_url=None,
        eodhd_timeout_seconds=1.0,
        eodhd_use_cache=True,
        eodhd_cache_dir=tmp_path / "eodhd-cache",
        eodhd_cache_ttl_seconds=0,
        eodhd_allow_partial_coverage=False,
        eodhd_max_retries=1,
        eodhd_backoff_seconds=0.0,
        eodhd_price_lookback_days=1,
    )

    assert config.eodhd.cache_ttl_seconds == 0
    assert config.eodhd.backoff_seconds == 0.0


@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        ("eodhd_timeout_seconds", 0.0),
        ("eodhd_max_retries", 0),
        ("eodhd_price_lookback_days", 0),
    ],
)
def test_external_validation_runtime_builder_does_not_reinterpret_invalid_zero_as_default(
    monkeypatch,
    field_name: str,
    override: float | int,
):
    monkeypatch.setattr(
        "dataset_core.external_validation_runtime.is_external_validation_runtime_enabled",
        lambda: True,
    )
    kwargs = {
        "enabled": True,
        "provider": "eodhd",
        "reference_dir": None,
        "manual_events_file": None,
        "eodhd_api_key": "secret",
        "eodhd_base_url": None,
        "eodhd_timeout_seconds": 1.0,
        "eodhd_use_cache": True,
        "eodhd_cache_dir": None,
        "eodhd_cache_ttl_seconds": 60,
        "eodhd_allow_partial_coverage": False,
        "eodhd_max_retries": 1,
        "eodhd_backoff_seconds": 0.5,
        "eodhd_price_lookback_days": 1,
    }
    kwargs[field_name] = override

    with pytest.raises(RequestContractError):
        build_external_validation_config(**kwargs)
