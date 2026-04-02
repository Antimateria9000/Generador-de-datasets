from __future__ import annotations

from pathlib import Path

import pytest

from dataset_core.contracts import EODHDExternalValidationConfig, ExternalValidationConfig, RequestContractError
from dataset_core.external_sources.factory import build_external_validation_service
from dataset_core.settings import resolve_eodhd_api_key, reset_local_env_cache


def test_external_validation_config_resolves_legacy_csv_provider(tmp_path):
    config = ExternalValidationConfig(
        reference_dir=tmp_path / "references",
        manual_events_file=tmp_path / "manual_events.csv",
    )

    assert config.is_enabled() is True
    assert config.resolved_provider() == "csv"


def test_external_validation_factory_builds_legacy_csv_service(tmp_path):
    config = ExternalValidationConfig(
        provider="csv",
        enabled=True,
        reference_dir=tmp_path / "references",
        manual_events_file=tmp_path / "manual_events.csv",
    )

    service = build_external_validation_service(config, output_root=tmp_path)

    assert [adapter.name() for adapter in service.price_adapters] == ["csv_reference"]
    assert [adapter.name() for adapter in service.event_adapters] == ["manual_events"]


def test_external_validation_factory_builds_eodhd_service_with_workspace_cache(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.metrics = {"request_count": 0, "cache_hits": 0, "cache_misses": 0}

    monkeypatch.setattr("dataset_core.external_sources.factory.EODHDClient", _StubClient)
    config = ExternalValidationConfig(
        enabled=True,
        provider="eodhd",
        eodhd=EODHDExternalValidationConfig(api_key="secret", use_cache=True),
    )

    service = build_external_validation_service(config, output_root=tmp_path)

    assert [adapter.name() for adapter in service.price_adapters] == ["eodhd_prices"]
    assert [adapter.name() for adapter in service.event_adapters] == ["eodhd_corporate_actions"]
    assert Path(captured["cache_dir"]) == tmp_path / "cache" / "external_validation" / "eodhd"
    assert service.price_adapters[0].price_lookback_days == 365


def test_external_validation_factory_accepts_api_key_resolved_from_env(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.metrics = {"request_count": 0, "cache_hits": 0, "cache_misses": 0}

    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_KEY=env-secret\n", encoding="utf-8")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    monkeypatch.setattr("dataset_core.external_sources.factory.EODHDClient", _StubClient)
    reset_local_env_cache()

    config = ExternalValidationConfig(
        enabled=True,
        provider="eodhd",
        eodhd=EODHDExternalValidationConfig(
            api_key=resolve_eodhd_api_key(project_root=tmp_path),
            use_cache=False,
        ),
    )

    service = build_external_validation_service(config, output_root=tmp_path)

    assert [adapter.name() for adapter in service.price_adapters] == ["eodhd_prices"]
    assert captured["api_key"] == "env-secret"


def test_external_validation_factory_never_receives_enabled_config_without_resolved_provider():
    with pytest.raises(RequestContractError, match="resolvable provider"):
        ExternalValidationConfig(enabled=True)
