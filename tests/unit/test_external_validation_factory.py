from __future__ import annotations

from pathlib import Path

from dataset_core.contracts import EODHDExternalValidationConfig, ExternalValidationConfig
from dataset_core.external_sources.factory import build_external_validation_service


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

