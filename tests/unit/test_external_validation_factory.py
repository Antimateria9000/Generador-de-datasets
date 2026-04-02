from __future__ import annotations

from pathlib import Path

import pytest

from dataset_core.contracts import EODHDExternalValidationConfig, ExternalValidationConfig
from dataset_core.external_sources.factory import build_external_validation_service
from dataset_core.settings import EXTERNAL_VALIDATION_DISABLED_REASON, resolve_eodhd_api_key, reset_local_env_cache
from dataset_core.validation_external import DisabledExternalValidationService


def test_external_validation_config_resolves_legacy_csv_provider(tmp_path):
    config = ExternalValidationConfig(
        reference_dir=tmp_path / "references",
        manual_events_file=tmp_path / "manual_events.csv",
    )

    assert config.is_enabled() is False
    assert config.resolved_provider() == "csv"


def test_external_validation_factory_returns_disabled_service_for_legacy_csv_config(tmp_path):
    config = ExternalValidationConfig(
        provider="csv",
        enabled=True,
        reference_dir=tmp_path / "references",
        manual_events_file=tmp_path / "manual_events.csv",
    )

    service = build_external_validation_service(config, output_root=tmp_path)

    assert isinstance(service, DisabledExternalValidationService)
    report = service.validate(frame=None, symbol="MSFT", start=None, end=None).to_dict()
    assert report["enabled"] is False
    assert report["status"] == "disabled"
    assert report["reason"] == EXTERNAL_VALIDATION_DISABLED_REASON


def test_external_validation_factory_never_instantiates_eodhd_client_while_module_is_disabled(
    tmp_path, monkeypatch
):
    def _raise_if_called(**_kwargs):
        raise AssertionError("EODHDClient should not be instantiated while external validation is disabled.")

    monkeypatch.setattr("dataset_core.external_sources.factory.EODHDClient", _raise_if_called)
    config = ExternalValidationConfig(
        enabled=True,
        provider="eodhd",
        eodhd=EODHDExternalValidationConfig(api_key="secret", use_cache=True),
    )

    service = build_external_validation_service(config, output_root=tmp_path)

    assert isinstance(service, DisabledExternalValidationService)
    report = service.validate(frame=None, symbol="MSFT", start=None, end=None).to_dict()
    assert report["status"] == "disabled"


def test_external_validation_factory_ignores_env_resolved_eodhd_credentials_while_disabled(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_KEY=env-secret\n", encoding="utf-8")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    monkeypatch.setattr(
        "dataset_core.external_sources.factory.EODHDClient",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("EODHDClient should not be constructed while external validation is disabled.")
        ),
    )
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

    assert isinstance(service, DisabledExternalValidationService)
    assert service.validate(frame=None, symbol="MSFT", start=None, end=None).to_dict()["status"] == "disabled"


def test_external_validation_factory_never_receives_enabled_config_without_resolved_provider():
    config = ExternalValidationConfig(enabled=True)

    assert config.is_enabled() is False
    assert config.to_dict()["status"] == "disabled"
