from __future__ import annotations

import export_ohlcv_csv
from app import streamlit_app


def test_cli_build_request_does_not_resolve_manual_api_key_while_external_validation_is_disabled(monkeypatch):
    monkeypatch.setattr(
        export_ohlcv_csv,
        "resolve_eodhd_api_key",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resolve_eodhd_api_key should not be called while external validation is disabled.")
        ),
    )
    args = export_ohlcv_csv.build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--external-validation-provider",
            "eodhd",
            "--eodhd-api-key",
            "manual-secret",
        ]
    )

    request = export_ohlcv_csv.build_request_from_args(args)

    assert request.external_validation.is_enabled() is False
    assert request.external_validation.eodhd.api_key is None
    assert request.external_validation.to_dict()["status"] == "disabled"


def test_cli_build_request_does_not_use_env_api_key_when_module_is_disabled(monkeypatch):
    monkeypatch.setattr(
        export_ohlcv_csv,
        "resolve_eodhd_api_key",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resolve_eodhd_api_key should not be called while external validation is disabled.")
        ),
    )
    args = export_ohlcv_csv.build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--external-validation-provider",
            "eodhd",
        ]
    )

    request = export_ohlcv_csv.build_request_from_args(args)

    assert request.external_validation.is_enabled() is False
    assert request.external_validation.resolved_provider() is None
    assert request.external_validation.to_dict()["status"] == "disabled"


def test_cli_keeps_eodhd_env_fallback_disabled_when_provider_is_not_selected(monkeypatch):
    monkeypatch.setattr(
        export_ohlcv_csv,
        "resolve_eodhd_api_key",
        lambda manual_value, allow_env_fallback=True: "env-secret" if allow_env_fallback else None,
    )
    args = export_ohlcv_csv.build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
        ]
    )

    request = export_ohlcv_csv.build_request_from_args(args)

    assert request.external_validation.eodhd.api_key is None
    assert request.external_validation.resolved_provider() is None
    assert request.external_validation.to_dict()["status"] == "disabled"


def test_cli_no_longer_raises_missing_api_key_when_external_validation_module_is_disabled(monkeypatch):
    monkeypatch.setattr(
        export_ohlcv_csv,
        "resolve_eodhd_api_key",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resolve_eodhd_api_key should not be called while external validation is disabled.")
        ),
    )
    args = export_ohlcv_csv.build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--external-validation-provider",
            "eodhd",
        ]
    )

    request = export_ohlcv_csv.build_request_from_args(args)

    assert request.external_validation.is_enabled() is False
    assert request.external_validation.to_dict()["status"] == "disabled"


def test_streamlit_resolution_prefers_manual_then_env_then_none(monkeypatch):
    monkeypatch.setattr(
        streamlit_app,
        "resolve_eodhd_api_key",
        lambda manual_value, allow_env_fallback=True: manual_value or ("env-secret" if allow_env_fallback else None),
    )

    assert (
        streamlit_app.resolve_requested_eodhd_api_key(
            "manual-secret",
            external_validation_provider="eodhd",
        )
        == "manual-secret"
    )
    assert (
        streamlit_app.resolve_requested_eodhd_api_key(
            "",
            external_validation_provider="eodhd",
        )
        == "env-secret"
    )
    assert (
        streamlit_app.resolve_requested_eodhd_api_key(
            "",
            external_validation_provider="csv",
        )
        is None
    )


def test_cli_propagates_provider_cache_mode():
    args = export_ohlcv_csv.build_parser().parse_args(
        [
            "--ticker",
            "MSFT",
            "--years",
            "5",
            "--provider-cache-mode",
            "run",
        ]
    )

    request = export_ohlcv_csv.build_request_from_args(args)

    assert request.provider.cache_mode == "run"
