from __future__ import annotations

import export_ohlcv_csv
import pytest
from app import streamlit_app


def test_cli_manual_api_key_has_priority_over_env():
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

    assert request.external_validation.eodhd.api_key == "manual-secret"


def test_cli_uses_env_api_key_when_eodhd_provider_is_selected(monkeypatch):
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
            "--external-validation-provider",
            "eodhd",
        ]
    )

    request = export_ohlcv_csv.build_request_from_args(args)

    assert request.external_validation.eodhd.api_key == "env-secret"


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


def test_cli_reports_a_clear_error_when_eodhd_provider_has_no_available_api_key(monkeypatch):
    monkeypatch.setattr(
        export_ohlcv_csv,
        "resolve_eodhd_api_key",
        lambda manual_value, allow_env_fallback=True: None,
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

    with pytest.raises(export_ohlcv_csv.RequestContractError, match="EODHD external validation requires an API key"):
        export_ohlcv_csv.build_request_from_args(args)


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
