from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset_core.contracts import (
    DatasetRequest,
    EODHDExternalValidationConfig,
    ExternalValidationConfig,
    RequestContractError,
    TemporalRange,
    load_tickers_from_file,
    parse_extras,
    parse_tickers_text,
    resolve_ticker_inputs,
)


def test_parse_tickers_text_deduplicates_preserving_order():
    assert parse_tickers_text("MSFT, AAPL\nMSFT   NVDA") == ["MSFT", "AAPL", "NVDA"]


def test_load_tickers_from_file_supports_txt_fixture():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "universe.txt"
    assert load_tickers_from_file(fixture) == ["MSFT", "AAPL", "NVDA"]


def test_load_tickers_from_file_discards_null_like_values_in_csv(tmp_path):
    csv_path = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": ["MSFT", None, float("nan"), "  ", "nvda", "null"]}).to_csv(csv_path, index=False)

    assert load_tickers_from_file(csv_path) == ["MSFT", "NVDA"]


def test_resolve_ticker_inputs_rejects_ambiguity():
    with pytest.raises(RequestContractError):
        resolve_ticker_inputs(ticker="MSFT", tickers="AAPL")


def test_parse_extras_normalizes_and_validates():
    assert parse_extras("adj_close, FACTOR, dividends") == ["adj_close", "factor", "dividends"]
    with pytest.raises(RequestContractError):
        parse_extras("foo")


def test_dataset_request_rejects_batch_filename_override():
    with pytest.raises(RequestContractError):
        DatasetRequest(
            tickers=["MSFT", "AAPL"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            filename_override="custom.csv",
        )


def test_dataset_request_normalizes_filename_override_to_safe_basename():
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        filename_override="..\\..\\unsafe folder\\report?.csv",
    )

    assert request.filename_override == "report_.csv"


def test_dataset_request_rejects_forbidden_qlib_extras():
    with pytest.raises(RequestContractError, match="Preset qlib is closed"):
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            mode="qlib",
            extras=["adj_close"],
        )


def test_dataset_request_rejects_filename_override_in_qlib_mode():
    with pytest.raises(RequestContractError, match="Custom filenames are not supported in qlib mode."):
        DatasetRequest(
            tickers=["MSFT"],
            time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
            mode="qlib",
            filename_override="custom.csv",
        )


def test_eodhd_config_repr_and_request_serialization_do_not_expose_api_key():
    secret = "unit-test-secret"
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        external_validation=ExternalValidationConfig(
            provider="eodhd",
            enabled=True,
            eodhd=EODHDExternalValidationConfig(api_key=secret),
        ),
    )

    payload = request.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert secret not in repr(request.external_validation.eodhd)
    assert secret not in serialized
    assert payload["external_validation"]["eodhd"]["api_key_configured"] is True


def test_external_validation_rejects_enabled_true_without_resolvable_provider():
    config = ExternalValidationConfig(enabled=True)

    assert config.is_enabled() is False
    assert config.resolved_provider() is None
    assert config.to_dict()["status"] == "disabled"


def test_external_validation_rejects_explicit_eodhd_provider_without_api_key():
    config = ExternalValidationConfig(enabled=True, provider="eodhd")

    assert config.is_enabled() is False
    assert config.resolved_provider() == "eodhd"
    assert config.to_dict()["status"] == "disabled"
