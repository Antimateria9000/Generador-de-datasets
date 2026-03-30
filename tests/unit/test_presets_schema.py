from __future__ import annotations

import pytest

from dataset_core.presets import resolve_preset
from dataset_core.schema_builder import DatasetSchemaBuilder, SchemaBuildError
from tests.fixtures.sample_data import make_provider_frame


def test_extended_preset_keeps_default_market_extras():
    resolved = resolve_preset("extended", [])
    assert resolved.output_columns == (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "dividends",
        "stock_splits",
    )


def test_qlib_preset_emits_closed_contract_columns():
    resolved = resolve_preset("qlib", [])
    assert resolved.output_columns == (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "factor",
    )
    assert resolved.selected_extras == ("factor",)


def test_qlib_preset_rejects_forbidden_extras():
    with pytest.raises(ValueError, match="closed"):
        resolve_preset("qlib", ["dividends", "adj_close"])


def test_schema_builder_extended_output_contains_requested_factor():
    builder = DatasetSchemaBuilder()
    frame = make_provider_frame()
    result = builder.build(frame=frame, mode="extended", extras=["factor"])
    assert result.columns == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "dividends",
        "stock_splits",
        "factor",
    ]
    assert result.factor_policy == "factor_only_from_adj_close_ratio"
    assert result.factor_source == "adj_close_ratio"


def test_schema_builder_rejects_closed_qlib_contract_bypass():
    builder = DatasetSchemaBuilder()
    with pytest.raises(SchemaBuildError, match="closed"):
        builder.build(frame=make_provider_frame(), mode="qlib", extras=["adj_close"])
