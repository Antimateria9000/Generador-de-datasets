from __future__ import annotations

from dataset_core.presets import resolve_preset
from dataset_core.schema_builder import DatasetSchemaBuilder
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


def test_qlib_preset_ignores_non_qlib_extras():
    resolved = resolve_preset("qlib", ["dividends", "stock_splits", "adj_close"])
    assert resolved.output_columns == (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "factor",
        "adj_close",
    )
    assert resolved.ignored_extras == ("dividends", "stock_splits")


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
    assert result.factor_policy == "split_factor_only_from_raw_ohlcv"
