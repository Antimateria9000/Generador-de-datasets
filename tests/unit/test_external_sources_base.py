from __future__ import annotations

import pandas as pd

from dataset_core.external_sources.base import (
    attach_source_metadata,
    extract_source_metadata,
    filter_event_frame,
    filter_reference_frame,
    normalize_event_frame,
    normalize_reference_frame,
)


def test_normalize_reference_frame_preserves_source_metadata_for_empty_frames():
    frame = attach_source_metadata(pd.DataFrame(columns=["date", "close"]), {"provider": "eodhd", "scope": "price"})

    normalized = normalize_reference_frame(frame)

    assert normalized.empty
    assert extract_source_metadata(normalized) == {"provider": "eodhd", "scope": "price"}


def test_normalize_event_frame_preserves_source_metadata_when_rows_are_filtered_out():
    frame = attach_source_metadata(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "dividends": [0.0],
                "stock_splits": [0.0],
            }
        ),
        {"provider": "eodhd", "scope": "event", "partial_coverage": True},
    )

    normalized = normalize_event_frame(frame)

    assert normalized.empty
    assert extract_source_metadata(normalized)["partial_coverage"] is True


def test_filter_reference_frame_preserves_metadata_and_uses_end_exclusive():
    frame = attach_source_metadata(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "close": [1.0, 2.0, 3.0],
            }
        ),
        {"provider": "eodhd", "effective_end": "2024-01-03"},
    )

    filtered = filter_reference_frame(frame, start="2024-01-02", end="2024-01-03")

    assert filtered["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    assert extract_source_metadata(filtered)["provider"] == "eodhd"


def test_filter_event_frame_preserves_metadata_and_uses_end_exclusive():
    frame = attach_source_metadata(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "dividends": [0.5, 0.6],
                "stock_splits": [0.0, 0.0],
            }
        ),
        {"provider": "eodhd", "scope": "event"},
    )

    filtered = filter_event_frame(frame, start="2024-01-01", end="2024-01-02")

    assert filtered["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01"]
    assert extract_source_metadata(filtered)["provider"] == "eodhd"
