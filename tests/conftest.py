from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.sample_data import FakeContext, make_dq_context_payload, make_provider_frame


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return make_provider_frame()


@pytest.fixture
def reference_dir(tmp_path: Path, sample_frame: pd.DataFrame) -> Path:
    reference_dir = tmp_path / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(reference_dir / "MSFT.csv", index=False)
    sample_frame.to_csv(reference_dir / "AAPL.csv", index=False)
    return reference_dir


@pytest.fixture
def patch_market_context(monkeypatch):
    def fake_resolve_instrument_context(
        symbol: str,
        market_override=None,
        listing_preference: str = "exact_symbol",
        metadata_timeout=None,
        **kwargs,
    ):
        return FakeContext(
            requested_symbol=symbol.upper(),
            preferred_symbol=symbol.upper(),
            listing_preference=listing_preference,
            market=market_override or "XNYS",
            calendar=market_override or "XNYS",
        )

    monkeypatch.setattr(
        "dataset_core.export_service.resolve_instrument_context",
        fake_resolve_instrument_context,
    )
    monkeypatch.setattr(
        "dataset_core.export_service.build_dq_context_payload",
        make_dq_context_payload,
    )
    return fake_resolve_instrument_context
