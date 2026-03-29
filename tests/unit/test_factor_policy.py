from __future__ import annotations

import pandas as pd
import pytest

from dataset_core.factor_policy import apply_factor_policy, compute_split_factor, resolve_provider_flags
from tests.fixtures.sample_data import make_split_frame


def test_compute_split_factor_matches_reverse_running_rule():
    frame = make_split_frame()
    factor = compute_split_factor(frame)
    assert factor.tolist() == pytest.approx([0.25, 1.0, 1.0, 1.0])


def test_apply_factor_policy_adjusts_ohlcv_for_qlib():
    frame = make_split_frame()
    result = apply_factor_policy(frame, adjust_ohlcv=True)
    adjusted = result.frame

    assert adjusted["factor"].tolist() == pytest.approx([0.25, 1.0, 1.0, 1.0])
    assert adjusted.loc[0, "open"] == pytest.approx(100.0)
    assert adjusted.loc[0, "close"] == pytest.approx(102.5)
    assert adjusted.loc[0, "volume"] == pytest.approx(4000.0)
    assert result.factor_policy == "qlib_split_adjusted_from_raw_ohlcv"


def test_resolve_provider_flags_forces_raw_mode_when_factor_is_required():
    auto_adjust, actions, warnings = resolve_provider_flags(
        auto_adjust=True,
        actions=False,
        requires_factor=True,
    )
    assert auto_adjust is False
    assert actions is True
    assert len(warnings) == 2
