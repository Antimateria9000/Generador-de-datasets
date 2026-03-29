from __future__ import annotations

import pytest

from dataset_core.factor_policy import FactorPolicyError, apply_factor_policy, compute_split_factor, resolve_provider_flags
from tests.fixtures.sample_data import (
    make_nvda_like_frame_without_adj_close,
    make_nvda_like_split_frame,
    make_raw_split_frame,
)


def test_compute_split_factor_matches_reverse_running_rule():
    frame = make_raw_split_frame()
    factor = compute_split_factor(frame)
    assert factor.tolist() == pytest.approx([0.25, 1.0, 1.0, 1.0])


def test_apply_factor_policy_prefers_adj_close_ratio_when_close_is_already_split_adjusted():
    frame = make_nvda_like_split_frame()
    result = apply_factor_policy(frame, adjust_ohlcv=True)
    adjusted = result.frame

    assert result.factor_source == "adj_close_ratio"
    assert result.factor_policy == "qlib_adjusted_from_adj_close_ratio"
    assert adjusted["factor"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert adjusted["close"].tolist() == pytest.approx(frame["adj_close"].tolist())
    assert adjusted.loc[0, "volume"] == pytest.approx(frame.loc[0, "volume"])


def test_apply_factor_policy_uses_split_fallback_when_adj_close_is_unavailable_but_geometry_is_raw():
    frame = make_raw_split_frame()
    result = apply_factor_policy(frame, adjust_ohlcv=True)
    adjusted = result.frame

    assert result.factor_source == "stock_splits_fallback"
    assert result.factor_policy == "qlib_adjusted_from_split_fallback"
    assert adjusted["factor"].tolist() == pytest.approx([0.25, 1.0, 1.0, 1.0])
    assert adjusted.loc[0, "open"] == pytest.approx(100.0)
    assert adjusted.loc[0, "close"] == pytest.approx(102.5)
    assert adjusted.loc[0, "volume"] == pytest.approx(4000.0)
    assert any("stock_splits fallback" in warning for warning in result.warnings)


def test_apply_factor_policy_rejects_unsafe_split_fallback_when_close_is_already_adjusted():
    with pytest.raises(FactorPolicyError, match="double-adjust"):
        apply_factor_policy(make_nvda_like_frame_without_adj_close(), adjust_ohlcv=True)


def test_resolve_provider_flags_forces_raw_mode_when_factor_is_required():
    auto_adjust, actions, warnings = resolve_provider_flags(
        auto_adjust=True,
        actions=False,
        requires_factor=True,
    )
    assert auto_adjust is False
    assert actions is True
    assert len(warnings) == 2
