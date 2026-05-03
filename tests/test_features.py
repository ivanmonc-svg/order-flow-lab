"""Tests for features.py — VWAP, delta, pivots, sweeps, engulfing, absorption."""

import numpy as np
import polars as pl
import pytest

from order_flow_lab.features import (
    classify_trades,
    compute_cvd,
    compute_vwap,
    compute_volume_profile,
    detect_absorption,
    detect_engulfing,
    detect_pivots,
    detect_sweeps,
)


class TestVWAP:
    def test_vwap_is_volume_weighted(self, synthetic_trades):
        result = compute_vwap(synthetic_trades)
        assert "vwap" in result.columns
        assert "vwap_upper" in result.columns
        assert "vwap_lower" in result.columns
        # VWAP should be between min and max price
        vwap_vals = result["vwap"].to_numpy()
        prices = result["price"].to_numpy()
        assert vwap_vals[-1] >= prices.min()
        assert vwap_vals[-1] <= prices.max()

    def test_vwap_bands_are_symmetric(self, synthetic_trades):
        result = compute_vwap(synthetic_trades)
        # Upper and lower should be equidistant from VWAP
        upper_dist = (result["vwap_upper"] - result["vwap"]).to_numpy()
        lower_dist = (result["vwap"] - result["vwap_lower"]).to_numpy()
        np.testing.assert_allclose(upper_dist, lower_dist, atol=1e-6)

    def test_upper_above_lower(self, synthetic_trades):
        result = compute_vwap(synthetic_trades)
        # Filter out first few rows where dev might be 0
        upper = result["vwap_upper"].to_numpy()[10:]
        lower = result["vwap_lower"].to_numpy()[10:]
        assert np.all(upper >= lower)


class TestTradeClassification:
    def test_classify_with_side_column(self, synthetic_trades):
        result = classify_trades(synthetic_trades)
        assert "trade_side" in result.columns
        assert "buy_volume" in result.columns
        assert "sell_volume" in result.columns
        assert "delta" in result.columns
        # Buy trades should have positive delta
        buys = result.filter(pl.col("trade_side") == 1)
        assert (buys["delta"] > 0).all()

    def test_classify_tick_rule_fallback(self, synthetic_trades):
        # Remove 'side' column to force tick rule
        df = synthetic_trades.drop("side")
        result = classify_trades(df)
        assert "trade_side" in result.columns

    def test_cvd_is_cumulative(self, synthetic_trades):
        classified = classify_trades(synthetic_trades)
        result = compute_cvd(classified)
        assert "cvd" in result.columns
        # CVD at end should equal sum of deltas
        expected = result["delta"].sum()
        actual = result["cvd"][-1]
        assert abs(actual - expected) < 1e-6


class TestVolumeProfile:
    def test_profile_has_required_columns(self, synthetic_trades):
        classified = classify_trades(synthetic_trades)
        profile = compute_volume_profile(classified, tick_size=0.25, size_col="size")
        assert "price_level" in profile.columns
        assert "total_volume" in profile.columns
        assert "volume_ratio" in profile.columns
        assert "buyer_seller_flag" in profile.columns

    def test_bs_flag_values(self, synthetic_trades):
        classified = classify_trades(synthetic_trades)
        profile = compute_volume_profile(classified, size_col="size")
        flags = profile["buyer_seller_flag"].unique().to_list()
        assert all(f in ("B", "S") for f in flags)


class TestPivots:
    def test_pivot_detection(self, synthetic_bars):
        result = detect_pivots(synthetic_bars, lookback=10)
        assert "is_pivot_high" in result.columns
        assert "is_pivot_low" in result.columns
        assert "pivot_high_level" in result.columns
        assert "pivot_low_level" in result.columns
        # Should detect at least some pivots
        n_ph = result["is_pivot_high"].sum()
        n_pl = result["is_pivot_low"].sum()
        assert n_ph > 0
        assert n_pl > 0

    def test_pivot_levels_forward_filled(self, synthetic_bars):
        result = detect_pivots(synthetic_bars, lookback=10)
        # After the first pivot, levels should not be NaN
        ph = result["pivot_high_level"].to_numpy()
        first_non_nan = np.argmax(~np.isnan(ph))
        if first_non_nan < len(ph) - 1:
            assert not np.any(np.isnan(ph[first_non_nan:]))


class TestSweeps:
    def test_sweep_detection(self, synthetic_bars):
        bars = detect_pivots(synthetic_bars, lookback=10)
        result = detect_sweeps(bars)
        assert "sweep_high" in result.columns
        assert "sweep_low" in result.columns
        # Sweep columns should be boolean
        assert result["sweep_high"].dtype == pl.Boolean
        assert result["sweep_low"].dtype == pl.Boolean


class TestEngulfing:
    def test_engulfing_detection(self, synthetic_bars):
        result = detect_engulfing(synthetic_bars)
        assert "bullish_engulfing" in result.columns
        assert "bearish_engulfing" in result.columns

    def test_engulfing_is_boolean(self, synthetic_bars):
        result = detect_engulfing(synthetic_bars)
        assert result["bullish_engulfing"].dtype == pl.Boolean
        assert result["bearish_engulfing"].dtype == pl.Boolean


class TestAbsorption:
    def test_absorption_returns_correct_schema(self, synthetic_trades):
        classified = classify_trades(synthetic_trades)
        result = detect_absorption(classified, window_seconds=10.0)
        expected_cols = {
            "ts_start", "ts_end", "price_level",
            "total_volume", "price_range_ticks",
            "net_delta", "absorption_side",
        }
        assert set(result.columns) == expected_cols

    def test_absorption_side_values(self, synthetic_trades):
        classified = classify_trades(synthetic_trades)
        result = detect_absorption(classified, window_seconds=10.0)
        if len(result) > 0:
            sides = result["absorption_side"].unique().to_list()
            assert all(s in ("bid", "ask") for s in sides)
