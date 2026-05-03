"""Tests for strategy.py and backtest.py — signal generation and backtesting."""

import numpy as np
import polars as pl
import pytest

from order_flow_lab.strategy import (
    Side,
    Signal,
    StrategyConfig,
    generate_signals,
    prepare_bars,
)
from order_flow_lab.backtest import BacktestEngine, BacktestResult, trades_to_dataframe


class TestPrepareBars:
    def test_prepare_adds_all_columns(self, synthetic_bars):
        result = prepare_bars(synthetic_bars)
        required = {
            "vwap", "vwap_upper", "vwap_lower",
            "is_pivot_high", "is_pivot_low",
            "pivot_high_level", "pivot_low_level",
            "sweep_high", "sweep_low",
            "bullish_engulfing", "bearish_engulfing",
        }
        assert required.issubset(set(result.columns))

    def test_vwap_between_price_range(self, synthetic_bars):
        result = prepare_bars(synthetic_bars)
        vwap = result["vwap"].to_numpy()
        lows = result["low"].to_numpy()
        highs = result["high"].to_numpy()
        # VWAP should be within the overall price range
        assert vwap[-1] >= lows.min()
        assert vwap[-1] <= highs.max()


class TestSignalGeneration:
    def test_signals_from_synthetic(self, synthetic_bars):
        prepped = prepare_bars(synthetic_bars, config=StrategyConfig(require_engulfing=False))
        signals = generate_signals(prepped, config=StrategyConfig(require_engulfing=False))
        # May or may not generate signals with random data, but should not crash
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)
            assert sig.side in (Side.LONG, Side.SHORT)

    def test_long_signal_has_valid_levels(self, synthetic_bars):
        config = StrategyConfig(require_engulfing=False, volume_ratio_min=0.0)
        prepped = prepare_bars(synthetic_bars, config=config)
        signals = generate_signals(prepped, config=config)
        for sig in signals:
            if sig.side == Side.LONG:
                assert sig.tp1 > sig.entry_price, "TP1 should be above entry for longs"
                assert sig.stop_loss < sig.entry_price, "SL should be below entry for longs"

    def test_short_signal_has_valid_levels(self, synthetic_bars):
        config = StrategyConfig(require_engulfing=False, volume_ratio_min=0.0)
        prepped = prepare_bars(synthetic_bars, config=config)
        signals = generate_signals(prepped, config=config)
        for sig in signals:
            if sig.side == Side.SHORT:
                assert sig.tp1 < sig.entry_price, "TP1 should be below entry for shorts"
                assert sig.stop_loss > sig.entry_price, "SL should be above entry for shorts"


class TestBacktest:
    def test_backtest_runs_without_error(self, synthetic_bars):
        config = StrategyConfig(require_engulfing=False, volume_ratio_min=0.0)
        prepped = prepare_bars(synthetic_bars, config=config)
        signals = generate_signals(prepped, config=config)
        engine = BacktestEngine(config=config, point_value=50.0)
        result = engine.run(prepped, signals)
        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 0

    def test_backtest_metrics_consistency(self, synthetic_bars):
        config = StrategyConfig(require_engulfing=False, volume_ratio_min=0.0)
        prepped = prepare_bars(synthetic_bars, config=config)
        signals = generate_signals(prepped, config=config)
        engine = BacktestEngine(config=config, point_value=50.0)
        result = engine.run(prepped, signals)
        assert result.num_winners + result.num_losers <= result.num_trades
        if result.num_trades > 0:
            assert 0 <= result.win_rate <= 1

    def test_empty_signals_produces_empty_result(self, synthetic_bars):
        config = StrategyConfig()
        prepped = prepare_bars(synthetic_bars, config=config)
        engine = BacktestEngine(config=config)
        result = engine.run(prepped, [])
        assert result.num_trades == 0
        assert result.total_pnl_points == 0

    def test_trades_to_dataframe(self, synthetic_bars):
        config = StrategyConfig(require_engulfing=False, volume_ratio_min=0.0)
        prepped = prepare_bars(synthetic_bars, config=config)
        signals = generate_signals(prepped, config=config)
        engine = BacktestEngine(config=config)
        result = engine.run(prepped, signals)
        df = trades_to_dataframe(result.trades)
        assert isinstance(df, pl.DataFrame)
        if len(result.trades) > 0:
            assert "pnl_points" in df.columns
            assert "exit_reason" in df.columns

    def test_result_summary_string(self, synthetic_bars):
        config = StrategyConfig(require_engulfing=False, volume_ratio_min=0.0)
        prepped = prepare_bars(synthetic_bars, config=config)
        signals = generate_signals(prepped, config=config)
        engine = BacktestEngine(config=config)
        result = engine.run(prepped, signals)
        summary = result.summary()
        assert "Trades:" in summary
        assert "Win rate:" in summary
