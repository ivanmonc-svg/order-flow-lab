"""Tests for book.py — Phase 2: typed MBP-10 parser, book reconstruction,
absorption flags, sweep detection, level imbalances, summary DF.
"""

import numpy as np
import polars as pl
import pytest

from order_flow_lab.book import (
    MBP10_SCHEMA,
    BookLevel,
    BookSnapshot,
    detect_absorption_from_book,
    detect_large_orders,
    detect_sweeps_from_book,
    parse_mbp10,
    reconstruct_book,
    snapshots_to_heatmap_df,
    snapshots_to_summary_df,
)


# ════════════════════════════════════════════════════════════════════════════
# parse_mbp10
# ════════════════════════════════════════════════════════════════════════════


class TestParseMbp10:
    def test_output_has_exact_schema(self, synthetic_mbp10):
        parsed = parse_mbp10(synthetic_mbp10)
        assert set(parsed.columns) == set(MBP10_SCHEMA.keys())

    def test_column_dtypes_match_schema(self, synthetic_mbp10):
        parsed = parse_mbp10(synthetic_mbp10)
        for col, expected_dtype in MBP10_SCHEMA.items():
            assert parsed[col].dtype == expected_dtype, (
                f"Column {col}: expected {expected_dtype}, got {parsed[col].dtype}"
            )

    def test_row_count_preserved(self, synthetic_mbp10):
        parsed = parse_mbp10(synthetic_mbp10)
        assert len(parsed) == len(synthetic_mbp10)

    def test_missing_columns_filled_with_defaults(self):
        """If a raw DF is missing some columns, parse_mbp10 fills them."""
        raw = pl.DataFrame({
            "ts_event": [1_000_000_000, 2_000_000_000],
            "price": [5500.0, 5500.25],
        })
        parsed = parse_mbp10(raw)
        assert len(parsed) == 2
        # Missing size column should be filled with 0
        assert parsed["size"].to_list() == [0, 0]
        # Missing string columns filled with ""
        assert parsed["action"].to_list() == ["", ""]

    def test_extra_columns_dropped(self, synthetic_mbp10):
        raw = synthetic_mbp10.with_columns(pl.lit("extra").alias("garbage_col"))
        parsed = parse_mbp10(raw)
        assert "garbage_col" not in parsed.columns


# ════════════════════════════════════════════════════════════════════════════
# reconstruct_book
# ════════════════════════════════════════════════════════════════════════════


class TestReconstructBook:
    def test_returns_snapshots(self, synthetic_mbp10):
        snapshots = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        assert len(snapshots) == 100
        assert isinstance(snapshots[0], BookSnapshot)

    def test_snapshot_has_bids_and_asks(self, synthetic_mbp10):
        snapshots = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        snap = snapshots[0]
        assert len(snap.bids) > 0
        assert len(snap.asks) > 0

    def test_best_bid_below_best_ask(self, synthetic_mbp10):
        snapshots = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        for snap in snapshots:
            if snap.best_bid is not None and snap.best_ask is not None:
                assert snap.best_bid < snap.best_ask

    def test_mid_price(self, synthetic_mbp10):
        snapshots = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        snap = snapshots[0]
        expected_mid = (snap.best_bid + snap.best_ask) / 2
        assert abs(snap.mid - expected_mid) < 1e-6

    def test_book_imbalance_range(self, synthetic_mbp10):
        snapshots = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        for snap in snapshots:
            assert -1 <= snap.book_imbalance <= 1

    def test_downsampling_reduces_count(self, synthetic_mbp10):
        """100 messages at 1s intervals → with 10s freq → ~10 snapshots."""
        all_snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        sampled = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=10_000)
        assert len(sampled) < len(all_snaps)
        assert len(sampled) == 10  # 100s of data / 10s window

    def test_freq_ms_zero_keeps_all(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        assert len(snaps) == len(synthetic_mbp10)

    def test_freq_ms_100_default(self, synthetic_mbp10):
        """Snapshots at 1s intervals with 100ms freq → each stays (no merging)."""
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=100)
        # Each message is 1s apart, 100ms windows → each in its own bucket
        assert len(snaps) == 100

    def test_empty_input(self):
        empty = pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Int64)})
        snaps = reconstruct_book(empty, snapshot_freq_ms=100)
        assert snaps == []


# ════════════════════════════════════════════════════════════════════════════
# BookSnapshot.level_imbalances
# ════════════════════════════════════════════════════════════════════════════


class TestLevelImbalances:
    def test_balanced_book(self):
        bids = [BookLevel(price=100 - i * 0.25, size=50, count=5) for i in range(3)]
        asks = [BookLevel(price=100 + (i + 1) * 0.25, size=50, count=5) for i in range(3)]
        snap = BookSnapshot(ts_event=0, bids=bids, asks=asks)
        imbs = snap.level_imbalances(n_levels=3)
        assert len(imbs) == 3
        assert all(abs(v) < 1e-10 for v in imbs)  # perfectly balanced

    def test_bid_dominated(self):
        bids = [BookLevel(price=100.0, size=100, count=10)]
        asks = [BookLevel(price=100.25, size=10, count=1)]
        snap = BookSnapshot(ts_event=0, bids=bids, asks=asks)
        imbs = snap.level_imbalances(n_levels=1)
        assert len(imbs) == 1
        # (100 - 10) / (100 + 10) ≈ 0.818
        assert imbs[0] > 0.5

    def test_ask_dominated(self):
        bids = [BookLevel(price=100.0, size=10, count=1)]
        asks = [BookLevel(price=100.25, size=100, count=10)]
        snap = BookSnapshot(ts_event=0, bids=bids, asks=asks)
        imbs = snap.level_imbalances(n_levels=1)
        assert imbs[0] < -0.5

    def test_n_levels_defaults_to_min(self):
        bids = [BookLevel(price=100 - i * 0.25, size=50, count=5) for i in range(5)]
        asks = [BookLevel(price=100 + (i + 1) * 0.25, size=50, count=5) for i in range(3)]
        snap = BookSnapshot(ts_event=0, bids=bids, asks=asks)
        imbs = snap.level_imbalances()  # should default to min(5, 3) = 3
        assert len(imbs) == 3


# ════════════════════════════════════════════════════════════════════════════
# snapshots_to_heatmap_df
# ════════════════════════════════════════════════════════════════════════════


class TestHeatmapDf:
    def test_columns(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_heatmap_df(snaps)
        assert set(df.columns) == {"ts_event", "price", "size", "side"}

    def test_sides_present(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_heatmap_df(snaps)
        sides = df["side"].unique().to_list()
        assert "bid" in sides
        assert "ask" in sides

    def test_row_count(self, synthetic_mbp10):
        """Each snapshot has 10 bids + 10 asks = 20 rows → 100 snaps × 20."""
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_heatmap_df(snaps)
        assert len(df) == 100 * 20


# ════════════════════════════════════════════════════════════════════════════
# snapshots_to_summary_df
# ════════════════════════════════════════════════════════════════════════════


class TestSummaryDf:
    def test_columns(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_summary_df(snaps)
        expected = {
            "ts_event", "best_bid", "best_ask", "mid", "spread",
            "total_bid_size", "total_ask_size", "book_imbalance",
            "imb_l0", "imb_l1", "imb_l2",
        }
        assert set(df.columns) == expected

    def test_row_count(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_summary_df(snaps)
        assert len(df) == 100

    def test_imbalance_range(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_summary_df(snaps)
        for col in ["book_imbalance", "imb_l0", "imb_l1", "imb_l2"]:
            vals = df[col].to_numpy()
            assert np.all(vals >= -1.0) and np.all(vals <= 1.0), f"{col} out of range"

    def test_spread_positive(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        df = snapshots_to_summary_df(snaps)
        assert (df["spread"] > 0).all()


# ════════════════════════════════════════════════════════════════════════════
# detect_sweeps_from_book
# ════════════════════════════════════════════════════════════════════════════


class TestSweepsFromBook:
    def test_detects_ask_sweep(self, synthetic_mbp10_with_sweep):
        snaps = reconstruct_book(synthetic_mbp10_with_sweep, snapshot_freq_ms=0)
        result = detect_sweeps_from_book(snaps, min_levels=3, max_time_ms=500)
        assert len(result) > 0
        assert "ask_sweep" in result["side"].to_list()

    def test_sweep_columns(self, synthetic_mbp10_with_sweep):
        snaps = reconstruct_book(synthetic_mbp10_with_sweep, snapshot_freq_ms=0)
        result = detect_sweeps_from_book(snaps, min_levels=3, max_time_ms=500)
        expected_cols = {
            "ts_start", "ts_end", "side", "levels_consumed",
            "price_start", "price_end", "duration_ms",
        }
        assert set(result.columns) == expected_cols

    def test_no_sweep_below_threshold(self, synthetic_mbp10):
        """Random data with small moves should not trigger sweeps at high threshold."""
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        result = detect_sweeps_from_book(snaps, min_levels=20, max_time_ms=50)
        assert len(result) == 0

    def test_empty_returns_schema(self):
        result = detect_sweeps_from_book([], min_levels=3)
        assert "side" in result.columns
        assert len(result) == 0


# ════════════════════════════════════════════════════════════════════════════
# detect_absorption_from_book
# ════════════════════════════════════════════════════════════════════════════


class TestAbsorptionFromBook:
    def test_detects_absorption(self, synthetic_absorption_data):
        mbp_df, trades_df = synthetic_absorption_data
        snaps = reconstruct_book(mbp_df, snapshot_freq_ms=0)
        result = detect_absorption_from_book(
            snaps, trades_df,
            window_ns=1_000_000_000,
            executed_vs_visible_ratio=3.0,
        )
        assert len(result) > 0
        # Ratio should be high (500 executed / ~20 visible ≈ 25)
        assert (result["absorption_ratio"] >= 3.0).all()

    def test_absorption_columns(self, synthetic_absorption_data):
        mbp_df, trades_df = synthetic_absorption_data
        snaps = reconstruct_book(mbp_df, snapshot_freq_ms=0)
        result = detect_absorption_from_book(snaps, trades_df)
        expected_cols = {
            "ts_window", "price_level", "executed_volume", "visible_size",
            "absorption_ratio", "absorption_side",
        }
        assert set(result.columns) == expected_cols

    def test_no_absorption_when_sizes_match(self, synthetic_mbp10, synthetic_trades):
        """With default synthetic data, absorption should be rare or absent
        because trade sizes are not dramatically larger than book sizes."""
        from order_flow_lab.features import classify_trades
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        classified = classify_trades(synthetic_trades)
        result = detect_absorption_from_book(
            snaps, classified,
            executed_vs_visible_ratio=100.0,  # very high threshold
        )
        # With ratio=100, unlikely to find any
        assert len(result) == 0

    def test_empty_returns_schema(self):
        result = detect_absorption_from_book(
            [], pl.DataFrame({"ts_event": [], "price": [], "size": [], "delta": []}),
        )
        assert "absorption_side" in result.columns
        assert len(result) == 0


# ════════════════════════════════════════════════════════════════════════════
# detect_large_orders
# ════════════════════════════════════════════════════════════════════════════


class TestLargeOrders:
    def test_threshold_filtering(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        result = detect_large_orders(snaps, threshold_contracts=50)
        assert "price" in result.columns
        assert "size" in result.columns
        if len(result) > 0:
            assert (result["size"] >= 50).all()

    def test_high_threshold_returns_fewer(self, synthetic_mbp10):
        snaps = reconstruct_book(synthetic_mbp10, snapshot_freq_ms=0)
        low = detect_large_orders(snaps, threshold_contracts=50)
        high = detect_large_orders(snaps, threshold_contracts=150)
        assert len(high) <= len(low)
