"""Limit Order Book (LOB) reconstruction from Databento MBP-10 data.

Phase 2 implementation:
  - Typed Polars schema for MBP-10 parsing
  - reconstruct_book(messages, snapshot_freq_ms=100) with time-based snapshots
  - Per-level imbalance, absorption flags, and sweep detection
  - Efficient columnar operations via Polars (no Python-loop hot path for large data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. TYPED SCHEMA FOR MBP-10
# ════════════════════════════════════════════════════════════════════════════

N_LEVELS = 10

# Databento MBP-10 flat column schema (after .to_df() → polars conversion)
# Prices come in as fixed-point int64 (1e-9 scale) from DBN; after to_df()
# they are already float64.  Sizes and counts are uint32.


def _level_columns() -> dict[str, pl.DataType]:
    """Generate the typed schema for 10 bid/ask levels."""
    cols: dict[str, pl.DataType] = {}
    for i in range(N_LEVELS):
        sfx = f"{i:02d}"
        cols[f"bid_px_{sfx}"] = pl.Float64
        cols[f"bid_sz_{sfx}"] = pl.UInt32
        cols[f"bid_ct_{sfx}"] = pl.UInt32
        cols[f"ask_px_{sfx}"] = pl.Float64
        cols[f"ask_sz_{sfx}"] = pl.UInt32
        cols[f"ask_ct_{sfx}"] = pl.UInt32
    return cols


MBP10_SCHEMA: dict[str, pl.DataType] = {
    "ts_event": pl.Int64,       # nanosecond epoch
    "ts_recv": pl.Int64,        # nanosecond epoch (receive time)
    "price": pl.Float64,        # event price (the update that triggered this snapshot)
    "size": pl.UInt32,          # event size
    "action": pl.Utf8,          # 'T' trade, 'A' add, 'C' cancel, 'M' modify, 'R' clear
    "side": pl.Utf8,            # 'A' ask, 'B' bid, 'N' none
    "flags": pl.UInt8,
    "sequence": pl.UInt64,
    **_level_columns(),
}

# Subset for the header (non-level) columns
MBP10_HEADER_COLS = [
    "ts_event", "ts_recv", "price", "size", "action", "side", "flags", "sequence",
]


def parse_mbp10(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Cast a raw MBP-10 DataFrame to strict typed schema.

    Handles both Databento SDK output (from .to_df()) and CSV/parquet imports.
    Missing columns are filled with defaults; extra columns are dropped.

    Returns:
        Polars DataFrame with guaranteed schema from MBP10_SCHEMA.
    """
    result_cols = {}
    for col_name, dtype in MBP10_SCHEMA.items():
        if col_name in raw_df.columns:
            result_cols[col_name] = raw_df[col_name].cast(dtype, strict=False)
        else:
            # Fill missing columns with sensible defaults
            n = len(raw_df)
            if dtype == pl.Float64:
                result_cols[col_name] = pl.Series(col_name, [0.0] * n, dtype=pl.Float64)
            elif dtype == pl.UInt32:
                result_cols[col_name] = pl.Series(col_name, [0] * n, dtype=pl.UInt32)
            elif dtype == pl.Int64:
                result_cols[col_name] = pl.Series(col_name, [0] * n, dtype=pl.Int64)
            elif dtype == pl.UInt64:
                result_cols[col_name] = pl.Series(col_name, [0] * n, dtype=pl.UInt64)
            elif dtype == pl.UInt8:
                result_cols[col_name] = pl.Series(col_name, [0] * n, dtype=pl.UInt8)
            elif dtype == pl.Utf8:
                result_cols[col_name] = pl.Series(col_name, [""] * n, dtype=pl.Utf8)

    df = pl.DataFrame(result_cols)
    logger.info("Parsed MBP-10: %d rows, %d columns", len(df), len(df.columns))
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. BOOK SNAPSHOT DATACLASS
# ════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class BookLevel:
    """A single price level in the order book."""

    price: float
    size: int
    count: int  # number of orders at this level


@dataclass
class BookSnapshot:
    """A point-in-time snapshot of the order book."""

    ts_event: int  # nanosecond timestamp
    bids: list[BookLevel] = field(default_factory=list)  # sorted best (highest) → worst
    asks: list[BookLevel] = field(default_factory=list)  # sorted best (lowest) → worst

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def total_bid_size(self) -> int:
        return sum(lv.size for lv in self.bids)

    @property
    def total_ask_size(self) -> int:
        return sum(lv.size for lv in self.asks)

    @property
    def book_imbalance(self) -> float:
        """Bid-ask imbalance: +1 = all bids, -1 = all asks, 0 = balanced."""
        total = self.total_bid_size + self.total_ask_size
        if total == 0:
            return 0.0
        return (self.total_bid_size - self.total_ask_size) / total

    def level_imbalances(self, n_levels: int | None = None) -> list[float]:
        """Per-level bid/ask imbalance for the top N levels.

        Returns list of imbalance values [-1, +1] for each level pair.
        +1 = bid dominates, -1 = ask dominates at that depth.
        """
        n = n_levels or min(len(self.bids), len(self.asks))
        imbalances = []
        for i in range(n):
            bid_sz = self.bids[i].size if i < len(self.bids) else 0
            ask_sz = self.asks[i].size if i < len(self.asks) else 0
            total = bid_sz + ask_sz
            if total == 0:
                imbalances.append(0.0)
            else:
                imbalances.append((bid_sz - ask_sz) / total)
        return imbalances


# ════════════════════════════════════════════════════════════════════════════
# 3. CORE: reconstruct_book() WITH CONFIGURABLE SNAPSHOT FREQUENCY
# ════════════════════════════════════════════════════════════════════════════


def _row_to_snapshot(row: dict, n_levels: int = N_LEVELS) -> BookSnapshot:
    """Extract a BookSnapshot from a single MBP-10 row dict."""
    bids, asks = [], []
    for i in range(n_levels):
        sfx = f"{i:02d}"
        bp = row.get(f"bid_px_{sfx}", 0.0)
        bs = row.get(f"bid_sz_{sfx}", 0)
        bc = row.get(f"bid_ct_{sfx}", 1)
        if bp > 0 and bs > 0:
            bids.append(BookLevel(price=float(bp), size=int(bs), count=int(bc)))

        ap = row.get(f"ask_px_{sfx}", 0.0)
        as_ = row.get(f"ask_sz_{sfx}", 0)
        ac = row.get(f"ask_ct_{sfx}", 1)
        if ap > 0 and as_ > 0:
            asks.append(BookLevel(price=float(ap), size=int(as_), count=int(ac)))

    return BookSnapshot(ts_event=int(row["ts_event"]), bids=bids, asks=asks)


def reconstruct_book(
    messages: pl.DataFrame,
    snapshot_freq_ms: int = 100,
) -> list[BookSnapshot]:
    """Reconstruct the order book from MBP-10 messages at a fixed frequency.

    Databento MBP-10 provides a full 10-level book snapshot with every update.
    This function down-samples to one snapshot per `snapshot_freq_ms` window by
    keeping only the LAST message in each window (most up-to-date book state).

    Args:
        messages: MBP-10 DataFrame (raw or parsed via parse_mbp10).
        snapshot_freq_ms: Snapshot interval in milliseconds. Default 100ms (10 Hz).
                         Use 0 to keep every message (no down-sampling).

    Returns:
        List of BookSnapshot objects at the requested frequency.
    """
    if messages.is_empty():
        return []

    df = messages.sort("ts_event")

    if snapshot_freq_ms <= 0:
        # No down-sampling: one snapshot per message
        snapshots = [_row_to_snapshot(row) for row in df.iter_rows(named=True)]
        logger.info("Reconstructed %d snapshots (every message)", len(snapshots))
        return snapshots

    # Down-sample: bucket by time window, keep last message per bucket
    freq_ns = snapshot_freq_ms * 1_000_000  # ms → ns

    df = df.with_columns(
        (pl.col("ts_event") // freq_ns).alias("_time_bucket"),
    )

    # Keep the last row per bucket (most recent book state in each window)
    sampled = df.group_by("_time_bucket", maintain_order=True).last().drop("_time_bucket")

    snapshots = [_row_to_snapshot(row) for row in sampled.iter_rows(named=True)]
    logger.info(
        "Reconstructed %d snapshots at %d ms intervals (from %d messages)",
        len(snapshots), snapshot_freq_ms, len(df),
    )
    return snapshots


# ════════════════════════════════════════════════════════════════════════════
# 4. COLUMNAR SNAPSHOT OUTPUT (for heatmap / vectorized features)
# ════════════════════════════════════════════════════════════════════════════


def snapshots_to_heatmap_df(
    snapshots: list[BookSnapshot],
) -> pl.DataFrame:
    """Flatten book snapshots into a long-form DataFrame for heatmap rendering.

    Returns DataFrame:
        ts_event (i64), price (f64), size (u32), side ('bid'|'ask')
    """
    ts_list, price_list, size_list, side_list = [], [], [], []

    for snap in snapshots:
        for lv in snap.bids:
            ts_list.append(snap.ts_event)
            price_list.append(lv.price)
            size_list.append(lv.size)
            side_list.append("bid")
        for lv in snap.asks:
            ts_list.append(snap.ts_event)
            price_list.append(lv.price)
            size_list.append(lv.size)
            side_list.append("ask")

    return pl.DataFrame({
        "ts_event": pl.Series("ts_event", ts_list, dtype=pl.Int64),
        "price": pl.Series("price", price_list, dtype=pl.Float64),
        "size": pl.Series("size", size_list, dtype=pl.Int64),
        "side": pl.Series("side", side_list, dtype=pl.Utf8),
    })


def snapshots_to_summary_df(
    snapshots: list[BookSnapshot],
) -> pl.DataFrame:
    """Convert snapshots to a summary DataFrame with aggregate metrics.

    Returns DataFrame:
        ts_event, best_bid, best_ask, mid, spread,
        total_bid_size, total_ask_size, book_imbalance,
        imb_l0, imb_l1, imb_l2 (level imbalances for top 3 levels)
    """
    records = []
    for snap in snapshots:
        imbs = snap.level_imbalances(n_levels=3)
        records.append({
            "ts_event": snap.ts_event,
            "best_bid": snap.best_bid or 0.0,
            "best_ask": snap.best_ask or 0.0,
            "mid": snap.mid or 0.0,
            "spread": snap.spread or 0.0,
            "total_bid_size": snap.total_bid_size,
            "total_ask_size": snap.total_ask_size,
            "book_imbalance": snap.book_imbalance,
            "imb_l0": imbs[0] if len(imbs) > 0 else 0.0,
            "imb_l1": imbs[1] if len(imbs) > 1 else 0.0,
            "imb_l2": imbs[2] if len(imbs) > 2 else 0.0,
        })
    return pl.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# 5. ABSORPTION FLAG DETECTION (from book + trade data)
# ════════════════════════════════════════════════════════════════════════════


def detect_absorption_from_book(
    snapshots: list[BookSnapshot],
    trades: pl.DataFrame,
    window_ns: int = 1_000_000_000,  # 1 second
    executed_vs_visible_ratio: float = 3.0,
    tick_size: float = 0.25,
) -> pl.DataFrame:
    """Detect absorption: volume executed at a level >> size visible in the book.

    The key insight from the video: institutional players use iceberg orders
    that show a small visible size but absorb many contracts. We detect this
    by comparing executed trade volume at a price level (from tape) against
    the resting size shown in the book at that level.

    Logic per time window:
        1. Sum executed trade volume at each price level
        2. Get the average resting book size at that level during the window
        3. If executed / visible >= ratio, flag as absorption

    Args:
        snapshots: Book snapshots (from reconstruct_book).
        trades: Trade DataFrame with ts_event, price, size, trade_side columns
                (from features.classify_trades).
        window_ns: Aggregation window in nanoseconds (default 1s).
        executed_vs_visible_ratio: Minimum ratio to flag absorption.
        tick_size: Tick size for price bucketing.

    Returns DataFrame:
        ts_window, price_level, executed_volume, visible_size,
        absorption_ratio, absorption_side ('bid'|'ask')
    """
    if not snapshots or trades.is_empty():
        return pl.DataFrame(schema={
            "ts_window": pl.Int64, "price_level": pl.Float64,
            "executed_volume": pl.Int64, "visible_size": pl.Int64,
            "absorption_ratio": pl.Float64, "absorption_side": pl.Utf8,
        })

    # Bucket trades by time window and price level
    trade_agg = (
        trades.with_columns(
            (pl.col("ts_event") // window_ns).alias("ts_window"),
            ((pl.col("price") / tick_size).round(0) * tick_size).alias("price_level"),
        )
        .group_by(["ts_window", "price_level"])
        .agg(
            pl.col("size").sum().alias("executed_volume"),
            pl.col("delta").sum().alias("net_delta"),
        )
    )

    # Build a lookup of visible sizes from book snapshots
    # For each snapshot, record resting sizes at each price level
    visible_records = []
    for snap in snapshots:
        bucket = snap.ts_event // window_ns
        for lv in snap.bids:
            rounded = round(lv.price / tick_size) * tick_size
            visible_records.append({
                "ts_window": bucket,
                "price_level": rounded,
                "visible_size": lv.size,
                "book_side": "bid",
            })
        for lv in snap.asks:
            rounded = round(lv.price / tick_size) * tick_size
            visible_records.append({
                "ts_window": bucket,
                "price_level": rounded,
                "visible_size": lv.size,
                "book_side": "ask",
            })

    if not visible_records:
        return pl.DataFrame(schema={
            "ts_window": pl.Int64, "price_level": pl.Float64,
            "executed_volume": pl.Int64, "visible_size": pl.Int64,
            "absorption_ratio": pl.Float64, "absorption_side": pl.Utf8,
        })

    visible_df = (
        pl.DataFrame(visible_records)
        .group_by(["ts_window", "price_level"])
        .agg(
            pl.col("visible_size").mean().cast(pl.Int64).alias("visible_size"),
            pl.col("book_side").first().alias("absorption_side"),
        )
    )

    # Join and compute ratio
    result = (
        trade_agg.join(visible_df, on=["ts_window", "price_level"], how="inner")
        .with_columns(
            pl.when(pl.col("visible_size") > 0)
            .then(pl.col("executed_volume").cast(pl.Float64) / pl.col("visible_size"))
            .otherwise(pl.lit(float("inf")))
            .alias("absorption_ratio"),
        )
        .filter(pl.col("absorption_ratio") >= executed_vs_visible_ratio)
        .select([
            "ts_window", "price_level", "executed_volume", "visible_size",
            "absorption_ratio", "absorption_side",
        ])
        .sort(["ts_window", "price_level"])
    )

    logger.info(
        "Detected %d absorption events (ratio >= %.1f)",
        len(result), executed_vs_visible_ratio,
    )
    return result


# ════════════════════════════════════════════════════════════════════════════
# 6. SWEEP DETECTION (N levels consumed in < T ms)
# ════════════════════════════════════════════════════════════════════════════


def detect_sweeps_from_book(
    snapshots: list[BookSnapshot],
    min_levels: int = 3,
    max_time_ms: int = 500,
    tick_size: float = 0.25,
) -> pl.DataFrame:
    """Detect sweep events: aggressive consumption of multiple price levels.

    A sweep occurs when the best bid (or ask) moves through N or more price
    levels within T milliseconds, indicating an aggressive market order or
    iceberg eating through resting liquidity.

    Algorithm:
        For each consecutive pair of snapshots, compute how many price levels
        the best bid/ask moved. If it moved >= min_levels within max_time_ms,
        flag as a sweep.

    Args:
        snapshots: Book snapshots (from reconstruct_book).
        min_levels: Minimum number of levels consumed to flag a sweep.
        max_time_ms: Maximum time window for the sweep (ms).
        tick_size: Tick size to compute level count.

    Returns DataFrame:
        ts_start, ts_end, side ('bid_sweep'|'ask_sweep'),
        levels_consumed, price_start, price_end, duration_ms
    """
    if len(snapshots) < 2:
        return pl.DataFrame(schema={
            "ts_start": pl.Int64, "ts_end": pl.Int64, "side": pl.Utf8,
            "levels_consumed": pl.Int64, "price_start": pl.Float64,
            "price_end": pl.Float64, "duration_ms": pl.Float64,
        })

    max_time_ns = max_time_ms * 1_000_000
    events = []

    # Track running sweep state
    # Ask sweep = best ask dropping rapidly (aggressive buyer eating asks)
    # Bid sweep = best bid rising rapidly (aggressive seller eating bids)
    # Actually: ask sweep = asks getting consumed, best ask moves UP
    # bid sweep = bids getting consumed, best bid moves DOWN

    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        dt_ns = curr.ts_event - prev.ts_event

        if dt_ns > max_time_ns or dt_ns <= 0:
            continue

        # ASK SWEEP: aggressive buyer pushes best ask up (consuming ask levels)
        if prev.best_ask is not None and curr.best_ask is not None:
            ask_move = curr.best_ask - prev.best_ask
            if ask_move > 0:
                levels = int(round(ask_move / tick_size))
                if levels >= min_levels:
                    events.append({
                        "ts_start": prev.ts_event,
                        "ts_end": curr.ts_event,
                        "side": "ask_sweep",
                        "levels_consumed": levels,
                        "price_start": prev.best_ask,
                        "price_end": curr.best_ask,
                        "duration_ms": dt_ns / 1_000_000,
                    })

        # BID SWEEP: aggressive seller pushes best bid down (consuming bid levels)
        if prev.best_bid is not None and curr.best_bid is not None:
            bid_move = prev.best_bid - curr.best_bid
            if bid_move > 0:
                levels = int(round(bid_move / tick_size))
                if levels >= min_levels:
                    events.append({
                        "ts_start": prev.ts_event,
                        "ts_end": curr.ts_event,
                        "side": "bid_sweep",
                        "levels_consumed": levels,
                        "price_start": prev.best_bid,
                        "price_end": curr.best_bid,
                        "duration_ms": dt_ns / 1_000_000,
                    })

    result = pl.DataFrame(events) if events else pl.DataFrame(schema={
        "ts_start": pl.Int64, "ts_end": pl.Int64, "side": pl.Utf8,
        "levels_consumed": pl.Int64, "price_start": pl.Float64,
        "price_end": pl.Float64, "duration_ms": pl.Float64,
    })

    n_ask = sum(1 for e in events if e["side"] == "ask_sweep") if events else 0
    n_bid = sum(1 for e in events if e["side"] == "bid_sweep") if events else 0
    logger.info(
        "Detected %d sweeps (%d ask, %d bid) — min %d levels in <%d ms",
        len(result), n_ask, n_bid, min_levels, max_time_ms,
    )
    return result


# ════════════════════════════════════════════════════════════════════════════
# 7. LARGE ORDER DETECTION (from book snapshots)
# ════════════════════════════════════════════════════════════════════════════


def detect_large_orders(
    snapshots: list[BookSnapshot],
    threshold_contracts: int = 100,
) -> pl.DataFrame:
    """Detect large resting orders (potential icebergs / institutional).

    Returns DataFrame: ts_event, price, size, side, count
    """
    ts_l, px_l, sz_l, sd_l, ct_l = [], [], [], [], []
    for snap in snapshots:
        for lv in snap.bids:
            if lv.size >= threshold_contracts:
                ts_l.append(snap.ts_event)
                px_l.append(lv.price)
                sz_l.append(lv.size)
                sd_l.append("bid")
                ct_l.append(lv.count)
        for lv in snap.asks:
            if lv.size >= threshold_contracts:
                ts_l.append(snap.ts_event)
                px_l.append(lv.price)
                sz_l.append(lv.size)
                sd_l.append("ask")
                ct_l.append(lv.count)

    df = pl.DataFrame({
        "ts_event": pl.Series("ts_event", ts_l, dtype=pl.Int64),
        "price": pl.Series("price", px_l, dtype=pl.Float64),
        "size": pl.Series("size", sz_l, dtype=pl.Int64),
        "side": pl.Series("side", sd_l, dtype=pl.Utf8),
        "count": pl.Series("count", ct_l, dtype=pl.Int64),
    })
    logger.info("Found %d large order observations (>= %d contracts)", len(df), threshold_contracts)
    return df
