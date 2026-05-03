"""Shared test fixtures — synthetic market data for unit tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_trades() -> pl.DataFrame:
    """Generate 1000 synthetic trades resembling ES futures."""
    np.random.seed(42)
    n = 1000
    base_price = 5500.0
    prices = base_price + np.cumsum(np.random.choice([-0.25, 0, 0.25], size=n, p=[0.3, 0.4, 0.3]))
    sizes = np.random.randint(1, 50, size=n)
    sides = np.random.choice(["A", "B"], size=n, p=[0.52, 0.48])  # slight buy bias
    ts = np.arange(n) * 1_000_000_000  # 1-second intervals in nanoseconds

    return pl.DataFrame({
        "ts_event": ts,
        "price": prices,
        "size": sizes,
        "side": sides,
    })


@pytest.fixture
def synthetic_bars() -> pl.DataFrame:
    """Generate 200 synthetic 15M OHLCV bars resembling ES futures."""
    np.random.seed(42)
    n = 200
    base = 5500.0
    close_prices = base + np.cumsum(np.random.normal(0, 2, size=n))

    opens = close_prices + np.random.normal(0, 1, size=n)
    highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, 3, size=n))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, 3, size=n))
    volumes = np.random.randint(500, 5000, size=n)
    ts = np.arange(n) * 15 * 60 * 1_000_000_000  # 15-min intervals in ns

    return pl.DataFrame({
        "ts_event": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close_prices,
        "volume": volumes,
    })


@pytest.fixture
def synthetic_mbp10() -> pl.DataFrame:
    """Generate synthetic MBP-10 snapshots with full schema.

    Includes header columns (ts_recv, price, size, action, side, flags,
    sequence) and per-level bid/ask count columns (bid_ct_*, ask_ct_*)
    required by parse_mbp10().
    """
    np.random.seed(42)
    n = 100
    base = 5500.0
    records = []

    for i in range(n):
        mid = base + np.random.normal(0, 2)
        row = {
            "ts_event": i * 1_000_000_000,
            "ts_recv": i * 1_000_000_000 + 500_000,  # 0.5 ms later
            "price": mid,
            "size": int(np.random.randint(1, 50)),
            "action": "T",
            "side": np.random.choice(["A", "B"]),
            "flags": 0,
            "sequence": i,
        }
        for j in range(10):
            row[f"bid_px_{j:02d}"] = mid - (j + 1) * 0.25
            row[f"bid_sz_{j:02d}"] = int(np.random.randint(10, 200))
            row[f"bid_ct_{j:02d}"] = int(np.random.randint(1, 20))
            row[f"ask_px_{j:02d}"] = mid + (j + 1) * 0.25
            row[f"ask_sz_{j:02d}"] = int(np.random.randint(10, 200))
            row[f"ask_ct_{j:02d}"] = int(np.random.randint(1, 20))
        records.append(row)

    return pl.DataFrame(records)


@pytest.fixture
def synthetic_mbp10_with_sweep() -> pl.DataFrame:
    """MBP-10 snapshots where the best ask jumps 4 levels between two
    consecutive snapshots (simulating an aggressive buyer sweep).
    """
    base = 5500.0
    records = []

    for i in range(5):
        # Normal snapshots: best ask at base + 0.25
        mid = base
        row = {
            "ts_event": i * 100_000_000,  # 100ms apart
            "ts_recv": i * 100_000_000 + 500_000,
            "price": mid,
            "size": 10,
            "action": "T",
            "side": "A",
            "flags": 0,
            "sequence": i,
        }
        for j in range(10):
            row[f"bid_px_{j:02d}"] = mid - (j + 1) * 0.25
            row[f"bid_sz_{j:02d}"] = 50
            row[f"bid_ct_{j:02d}"] = 5
            row[f"ask_px_{j:02d}"] = mid + (j + 1) * 0.25
            row[f"ask_sz_{j:02d}"] = 50
            row[f"ask_ct_{j:02d}"] = 5
        records.append(row)

    # Snapshot 5: best ask jumps up by 4 levels (1.0 point = 4 ticks)
    mid = base
    row = {
        "ts_event": 5 * 100_000_000,
        "ts_recv": 5 * 100_000_000 + 500_000,
        "price": mid + 1.0,
        "size": 200,
        "action": "T",
        "side": "A",
        "flags": 0,
        "sequence": 5,
    }
    for j in range(10):
        row[f"bid_px_{j:02d}"] = mid - (j + 1) * 0.25
        row[f"bid_sz_{j:02d}"] = 50
        row[f"bid_ct_{j:02d}"] = 5
        # Ask side shifted up by 4 levels (1.0 = 4 * 0.25)
        row[f"ask_px_{j:02d}"] = mid + 1.0 + (j + 1) * 0.25
        row[f"ask_sz_{j:02d}"] = 50
        row[f"ask_ct_{j:02d}"] = 5
    records.append(row)

    return pl.DataFrame(records)


@pytest.fixture
def synthetic_absorption_data():
    """Book snapshots + trades where executed volume >> visible size at a level.

    Returns (mbp10_df, trades_df) where trades at 5500.25 have volume 500
    but the book only shows ~20 resting at that level → absorption ratio ~25.
    """
    np.random.seed(42)
    base = 5500.0
    target_price = base + 0.25  # best ask level

    # Book: 5 snapshots, each showing ~20 contracts at target_price
    mbp_records = []
    for i in range(5):
        row = {
            "ts_event": i * 200_000_000,  # 200ms apart
            "ts_recv": i * 200_000_000 + 500_000,
            "price": base,
            "size": 10,
            "action": "T",
            "side": "A",
            "flags": 0,
            "sequence": i,
        }
        for j in range(10):
            row[f"bid_px_{j:02d}"] = base - (j + 1) * 0.25
            row[f"bid_sz_{j:02d}"] = 50
            row[f"bid_ct_{j:02d}"] = 5
            row[f"ask_px_{j:02d}"] = base + (j + 1) * 0.25
            row[f"ask_sz_{j:02d}"] = 20  # small visible size
            row[f"ask_ct_{j:02d}"] = 2
        mbp_records.append(row)
    mbp_df = pl.DataFrame(mbp_records)

    # Trades: 50 trades of size 10 = 500 total at target_price within 1 second
    trade_records = []
    for k in range(50):
        trade_records.append({
            "ts_event": k * 20_000_000,  # 20ms apart, all within 1 second
            "price": target_price,
            "size": 10,
            "side": "A",
            "trade_side": 1,
            "buy_volume": 10,
            "sell_volume": 0,
            "delta": 10,
        })
    trades_df = pl.DataFrame(trade_records)

    return mbp_df, trades_df
