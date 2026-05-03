"""Feature engineering for the VWAP Deviation Stop-Hunt Reversal strategy.

Computes from trade-level data:
    - Session VWAP with +/- 2 sigma bands
    - Volume delta (bid vs ask classification)
    - CVD (Cumulative Volume Delta)
    - Volume profile with relative multipliers
    - Pivot highs/lows with sweep detection
    - Absorption / iceberg inference
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. VWAP WITH DEVIATION BANDS
# ════════════════════════════════════════════════════════════════════════════


def compute_vwap(
    trades: pl.DataFrame,
    price_col: str = "price",
    size_col: str = "size",
    ts_col: str = "ts_event",
    num_deviations: float = 2.0,
) -> pl.DataFrame:
    """Compute session VWAP with standard deviation bands.

    Returns the input DataFrame with added columns:
        vwap, vwap_upper, vwap_lower, vwap_dev
    """
    # Cumulative volume-weighted price
    df = trades.sort(ts_col).with_columns(
        (pl.col(price_col) * pl.col(size_col)).cum_sum().alias("_cum_pv"),
        pl.col(size_col).cum_sum().alias("_cum_vol"),
    ).with_columns(
        (pl.col("_cum_pv") / pl.col("_cum_vol")).alias("vwap"),
    )

    # Rolling variance for deviation bands
    # Var = E[P^2] - E[P]^2, volume-weighted
    df = df.with_columns(
        (pl.col(price_col).pow(2) * pl.col(size_col)).cum_sum().alias("_cum_p2v"),
    ).with_columns(
        (
            (pl.col("_cum_p2v") / pl.col("_cum_vol"))
            - pl.col("vwap").pow(2)
        )
        .clip(lower_bound=0)
        .sqrt()
        .alias("vwap_dev"),
    ).with_columns(
        (pl.col("vwap") + num_deviations * pl.col("vwap_dev")).alias("vwap_upper"),
        (pl.col("vwap") - num_deviations * pl.col("vwap_dev")).alias("vwap_lower"),
    ).drop("_cum_pv", "_cum_vol", "_cum_p2v")

    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. VOLUME DELTA & CVD
# ════════════════════════════════════════════════════════════════════════════


def classify_trades(
    trades: pl.DataFrame,
    price_col: str = "price",
    size_col: str = "size",
) -> pl.DataFrame:
    """Classify trades as buyer- or seller-initiated using tick rule.

    If the trade's 'side' column exists in the data (Databento provides it),
    use that directly. Otherwise, fall back to the tick rule (compare to
    previous trade price).

    Returns DataFrame with added columns:
        trade_side  (1 = buy, -1 = sell)
        buy_volume  (size if buy, else 0)
        sell_volume (size if sell, else 0)
        delta       (buy_volume - sell_volume for this trade)
    """
    if "side" in trades.columns:
        # Databento side: 'A' (ask side = buy aggressor), 'B' (bid side = sell aggressor)
        df = trades.with_columns(
            pl.when(pl.col("side") == "A")
            .then(pl.lit(1))
            .when(pl.col("side") == "B")
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("trade_side")
        )
    else:
        # Tick rule fallback
        df = trades.with_columns(
            pl.when(pl.col(price_col) > pl.col(price_col).shift(1))
            .then(pl.lit(1))
            .when(pl.col(price_col) < pl.col(price_col).shift(1))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("trade_side")
        )

    df = df.with_columns(
        pl.when(pl.col("trade_side") == 1)
        .then(pl.col(size_col))
        .otherwise(pl.lit(0))
        .alias("buy_volume"),
        pl.when(pl.col("trade_side") == -1)
        .then(pl.col(size_col))
        .otherwise(pl.lit(0))
        .alias("sell_volume"),
    ).with_columns(
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"),
    )

    return df


def compute_cvd(trades: pl.DataFrame) -> pl.DataFrame:
    """Compute Cumulative Volume Delta from classified trades.

    Requires: trades already processed by classify_trades().

    Returns DataFrame with added column: cvd
    """
    return trades.with_columns(
        pl.col("delta").cum_sum().alias("cvd"),
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. VOLUME PROFILE WITH MULTIPLIERS
# ════════════════════════════════════════════════════════════════════════════


def compute_volume_profile(
    trades: pl.DataFrame,
    tick_size: float = 0.25,
    price_col: str = "price",
    size_col: str = "size",
) -> pl.DataFrame:
    """Compute session volume profile: volume at each price level.

    Returns DataFrame with columns:
        price_level, total_volume, buy_volume, sell_volume,
        volume_ratio (vs session average), buyer_seller_flag ('B' or 'S')
    """
    # Build profile grouped by tick-rounded price levels
    profile = (
        trades.with_columns(
            ((pl.col(price_col) / tick_size).round(0) * tick_size).alias("price_level")
        )
        .group_by("price_level")
        .agg(
            pl.col(size_col).sum().alias("total_volume"),
            pl.col("buy_volume").sum().alias("buy_volume"),
            pl.col("sell_volume").sum().alias("sell_volume"),
        )
        .sort("price_level")
    )

    # Compute multiplier vs average
    avg_vol = profile["total_volume"].mean()
    profile = profile.with_columns(
        (pl.col("total_volume") / avg_vol).round(1).alias("volume_ratio"),
        pl.when(pl.col("buy_volume") > pl.col("sell_volume"))
        .then(pl.lit("B"))
        .otherwise(pl.lit("S"))
        .alias("buyer_seller_flag"),
    )

    return profile


def compute_rolling_volume_profile(
    trades: pl.DataFrame,
    bar_df: pl.DataFrame,
    tick_size: float = 0.25,
    price_col: str = "price",
    size_col: str = "size",
    ts_col: str = "ts_event",
) -> pl.DataFrame:
    """Compute volume profile per bar (for footprint chart).

    Args:
        trades: Classified trades DataFrame.
        bar_df: OHLCV bars with 'ts_open' and 'ts_close' columns.
        tick_size: Tick size for price bucketing.

    Returns DataFrame with columns:
        bar_idx, price_level, total_volume, buy_volume, sell_volume,
        volume_ratio, buyer_seller_flag
    """
    results = []
    for i, bar in enumerate(bar_df.iter_rows(named=True)):
        t_start = bar.get("ts_open", bar.get(ts_col))
        t_end = bar.get("ts_close", None)
        if t_end is None:
            continue

        bar_trades = trades.filter(
            (pl.col(ts_col) >= t_start) & (pl.col(ts_col) < t_end)
        )
        if len(bar_trades) == 0:
            continue

        profile = compute_volume_profile(bar_trades, tick_size, price_col, size_col)
        profile = profile.with_columns(pl.lit(i).alias("bar_idx"))
        results.append(profile)

    if not results:
        return pl.DataFrame()
    return pl.concat(results)


# ════════════════════════════════════════════════════════════════════════════
# 4. PIVOT HIGHS/LOWS WITH SWEEP DETECTION
# ════════════════════════════════════════════════════════════════════════════


def detect_pivots(
    bars: pl.DataFrame,
    lookback: int = 20,
    high_col: str = "high",
    low_col: str = "low",
) -> pl.DataFrame:
    """Detect pivot highs and lows in OHLCV bars.

    A pivot high is a bar whose high is the max of the surrounding `lookback`
    bars on each side. Similarly for pivot low.

    Returns bars with added columns:
        is_pivot_high (bool), is_pivot_low (bool),
        pivot_high_level (float, forward-filled), pivot_low_level (float, forward-filled)
    """
    n = len(bars)
    highs = bars[high_col].to_numpy()
    lows = bars[low_col].to_numpy()

    pivot_high = np.full(n, False)
    pivot_low = np.full(n, False)

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        if highs[i] == window_high.max():
            pivot_high[i] = True

        window_low = lows[i - lookback : i + lookback + 1]
        if lows[i] == window_low.min():
            pivot_low[i] = True

    # Forward-fill the most recent pivot levels
    ph_levels = np.full(n, np.nan)
    pl_levels = np.full(n, np.nan)
    for i in range(n):
        if pivot_high[i]:
            ph_levels[i] = highs[i]
        elif i > 0:
            ph_levels[i] = ph_levels[i - 1]

        if pivot_low[i]:
            pl_levels[i] = lows[i]
        elif i > 0:
            pl_levels[i] = pl_levels[i - 1]

    result = bars.with_columns(
        pl.Series("is_pivot_high", pivot_high),
        pl.Series("is_pivot_low", pivot_low),
        pl.Series("pivot_high_level", ph_levels),
        pl.Series("pivot_low_level", pl_levels),
    )

    n_ph = int(pivot_high.sum())
    n_pl = int(pivot_low.sum())
    logger.info("Detected %d pivot highs, %d pivot lows (lookback=%d)", n_ph, n_pl, lookback)
    return result


def detect_sweeps(
    bars: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Detect sweep events — when price breaks a pivot level then closes back.

    A sweep HIGH occurs when:
        bar_high > pivot_high_level AND bar_close < pivot_high_level
    (price exceeded the pivot high but closed below it — stop-run above)

    A sweep LOW occurs when:
        bar_low < pivot_low_level AND bar_close > pivot_low_level
    (price exceeded the pivot low but closed above it — stop-run below)

    Returns bars with added columns:
        sweep_high (bool), sweep_low (bool)
    """
    return bars.with_columns(
        (
            (pl.col(high_col) > pl.col("pivot_high_level"))
            & (pl.col(close_col) < pl.col("pivot_high_level"))
        ).alias("sweep_high"),
        (
            (pl.col(low_col) < pl.col("pivot_low_level"))
            & (pl.col(close_col) > pl.col("pivot_low_level"))
        ).alias("sweep_low"),
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. ABSORPTION DETECTION
# ════════════════════════════════════════════════════════════════════════════


def detect_absorption(
    trades: pl.DataFrame,
    window_seconds: float = 60.0,
    volume_threshold_mult: float = 3.0,
    price_move_threshold: float = 2.0,
    tick_size: float = 0.25,
    ts_col: str = "ts_event",
    price_col: str = "price",
    size_col: str = "size",
) -> pl.DataFrame:
    """Detect absorption events: high volume with little price movement.

    Absorption = institutional participant absorbing aggressive orders without
    letting price move. Signature: volume >> average but price range << average.

    Args:
        window_seconds: Rolling window in seconds for volume/price aggregation.
        volume_threshold_mult: Volume must be >= this multiple of session average.
        price_move_threshold: Price range must be <= this many ticks.

    Returns DataFrame with absorption events:
        ts_start, ts_end, price_level, total_volume, price_range_ticks,
        net_delta, absorption_side ('bid' or 'ask')
    """
    # Convert timestamps to seconds for windowing
    df = trades.sort(ts_col)

    if df.is_empty():
        return pl.DataFrame(
            schema={
                "ts_start": pl.Int64,
                "ts_end": pl.Int64,
                "price_level": pl.Float64,
                "total_volume": pl.Int64,
                "price_range_ticks": pl.Float64,
                "net_delta": pl.Int64,
                "absorption_side": pl.Utf8,
            }
        )

    # Aggregate into time windows
    ts_arr = df[ts_col].to_numpy().astype(np.int64)
    prices = df[price_col].to_numpy().astype(np.float64)
    sizes = df[size_col].to_numpy().astype(np.int64)
    deltas = df["delta"].to_numpy().astype(np.int64) if "delta" in df.columns else np.zeros_like(sizes)

    window_ns = int(window_seconds * 1e9)
    session_avg_vol = sizes.sum() / max(1, (ts_arr[-1] - ts_arr[0]) / window_ns)

    events = []
    i = 0
    while i < len(ts_arr):
        j = i
        while j < len(ts_arr) and (ts_arr[j] - ts_arr[i]) < window_ns:
            j += 1

        window_vol = int(sizes[i:j].sum())
        window_delta = int(deltas[i:j].sum())
        price_range = float(prices[i:j].max() - prices[i:j].min())
        price_range_ticks = price_range / tick_size

        if (
            window_vol >= volume_threshold_mult * session_avg_vol
            and price_range_ticks <= price_move_threshold
        ):
            events.append(
                {
                    "ts_start": int(ts_arr[i]),
                    "ts_end": int(ts_arr[j - 1]),
                    "price_level": float(np.median(prices[i:j])),
                    "total_volume": window_vol,
                    "price_range_ticks": price_range_ticks,
                    "net_delta": window_delta,
                    "absorption_side": "bid" if window_delta > 0 else "ask",
                }
            )

        i = j if j > i else i + 1

    result = pl.DataFrame(events) if events else pl.DataFrame(
        schema={
            "ts_start": pl.Int64,
            "ts_end": pl.Int64,
            "price_level": pl.Float64,
            "total_volume": pl.Int64,
            "price_range_ticks": pl.Float64,
            "net_delta": pl.Int64,
            "absorption_side": pl.Utf8,
        }
    )
    logger.info("Detected %d absorption events", len(result))
    return result


# ════════════════════════════════════════════════════════════════════════════
# 6. ENGULFING CANDLE DETECTION
# ════════════════════════════════════════════════════════════════════════════


def detect_engulfing(
    bars: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Detect bullish and bearish engulfing candles.

    Bullish engulfing: previous bar bearish (close < open), current bar bullish
    (close > open) with body fully engulfing previous bar's body.

    Returns bars with added columns:
        bullish_engulfing (bool), bearish_engulfing (bool)
    """
    prev_open = pl.col(open_col).shift(1)
    prev_close = pl.col(close_col).shift(1)

    return bars.with_columns(
        # Bullish engulfing
        (
            (prev_close < prev_open)  # prev bearish
            & (pl.col(close_col) > pl.col(open_col))  # current bullish
            & (pl.col(open_col) <= prev_close)  # open <= prev close
            & (pl.col(close_col) >= prev_open)  # close >= prev open
        ).alias("bullish_engulfing"),
        # Bearish engulfing
        (
            (prev_close > prev_open)  # prev bullish
            & (pl.col(close_col) < pl.col(open_col))  # current bearish
            & (pl.col(open_col) >= prev_close)  # open >= prev close
            & (pl.col(close_col) <= prev_open)  # close <= prev open
        ).alias("bearish_engulfing"),
    )
