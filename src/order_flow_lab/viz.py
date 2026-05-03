"""Bookmap-style heatmap and order flow visualization — Phase 3.

Uses HoloViews + Datashader + Bokeh for rendering large LOB datasets,
with trade bubble overlays, synchronized CVD panel, and strategy signal markers.

All plot functions return HoloViews elements that share a common X-axis
(ts_event in nanoseconds) so they can be linked in the Panel dashboard.
"""

from __future__ import annotations

import logging

import colorcet as cc
import datashader as ds
import holoviews as hv
import numpy as np
import panel as pn
import polars as pl
from holoviews.operation.datashader import rasterize

hv.extension("bokeh")
pn.extension()

logger = logging.getLogger(__name__)

# ── Shared plot defaults ────────────────────────────────────────────────
_DARK_BG = "#0d1117"
_GRID_COLOR = "#21262d"
_TEXT_COLOR = "#c9d1d9"
_BUY_COLOR = "#3fb950"
_SELL_COLOR = "#f85149"
_VWAP_COLOR = "#d2a8ff"
_TP_COLOR = "#58a6ff"
_SL_COLOR = "#f85149"


def _ts_to_seconds(ts_ns: np.ndarray) -> np.ndarray:
    """Convert nanosecond timestamps to seconds-from-start for cleaner axis."""
    t0 = ts_ns.min()
    return (ts_ns - t0) / 1e9


# ════════════════════════════════════════════════════════════════════════════
# 1. BOOKMAP-STYLE HEATMAP
# ════════════════════════════════════════════════════════════════════════════


def plot_book_heatmap(
    heatmap_df: pl.DataFrame,
    trades_df: pl.DataFrame | None = None,
    title: str = "Order Book Heatmap",
    width: int = 1200,
    height: int = 500,
    cmap: list | str = "fire",
) -> hv.Overlay:
    """Render a Bookmap-style heatmap of order book depth over time.

    X = time (seconds from session start)
    Y = price
    Intensity = aggregated resting size at each (time, price) cell

    Args:
        heatmap_df: DataFrame with columns: ts_event, price, size, side
                    (from book.snapshots_to_heatmap_df).
        trades_df: Optional trade DataFrame with ts_event, price, size, trade_side
                   for overlaying trade bubbles.
        title: Chart title.
        cmap: Colorcet colormap name or list of hex colors.

    Returns:
        HoloViews Overlay (heatmap + optional trade scatter).
    """
    pdf = heatmap_df.to_pandas().copy()
    pdf["t_sec"] = _ts_to_seconds(pdf["ts_event"].values)

    # Rasterized heatmap — datashader aggregates size per pixel bin
    points = hv.Points(pdf, kdims=["t_sec", "price"], vdims=["size"])

    heatmap = rasterize(
        points,
        aggregator=ds.sum("size"),
        width=width,
        height=height,
    ).opts(
        cmap=cmap,
        cnorm="log",              # log scale makes thin liquidity visible
        colorbar=True,
        colorbar_position="right",
        title=title,
        width=width,
        height=height,
        tools=["hover", "box_zoom", "reset", "crosshair"],
        xlabel="Time (s)",
        ylabel="Price",
        bgcolor=_DARK_BG,
        gridstyle={"grid_line_color": _GRID_COLOR},
        show_grid=True,
    )

    overlay = heatmap

    # ── Trade bubble overlay ────────────────────────────────────────
    if trades_df is not None and len(trades_df) > 0:
        tpdf = trades_df.to_pandas().copy()
        tpdf["t_sec"] = _ts_to_seconds(tpdf["ts_event"].values)

        # Size proportional to trade size (clamped 3–20 px)
        max_sz = tpdf["size"].max() if tpdf["size"].max() > 0 else 1
        tpdf["bubble_size"] = (tpdf["size"] / max_sz * 18).clip(lower=3, upper=20)

        # Color by aggressor side
        if "trade_side" in tpdf.columns:
            tpdf["color"] = tpdf["trade_side"].map(
                {1: _BUY_COLOR, -1: _SELL_COLOR, 0: "#8b949e"}
            )
        elif "side" in tpdf.columns:
            tpdf["color"] = tpdf["side"].map(
                {"A": _BUY_COLOR, "B": _SELL_COLOR}
            ).fillna("#8b949e")
        else:
            tpdf["color"] = "#8b949e"

        # Separate buy/sell for legend clarity
        buys = tpdf[tpdf.get("color", "") == _BUY_COLOR] if "color" in tpdf.columns else tpdf.iloc[:0]
        sells = tpdf[tpdf.get("color", "") == _SELL_COLOR] if "color" in tpdf.columns else tpdf.iloc[:0]

        if len(buys) > 0:
            overlay = overlay * hv.Points(
                buys, kdims=["t_sec", "price"],
                vdims=["size", "bubble_size"],
                label="Buy",
            ).opts(
                size="bubble_size", color=_BUY_COLOR, alpha=0.55,
                tools=["hover"], marker="circle",
            )
        if len(sells) > 0:
            overlay = overlay * hv.Points(
                sells, kdims=["t_sec", "price"],
                vdims=["size", "bubble_size"],
                label="Sell",
            ).opts(
                size="bubble_size", color=_SELL_COLOR, alpha=0.55,
                tools=["hover"], marker="circle",
            )

    return overlay


# ════════════════════════════════════════════════════════════════════════════
# 2. TRADE BUBBLES STANDALONE (for composing in layouts)
# ════════════════════════════════════════════════════════════════════════════


def plot_trade_bubbles(
    trades_df: pl.DataFrame,
    width: int = 1200,
    height: int = 500,
    title: str = "Trade Tape",
) -> hv.Overlay:
    """Scatter plot: trade bubbles colored by aggressor, sized by volume.

    Useful as a standalone chart or overlaid on the heatmap.
    """
    tpdf = trades_df.to_pandas().copy()
    tpdf["t_sec"] = _ts_to_seconds(tpdf["ts_event"].values)
    max_sz = tpdf["size"].max() if tpdf["size"].max() > 0 else 1
    tpdf["bubble_size"] = (tpdf["size"] / max_sz * 18).clip(lower=3, upper=20)

    if "trade_side" in tpdf.columns:
        buys = tpdf[tpdf["trade_side"] == 1]
        sells = tpdf[tpdf["trade_side"] == -1]
    elif "side" in tpdf.columns:
        buys = tpdf[tpdf["side"] == "A"]
        sells = tpdf[tpdf["side"] == "B"]
    else:
        buys = tpdf
        sells = tpdf.iloc[:0]

    layers = []
    if len(buys) > 0:
        layers.append(
            hv.Points(buys, kdims=["t_sec", "price"], vdims=["size", "bubble_size"], label="Buy")
            .opts(size="bubble_size", color=_BUY_COLOR, alpha=0.6, marker="circle")
        )
    if len(sells) > 0:
        layers.append(
            hv.Points(sells, kdims=["t_sec", "price"], vdims=["size", "bubble_size"], label="Sell")
            .opts(size="bubble_size", color=_SELL_COLOR, alpha=0.6, marker="circle")
        )

    overlay = hv.Overlay(layers) if layers else hv.Points([])
    return overlay.opts(
        title=title, width=width, height=height,
        bgcolor=_DARK_BG, xlabel="Time (s)", ylabel="Price",
        tools=["hover", "box_zoom", "reset"], show_legend=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. CVD CHART (synchronized X-axis)
# ════════════════════════════════════════════════════════════════════════════


def plot_cvd(
    trades: pl.DataFrame,
    title: str = "Cumulative Volume Delta",
    width: int = 1200,
    height: int = 200,
) -> hv.Overlay:
    """Plot CVD as an area chart with positive/negative fill coloring.

    X-axis in seconds-from-start to synchronize with the heatmap above.
    """
    pdf = trades.select(["ts_event", "cvd"]).to_pandas().copy()
    pdf["t_sec"] = _ts_to_seconds(pdf["ts_event"].values)

    cvd_vals = pdf["cvd"].values
    zero_line = hv.HLine(0).opts(color="#484f58", line_width=1, line_dash="dashed")

    # Main CVD line
    cvd_curve = hv.Curve(
        pdf[["t_sec", "cvd"]], kdims=["t_sec"], vdims=["cvd"]
    ).opts(
        color="#FFA657", line_width=1.5,
    )

    # Positive area (above zero) and negative area (below zero)
    pos_area = hv.Area(
        pdf[["t_sec", "cvd"]].assign(cvd=np.maximum(cvd_vals, 0)),
        kdims=["t_sec"], vdims=["cvd"],
    ).opts(color=_BUY_COLOR, alpha=0.15)

    neg_area = hv.Area(
        pdf[["t_sec", "cvd"]].assign(cvd=np.minimum(cvd_vals, 0)),
        kdims=["t_sec"], vdims=["cvd"],
    ).opts(color=_SELL_COLOR, alpha=0.15)

    return (pos_area * neg_area * zero_line * cvd_curve).opts(
        title=title, width=width, height=height,
        bgcolor=_DARK_BG, xlabel="Time (s)", ylabel="CVD",
        gridstyle={"grid_line_color": _GRID_COLOR}, show_grid=True,
        tools=["crosshair", "box_zoom", "reset"],
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. VOLUME PROFILE (HORIZONTAL BARS)
# ════════════════════════════════════════════════════════════════════════════


def plot_volume_profile(
    profile_df: pl.DataFrame,
    title: str = "Volume Profile",
    width: int = 280,
    height: int = 500,
) -> hv.Overlay:
    """Render volume profile as mirrored horizontal bars with B/S coloring.

    Buy volume extends right (green), sell volume extends left (red).
    """
    pdf = profile_df.to_pandas()

    buy_bars = hv.Bars(
        pdf, kdims=["price_level"], vdims=["buy_volume"]
    ).opts(
        color=_BUY_COLOR, alpha=0.75, invert_axes=True,
        width=width, height=height,
    )

    sell_bars = hv.Bars(
        pdf, kdims=["price_level"], vdims=["sell_volume"]
    ).opts(
        color=_SELL_COLOR, alpha=0.75, invert_axes=True,
        width=width, height=height,
    )

    # High volume nodes (HVN) markers — levels with volume_ratio >= 2.0
    hvn = pdf[pdf["volume_ratio"] >= 2.0]
    hvn_markers = hv.Points(
        hvn, kdims=["total_volume", "price_level"],
    ).opts(
        color="#FFA657", size=6, marker="diamond",
    ) if len(hvn) > 0 else hv.Points([])

    return (buy_bars * sell_bars * hvn_markers).opts(
        title=title, bgcolor=_DARK_BG,
        xlabel="Volume", ylabel="Price",
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. CANDLESTICK + VWAP + SIGNAL MARKERS
# ════════════════════════════════════════════════════════════════════════════


def plot_candlestick_vwap(
    bars: pl.DataFrame,
    signals: list | None = None,
    backtest_trades: list | None = None,
    title: str = "Price + VWAP Bands",
    width: int = 1200,
    height: int = 500,
) -> hv.Overlay:
    """Candlestick chart with VWAP bands, sweep dots, and signal entry/exit markers.

    Args:
        bars: OHLCV bars with vwap, vwap_upper, vwap_lower, sweep_high, sweep_low.
        signals: Optional list of Signal objects for entry markers.
        backtest_trades: Optional list of Trade objects for exit markers.
    """
    pdf = bars.to_pandas().reset_index(drop=True)
    pdf["idx"] = range(len(pdf))

    # ── Candle bodies + wicks ─────────────────────────────────────
    green = pdf[pdf["close"] >= pdf["open"]]
    red = pdf[pdf["close"] < pdf["open"]]

    wicks_up = hv.Segments(
        green, kdims=["idx", "low", "idx", "high"]
    ).opts(color="#3fb950", line_width=1)

    wicks_down = hv.Segments(
        red, kdims=["idx", "low", "idx", "high"]
    ).opts(color="#f85149", line_width=1)

    bodies_up = hv.Rectangles(
        [(r.idx - 0.35, r.open, r.idx + 0.35, r.close) for r in green.itertuples()]
    ).opts(color="#3fb950", alpha=0.85, line_color="#3fb950")

    bodies_down = hv.Rectangles(
        [(r.idx - 0.35, r.close, r.idx + 0.35, r.open) for r in red.itertuples()]
    ).opts(color="#f85149", alpha=0.85, line_color="#f85149")

    chart = wicks_up * wicks_down * bodies_up * bodies_down

    # ── VWAP lines ────────────────────────────────────────────────
    if "vwap" in pdf.columns:
        chart = chart * hv.Curve(
            pdf[["idx", "vwap"]], kdims=["idx"], vdims=["vwap"]
        ).opts(color=_VWAP_COLOR, line_width=2, line_dash="solid")

    if "vwap_upper" in pdf.columns:
        chart = chart * hv.Curve(
            pdf[["idx", "vwap_upper"]], kdims=["idx"], vdims=["vwap_upper"]
        ).opts(color="#f0883e", line_width=1, line_dash="dashed")

    if "vwap_lower" in pdf.columns:
        chart = chart * hv.Curve(
            pdf[["idx", "vwap_lower"]], kdims=["idx"], vdims=["vwap_lower"]
        ).opts(color="#58a6ff", line_width=1, line_dash="dashed")

    # ── Sweep dots ────────────────────────────────────────────────
    if "sweep_high" in pdf.columns:
        sh = pdf[pdf["sweep_high"] == True]
        if len(sh) > 0:
            chart = chart * hv.Points(
                sh, kdims=["idx", "high"], label="Sweep High"
            ).opts(color=_SELL_COLOR, size=9, marker="inverted_triangle", alpha=0.9)

    if "sweep_low" in pdf.columns:
        sl = pdf[pdf["sweep_low"] == True]
        if len(sl) > 0:
            chart = chart * hv.Points(
                sl, kdims=["idx", "low"], label="Sweep Low"
            ).opts(color=_BUY_COLOR, size=9, marker="triangle", alpha=0.9)

    # ── Strategy entry markers ────────────────────────────────────
    if signals:
        from .strategy import Side

        long_entries = [(s.bar_idx, s.entry_price) for s in signals if s.side == Side.LONG]
        short_entries = [(s.bar_idx, s.entry_price) for s in signals if s.side == Side.SHORT]

        if long_entries:
            chart = chart * hv.Points(
                long_entries, kdims=["x", "y"], label="LONG Entry"
            ).opts(
                color=_BUY_COLOR, size=14, marker="triangle",
                line_color="white", line_width=1.5,
            )
        if short_entries:
            chart = chart * hv.Points(
                short_entries, kdims=["x", "y"], label="SHORT Entry"
            ).opts(
                color=_SELL_COLOR, size=14, marker="inverted_triangle",
                line_color="white", line_width=1.5,
            )

    # ── Backtest exit markers (with TP/SL distinction) ────────────
    if backtest_trades:
        from .strategy import Side

        tp_exits, sl_exits, time_exits = [], [], []
        for t in backtest_trades:
            pt = (t.exit_bar, t.exit_price)
            if t.exit_reason in ("tp1", "tp2"):
                tp_exits.append(pt)
            elif t.exit_reason == "stop_loss":
                sl_exits.append(pt)
            else:
                time_exits.append(pt)

        if tp_exits:
            chart = chart * hv.Points(
                tp_exits, kdims=["x", "y"], label="TP Exit"
            ).opts(color=_TP_COLOR, size=10, marker="star", line_color="white")
        if sl_exits:
            chart = chart * hv.Points(
                sl_exits, kdims=["x", "y"], label="SL Exit"
            ).opts(color=_SL_COLOR, size=10, marker="x", line_width=2)
        if time_exits:
            chart = chart * hv.Points(
                time_exits, kdims=["x", "y"], label="Time Exit"
            ).opts(color="#8b949e", size=8, marker="square", alpha=0.7)

        # ── Entry-to-exit connector lines ─────────────────────────
        for t in backtest_trades:
            line_color = _BUY_COLOR if t.signal.side == Side.LONG else _SELL_COLOR
            chart = chart * hv.Curve(
                [(t.entry_bar, t.entry_price), (t.exit_bar, t.exit_price)],
                kdims=["x"], vdims=["y"],
            ).opts(color=line_color, line_width=1, line_dash="dotted", alpha=0.5)

    return chart.opts(
        title=title, width=width, height=height,
        bgcolor=_DARK_BG, xlabel="Bar #", ylabel="Price",
        gridstyle={"grid_line_color": _GRID_COLOR}, show_grid=True,
        tools=["hover", "box_zoom", "reset", "crosshair"],
        show_legend=True, legend_position="top_left",
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. EQUITY CURVE
# ════════════════════════════════════════════════════════════════════════════


def plot_equity_curve(
    trades: list,
    point_value: float = 50.0,
    title: str = "Equity Curve",
    width: int = 1200,
    height: int = 250,
) -> hv.Overlay:
    """Plot cumulative PnL from completed trades with drawdown fill."""
    if not trades:
        return hv.Curve([]).opts(title=title, width=width, height=height)

    pnls = np.array([t.pnl_points * point_value for t in trades])
    cum_pnl = np.cumsum(pnls)
    indices = np.arange(len(cum_pnl))

    # Drawdown band
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max  # always <= 0

    dd_area = hv.Area(
        list(zip(indices, drawdown)), kdims=["Trade #"], vdims=["Drawdown ($)"]
    ).opts(color=_SELL_COLOR, alpha=0.2)

    equity = hv.Curve(
        list(zip(indices, cum_pnl)), kdims=["Trade #"], vdims=["PnL ($)"]
    ).opts(color=_BUY_COLOR, line_width=2)

    zero = hv.HLine(0).opts(color="#484f58", line_width=1, line_dash="dashed")

    return (dd_area * zero * equity).opts(
        title=title, width=width, height=height,
        bgcolor=_DARK_BG, xlabel="Trade #", ylabel="Cumulative PnL ($)",
        gridstyle={"grid_line_color": _GRID_COLOR}, show_grid=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. BOOK IMBALANCE STRIP
# ════════════════════════════════════════════════════════════════════════════


def plot_imbalance_strip(
    summary_df: pl.DataFrame,
    title: str = "Book Imbalance",
    width: int = 1200,
    height: int = 120,
) -> hv.Overlay:
    """Horizontal strip showing book imbalance over time.

    Green = bid-heavy, Red = ask-heavy. Useful below heatmap for context.
    """
    pdf = summary_df.to_pandas().copy()
    pdf["t_sec"] = _ts_to_seconds(pdf["ts_event"].values)

    imb_curve = hv.Curve(
        pdf[["t_sec", "book_imbalance"]], kdims=["t_sec"], vdims=["book_imbalance"]
    ).opts(color="#FFA657", line_width=1)

    pos_fill = hv.Area(
        pdf[["t_sec", "book_imbalance"]].assign(
            book_imbalance=np.maximum(pdf["book_imbalance"].values, 0)
        ),
        kdims=["t_sec"], vdims=["book_imbalance"],
    ).opts(color=_BUY_COLOR, alpha=0.2)

    neg_fill = hv.Area(
        pdf[["t_sec", "book_imbalance"]].assign(
            book_imbalance=np.minimum(pdf["book_imbalance"].values, 0)
        ),
        kdims=["t_sec"], vdims=["book_imbalance"],
    ).opts(color=_SELL_COLOR, alpha=0.2)

    zero = hv.HLine(0).opts(color="#484f58", line_width=1, line_dash="dashed")

    return (pos_fill * neg_fill * zero * imb_curve).opts(
        title=title, width=width, height=height,
        bgcolor=_DARK_BG, xlabel="Time (s)", ylabel="Imbalance",
        gridstyle={"grid_line_color": _GRID_COLOR}, show_grid=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 8. ABSORPTION / SWEEP EVENT MARKERS (overlay on heatmap)
# ════════════════════════════════════════════════════════════════════════════


def absorption_markers(
    absorption_df: pl.DataFrame,
    t0_ns: int = 0,
) -> hv.Points:
    """Return HoloViews Points for absorption events to overlay on heatmap."""
    if absorption_df.is_empty():
        return hv.Points([])

    pdf = absorption_df.to_pandas().copy()
    # Use ts_window or ts_start depending on which column exists
    ts_col = "ts_window" if "ts_window" in pdf.columns else "ts_start"
    pdf["t_sec"] = (pdf[ts_col] - t0_ns) / 1e9 if t0_ns > 0 else pdf[ts_col] / 1e9

    return hv.Points(
        pdf, kdims=["t_sec", "price_level"],
        vdims=["absorption_ratio"] if "absorption_ratio" in pdf.columns else [],
        label="Absorption",
    ).opts(
        color="#FFA657", size=12, marker="hex", alpha=0.8,
        line_color="white", line_width=1,
    )


def sweep_markers(
    sweep_df: pl.DataFrame,
    t0_ns: int = 0,
) -> hv.Points:
    """Return HoloViews Points for sweep events to overlay on heatmap."""
    if sweep_df.is_empty():
        return hv.Points([])

    pdf = sweep_df.to_pandas().copy()
    pdf["t_sec"] = (pdf["ts_start"] - t0_ns) / 1e9 if t0_ns > 0 else pdf["ts_start"] / 1e9

    ask_sweeps = pdf[pdf["side"] == "ask_sweep"]
    bid_sweeps = pdf[pdf["side"] == "bid_sweep"]

    layers = []
    if len(ask_sweeps) > 0:
        layers.append(
            hv.Points(
                ask_sweeps, kdims=["t_sec", "price_start"],
                vdims=["levels_consumed"], label="Ask Sweep",
            ).opts(color=_BUY_COLOR, size=10, marker="triangle", alpha=0.9)
        )
    if len(bid_sweeps) > 0:
        layers.append(
            hv.Points(
                bid_sweeps, kdims=["t_sec", "price_start"],
                vdims=["levels_consumed"], label="Bid Sweep",
            ).opts(color=_SELL_COLOR, size=10, marker="inverted_triangle", alpha=0.9)
        )

    return hv.Overlay(layers) if layers else hv.Points([])
