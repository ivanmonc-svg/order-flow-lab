"""Panel dashboard — Bookmap-style Order Flow Analyzer (Phase 3).

Run with:
    panel serve dashboard.py --show --autoreload

Or from Python:
    python dashboard.py

Tabs:
    1. Bookmap   — LOB heatmap + trade bubbles + CVD (synced X-axis)
    2. Price     — Candlestick + VWAP bands + signal entry/exit markers
    3. Profile   — Session volume profile + book imbalance strip
    4. Backtest  — Equity curve + trade log table + metrics sidebar
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import panel as pn

pn.extension("tabulator", sizing_mode="stretch_width")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR WIDGETS
# ══════════════════════════════════════════════════════════════════════════

symbol_select = pn.widgets.Select(
    name="Symbol", options=["ES", "NQ"], value="ES",
)
date_picker = pn.widgets.DatePicker(
    name="Date", value=date(2026, 4, 28),
)
lookback_slider = pn.widgets.IntSlider(
    name="Pivot Lookback", start=5, end=50, step=5, value=20,
)
vwap_dev_slider = pn.widgets.FloatSlider(
    name="VWAP Deviations (sigma)", start=1.0, end=3.0, step=0.25, value=2.0,
)
vol_ratio_slider = pn.widgets.FloatSlider(
    name="Min Volume Ratio", start=1.0, end=5.0, step=0.25, value=1.5,
)
start_hour_slider = pn.widgets.IntSlider(
    name="Start Hour (UTC)", start=0, end=23, step=1, value=14,
)
end_hour_slider = pn.widgets.IntSlider(
    name="End Hour (UTC)", start=1, end=24, step=1, value=15,
)
snapshot_freq_slider = pn.widgets.IntSlider(
    name="Book Snapshot (ms)", start=50, end=1000, step=50, value=100,
)
load_button = pn.widgets.Button(name="Load & Analyze", button_type="primary")

# ══════════════════════════════════════════════════════════════════════════
# DISPLAY PANES
# ══════════════════════════════════════════════════════════════════════════

status_pane = pn.pane.Markdown(
    "**Status:** Ready. Configure parameters and click **Load & Analyze**.",
)
# Tab 1 — Bookmap
heatmap_pane = pn.pane.HoloViews(None, sizing_mode="stretch_width")
cvd_pane = pn.pane.HoloViews(None, sizing_mode="stretch_width")
imbalance_pane = pn.pane.HoloViews(None, sizing_mode="stretch_width")
# Tab 2 — Price
chart_pane = pn.pane.HoloViews(None, sizing_mode="stretch_width")
# Tab 3 — Profile
profile_pane = pn.pane.HoloViews(None)
# Tab 4 — Backtest
equity_pane = pn.pane.HoloViews(None, sizing_mode="stretch_width")
trades_pane = pn.pane.DataFrame(None, sizing_mode="stretch_width")
metrics_pane = pn.pane.Markdown("")


# ══════════════════════════════════════════════════════════════════════════
# LOAD CALLBACK
# ══════════════════════════════════════════════════════════════════════════


def on_load(event):
    """Load data → compute features → generate signals → backtest → render."""
    # Lazy imports so the dashboard starts instantly
    from order_flow_lab.backtest import BacktestEngine, trades_to_dataframe
    from order_flow_lab.book import (
        reconstruct_book,
        snapshots_to_heatmap_df,
        snapshots_to_summary_df,
        detect_absorption_from_book,
        detect_sweeps_from_book,
    )
    from order_flow_lab.data_loader import DataLoader, POINT_VALUE
    from order_flow_lab.features import (
        classify_trades,
        compute_cvd,
        compute_volume_profile,
    )
    from order_flow_lab.strategy import StrategyConfig, generate_signals, prepare_bars
    from order_flow_lab.viz import (
        plot_book_heatmap,
        plot_candlestick_vwap,
        plot_cvd,
        plot_equity_curve,
        plot_imbalance_strip,
        plot_volume_profile,
        absorption_markers,
        sweep_markers,
    )

    sym = symbol_select.value
    dt = date_picker.value
    sh = start_hour_slider.value
    eh = end_hour_slider.value

    try:
        # ── 1. Load raw data ──────────────────────────────────────
        status_pane.object = f"**Status:** Loading {sym} data for {dt} ({sh}:00–{eh}:00 UTC)..."
        loader = DataLoader(data_dir="./data")

        status_pane.object = "**Status:** Downloading OHLCV bars..."
        bars_raw = loader.get_ohlcv(sym, dt, interval="1m", start_hour=sh, end_hour=eh)

        status_pane.object = "**Status:** Loading trades..."
        trades = loader.get_trades(sym, dt, start_hour=sh, end_hour=eh)

        status_pane.object = "**Status:** Loading MBP-10 book data..."
        mbp = loader.get_mbp(sym, dt, depth=10, start_hour=sh, end_hour=eh)

        # ── 2. Trade features ─────────────────────────────────────
        status_pane.object = "**Status:** Classifying trades (buy/sell)..."
        trades = classify_trades(trades)
        trades = compute_cvd(trades)

        # ── 3. Volume profile ─────────────────────────────────────
        status_pane.object = "**Status:** Computing volume profile..."
        profile = compute_volume_profile(trades, tick_size=0.25)

        # ── 4. Book reconstruction ────────────────────────────────
        status_pane.object = "**Status:** Reconstructing order book..."
        freq_ms = snapshot_freq_slider.value
        snapshots = reconstruct_book(mbp, snapshot_freq_ms=freq_ms)
        heatmap_df = snapshots_to_heatmap_df(snapshots)
        summary_df = snapshots_to_summary_df(snapshots)

        # ── 5. Microstructure features ────────────────────────────
        status_pane.object = "**Status:** Detecting absorption & sweeps..."
        absorption = detect_absorption_from_book(snapshots, trades)
        sweeps = detect_sweeps_from_book(snapshots)

        # ── 6. Strategy features on bars ──────────────────────────
        status_pane.object = "**Status:** Computing VWAP, pivots, sweeps..."
        config = StrategyConfig(
            vwap_num_deviations=vwap_dev_slider.value,
            pivot_lookback=lookback_slider.value,
            volume_ratio_min=vol_ratio_slider.value,
            tick_size=0.25,
        )
        prepped = prepare_bars(bars_raw, config=config)

        # ── 7. Signals & backtest ─────────────────────────────────
        status_pane.object = "**Status:** Generating signals & backtesting..."
        signals = generate_signals(prepped, config=config)
        engine = BacktestEngine(config=config, point_value=POINT_VALUE[sym])
        result = engine.run(prepped, signals)

        # ── 8. Render everything ──────────────────────────────────
        status_pane.object = "**Status:** Rendering charts..."

        # Tab 1 — Bookmap heatmap + trade bubbles + CVD + imbalance
        hm = plot_book_heatmap(
            heatmap_df, trades_df=trades,
            title=f"{sym} — Order Book Heatmap",
        )
        # Overlay absorption & sweep markers
        t0 = int(heatmap_df["ts_event"].min()) if len(heatmap_df) > 0 else 0
        hm = hm * absorption_markers(absorption, t0_ns=t0)
        hm = hm * sweep_markers(sweeps, t0_ns=t0)
        heatmap_pane.object = hm

        cvd_pane.object = plot_cvd(trades, title=f"{sym} — CVD")
        imbalance_pane.object = plot_imbalance_strip(summary_df)

        # Tab 2 — Candlestick + VWAP + signals
        chart_pane.object = plot_candlestick_vwap(
            prepped, signals=signals, backtest_trades=result.trades,
            title=f"{sym} — VWAP Deviation Stop-Hunt Reversal",
        )

        # Tab 3 — Volume profile
        profile_pane.object = plot_volume_profile(profile, title=f"{sym} Volume Profile")

        # Tab 4 — Backtest
        equity_pane.object = plot_equity_curve(
            result.trades, point_value=POINT_VALUE[sym],
        )
        metrics_pane.object = f"```\n{result.summary()}\n```"
        if result.trades:
            trades_pane.object = trades_to_dataframe(result.trades).to_pandas()

        # Final status
        n_sigs = len(signals)
        n_trades = result.num_trades
        n_snaps = len(snapshots)
        status_pane.object = (
            f"**Status:** Done. "
            f"{n_snaps:,} book snapshots | "
            f"{n_sigs} signals | {n_trades} trades | "
            f"PnL: {result.total_pnl_points:+.2f} pts "
            f"(${result.total_pnl_dollars:+,.0f})"
        )

    except Exception as e:
        status_pane.object = f"**Error:** {e}"
        logger.exception("Dashboard error")


load_button.on_click(on_load)

# ══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════

sidebar = pn.Column(
    "## Order Flow Lab",
    pn.layout.Divider(),
    "### Data",
    symbol_select,
    date_picker,
    start_hour_slider,
    end_hour_slider,
    pn.pane.Markdown("*RTH ≈ 13:30–20:00 UTC*", styles={"color": "#888", "font-size": "11px"}),
    pn.layout.Divider(),
    "### Strategy",
    lookback_slider,
    vwap_dev_slider,
    vol_ratio_slider,
    pn.layout.Divider(),
    "### Book",
    snapshot_freq_slider,
    pn.layout.Divider(),
    load_button,
    pn.layout.Divider(),
    "### Metrics",
    metrics_pane,
    width=320,
)

# Bookmap tab: heatmap stacked above CVD and imbalance (shared X-axis)
bookmap_tab = pn.Column(
    heatmap_pane,
    cvd_pane,
    imbalance_pane,
    name="Bookmap",
)

price_tab = pn.Column(
    chart_pane,
    name="Price + VWAP",
)

profile_tab = pn.Column(
    profile_pane,
    name="Volume Profile",
)

backtest_tab = pn.Column(
    equity_pane,
    pn.layout.Divider(),
    "### Trade Log",
    trades_pane,
    name="Backtest",
)

main = pn.Column(
    status_pane,
    pn.Tabs(
        bookmap_tab,
        price_tab,
        profile_tab,
        backtest_tab,
        dynamic=True,
    ),
)

template = pn.template.FastListTemplate(
    title="Order Flow Lab — VWAP Deviation Stop-Hunt Reversal",
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#d2a8ff",
    header_background="#0d1117",
    theme="dark",
)

template.servable()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5006))
    template.show(port=port)
