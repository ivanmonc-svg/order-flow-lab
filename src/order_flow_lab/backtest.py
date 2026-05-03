"""Event-driven backtest engine for the VWAP Deviation Stop-Hunt Reversal strategy.

Processes signals bar-by-bar, manages positions with partial exits (TP1/TP2),
stop losses, and time-stops. Produces a trade log and summary metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .strategy import Side, Signal, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A completed trade with entry/exit details."""

    signal: Signal
    entry_bar: int
    entry_price: float
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    bars_held: int = 0
    partial_exits: list[dict] = field(default_factory=list)


@dataclass
class Position:
    """An active position being managed."""

    signal: Signal
    entry_bar: int
    entry_price: float
    remaining_qty: float = 1.0  # 1.0 = full, 0.5 = after TP1
    tp1_hit: bool = False


@dataclass
class BacktestResult:
    """Summary of backtest results."""

    trades: list[Trade]
    total_pnl_points: float
    total_pnl_dollars: float
    num_trades: int
    num_winners: int
    num_losers: int
    win_rate: float
    avg_winner_pts: float
    avg_loser_pts: float
    profit_factor: float
    max_drawdown_pts: float
    sharpe_ratio: float
    avg_bars_held: float

    def summary(self) -> str:
        return (
            f"═══ Backtest Summary ═══\n"
            f"Trades:        {self.num_trades}\n"
            f"Win rate:      {self.win_rate:.1%}\n"
            f"Total PnL:     {self.total_pnl_points:+.2f} pts (${self.total_pnl_dollars:+,.0f})\n"
            f"Avg winner:    {self.avg_winner_pts:+.2f} pts\n"
            f"Avg loser:     {self.avg_loser_pts:+.2f} pts\n"
            f"Profit factor: {self.profit_factor:.2f}\n"
            f"Max drawdown:  {self.max_drawdown_pts:.2f} pts\n"
            f"Sharpe ratio:  {self.sharpe_ratio:.2f}\n"
            f"Avg bars held: {self.avg_bars_held:.1f}\n"
        )


class BacktestEngine:
    """Bar-by-bar backtest engine with position management."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        point_value: float = 50.0,  # ES default
        max_positions: int = 1,
    ) -> None:
        self.config = config or StrategyConfig()
        self.point_value = point_value
        self.max_positions = max_positions

    def run(
        self,
        bars: pl.DataFrame,
        signals: list[Signal],
    ) -> BacktestResult:
        """Execute backtest over bars with given signals.

        Args:
            bars: OHLCV bars with VWAP columns (from prepare_bars).
            signals: List of Signal objects (from generate_signals).

        Returns:
            BacktestResult with trade log and metrics.
        """
        # Index signals by bar
        signal_map: dict[int, list[Signal]] = {}
        for sig in signals:
            signal_map.setdefault(sig.bar_idx, []).append(sig)

        positions: list[Position] = []
        completed_trades: list[Trade] = []
        n_bars = len(bars)

        bars_list = bars.to_dicts()

        for i in range(n_bars):
            bar = bars_list[i]
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_close = bar["close"]

            # ── Manage existing positions ─────────────────────────
            still_open = []
            for pos in positions:
                bars_held = i - pos.entry_bar
                closed = False
                trade = Trade(
                    signal=pos.signal,
                    entry_bar=pos.entry_bar,
                    entry_price=pos.entry_price,
                )

                if pos.signal.side == Side.LONG:
                    # Check stop loss
                    if bar_low <= pos.signal.stop_loss:
                        trade.exit_bar = i
                        trade.exit_price = pos.signal.stop_loss
                        trade.exit_reason = "stop_loss"
                        trade.pnl_points = (trade.exit_price - pos.entry_price) * pos.remaining_qty
                        closed = True

                    # Check TP1
                    elif not pos.tp1_hit and bar_high >= pos.signal.tp1:
                        pos.tp1_hit = True
                        partial_pnl = (pos.signal.tp1 - pos.entry_price) * self.config.tp1_pct
                        trade.partial_exits.append({
                            "bar": i, "price": pos.signal.tp1,
                            "qty": self.config.tp1_pct, "pnl_pts": partial_pnl,
                            "reason": "tp1",
                        })
                        pos.remaining_qty -= self.config.tp1_pct

                    # Check TP2
                    if not closed and pos.tp1_hit and bar_high >= pos.signal.tp2:
                        trade.exit_bar = i
                        trade.exit_price = pos.signal.tp2
                        trade.exit_reason = "tp2"
                        tp2_pnl = (pos.signal.tp2 - pos.entry_price) * pos.remaining_qty
                        total_partial = sum(p["pnl_pts"] for p in trade.partial_exits)
                        trade.pnl_points = total_partial + tp2_pnl
                        closed = True

                    # Check time-stop
                    if not closed and bars_held >= self.config.time_stop_bars:
                        trade.exit_bar = i
                        trade.exit_price = bar_close
                        trade.exit_reason = "time_stop"
                        remaining_pnl = (bar_close - pos.entry_price) * pos.remaining_qty
                        total_partial = sum(p["pnl_pts"] for p in trade.partial_exits)
                        trade.pnl_points = total_partial + remaining_pnl
                        closed = True

                else:  # SHORT
                    if bar_high >= pos.signal.stop_loss:
                        trade.exit_bar = i
                        trade.exit_price = pos.signal.stop_loss
                        trade.exit_reason = "stop_loss"
                        trade.pnl_points = (pos.entry_price - trade.exit_price) * pos.remaining_qty
                        closed = True

                    elif not pos.tp1_hit and bar_low <= pos.signal.tp1:
                        pos.tp1_hit = True
                        partial_pnl = (pos.entry_price - pos.signal.tp1) * self.config.tp1_pct
                        trade.partial_exits.append({
                            "bar": i, "price": pos.signal.tp1,
                            "qty": self.config.tp1_pct, "pnl_pts": partial_pnl,
                            "reason": "tp1",
                        })
                        pos.remaining_qty -= self.config.tp1_pct

                    if not closed and pos.tp1_hit and bar_low <= pos.signal.tp2:
                        trade.exit_bar = i
                        trade.exit_price = pos.signal.tp2
                        trade.exit_reason = "tp2"
                        tp2_pnl = (pos.entry_price - pos.signal.tp2) * pos.remaining_qty
                        total_partial = sum(p["pnl_pts"] for p in trade.partial_exits)
                        trade.pnl_points = total_partial + tp2_pnl
                        closed = True

                    if not closed and bars_held >= self.config.time_stop_bars:
                        trade.exit_bar = i
                        trade.exit_price = bar_close
                        trade.exit_reason = "time_stop"
                        remaining_pnl = (pos.entry_price - bar_close) * pos.remaining_qty
                        total_partial = sum(p["pnl_pts"] for p in trade.partial_exits)
                        trade.pnl_points = total_partial + remaining_pnl
                        closed = True

                if closed:
                    trade.bars_held = i - pos.entry_bar
                    trade.pnl_dollars = trade.pnl_points * self.point_value
                    completed_trades.append(trade)
                else:
                    still_open.append(pos)

            positions = still_open

            # ── Open new positions ────────────────────────────────
            if i in signal_map and len(positions) < self.max_positions:
                for sig in signal_map[i]:
                    if len(positions) >= self.max_positions:
                        break
                    positions.append(
                        Position(
                            signal=sig,
                            entry_bar=i,
                            entry_price=sig.entry_price,
                        )
                    )

        # Close any remaining positions at last bar
        if positions:
            last_bar = bars_list[-1]
            for pos in positions:
                trade = Trade(
                    signal=pos.signal,
                    entry_bar=pos.entry_bar,
                    entry_price=pos.entry_price,
                    exit_bar=n_bars - 1,
                    exit_price=last_bar["close"],
                    exit_reason="end_of_data",
                    bars_held=n_bars - 1 - pos.entry_bar,
                )
                if pos.signal.side == Side.LONG:
                    remaining_pnl = (last_bar["close"] - pos.entry_price) * pos.remaining_qty
                else:
                    remaining_pnl = (pos.entry_price - last_bar["close"]) * pos.remaining_qty
                total_partial = sum(p["pnl_pts"] for p in trade.partial_exits)
                trade.pnl_points = total_partial + remaining_pnl
                trade.pnl_dollars = trade.pnl_points * self.point_value
                completed_trades.append(trade)

        return self._compute_metrics(completed_trades)

    def _compute_metrics(self, trades: list[Trade]) -> BacktestResult:
        """Compute summary statistics from completed trades."""
        if not trades:
            return BacktestResult(
                trades=[], total_pnl_points=0, total_pnl_dollars=0,
                num_trades=0, num_winners=0, num_losers=0,
                win_rate=0, avg_winner_pts=0, avg_loser_pts=0,
                profit_factor=0, max_drawdown_pts=0, sharpe_ratio=0,
                avg_bars_held=0,
            )

        pnls = np.array([t.pnl_points for t in trades])
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        # Drawdown
        equity = np.cumsum(pnls)
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        # Sharpe (annualized, assuming ~252 trading days)
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0

        return BacktestResult(
            trades=trades,
            total_pnl_points=float(pnls.sum()),
            total_pnl_dollars=float(pnls.sum() * self.point_value),
            num_trades=len(trades),
            num_winners=len(winners),
            num_losers=len(losers),
            win_rate=len(winners) / len(trades) if trades else 0,
            avg_winner_pts=float(winners.mean()) if len(winners) > 0 else 0,
            avg_loser_pts=float(losers.mean()) if len(losers) > 0 else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            max_drawdown_pts=max_dd,
            sharpe_ratio=sharpe,
            avg_bars_held=float(np.mean([t.bars_held for t in trades])),
        )


def trades_to_dataframe(trades: list[Trade]) -> pl.DataFrame:
    """Convert trade list to a Polars DataFrame for analysis."""
    records = []
    for t in trades:
        records.append({
            "entry_bar": t.entry_bar,
            "exit_bar": t.exit_bar,
            "side": t.signal.side.name,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.signal.stop_loss,
            "tp1": t.signal.tp1,
            "tp2": t.signal.tp2,
            "pnl_points": t.pnl_points,
            "pnl_dollars": t.pnl_dollars,
            "bars_held": t.bars_held,
            "exit_reason": t.exit_reason,
            "volume_ratio": t.signal.volume_ratio,
            "reason": t.signal.reason,
        })
    return pl.DataFrame(records)
