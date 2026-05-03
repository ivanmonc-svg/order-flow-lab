"""Databento interface for downloading and caching MBO/MBP/trades data for ES & NQ futures.

Usage:
    loader = DataLoader(api_key="your-key", data_dir="./data")
    trades = loader.get_trades("ES", date(2026, 4, 28))
    mbp = loader.get_mbp("NQ", date(2026, 4, 28), depth=10)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import databento as db
import polars as pl

logger = logging.getLogger(__name__)

# CME Globex dataset for ES & NQ futures
DATASET = "GLBX.MDP3"

# Continuous front-month symbols
SYMBOLS = {
    "ES": "ES.c.0",
    "NQ": "NQ.c.0",
}

# Tick sizes (in price units)
TICK_SIZE = {
    "ES": 0.25,
    "NQ": 0.25,
}

# Point values (dollars per point)
POINT_VALUE = {
    "ES": 50.0,
    "NQ": 20.0,
}


class DataLoader:
    """Interface to Databento for downloading and caching market data."""

    def __init__(
        self,
        api_key: str | None = None,
        data_dir: str | Path = "./data",
    ) -> None:
        self.client = db.Historical(key=api_key)  # uses DATABENTO_API_KEY env var if None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Cache helpers ────────────────────────────────────────────────────

    def _cache_path(self, symbol: str, dt: date, schema: str) -> Path:
        return self.data_dir / f"{symbol}_{dt.isoformat()}_{schema}.parquet"

    def _has_cache(self, symbol: str, dt: date, schema: str) -> bool:
        return self._cache_path(symbol, dt, schema).exists()

    # ── Core download ────────────────────────────────────────────────────

    def _download(
        self,
        symbol: str,
        dt: date,
        schema: str,
        stype_in: str = "continuous",
    ) -> pl.DataFrame:
        """Download one day of data from Databento and cache as parquet."""
        cache = self._cache_path(symbol, dt, schema)
        if cache.exists():
            logger.info("Cache hit: %s", cache)
            return pl.read_parquet(cache)

        logger.info("Downloading %s %s %s from Databento...", symbol, dt, schema)
        raw_sym = SYMBOLS.get(symbol, symbol)
        data = self.client.timeseries.get_range(
            dataset=DATASET,
            symbols=[raw_sym],
            schema=schema,
            stype_in=stype_in,
            start=dt.isoformat(),
            end=(dt + timedelta(days=1)).isoformat(),
        )

        # Convert DBN to DataFrame via to_df() then to polars
        pdf = data.to_df()
        df = pl.from_pandas(pdf.reset_index())

        # Write cache
        df.write_parquet(cache)
        logger.info("Cached %d rows → %s", len(df), cache)
        return df

    # ── Public API ───────────────────────────────────────────────────────

    def get_trades(self, symbol: str, dt: date) -> pl.DataFrame:
        """Get all trades for a symbol on a given date.

        Returns DataFrame with columns:
            ts_event, price, size, side, action, flags, sequence, ...
        """
        return self._download(symbol, dt, "trades")

    def get_mbp(self, symbol: str, dt: date, depth: int = 1) -> pl.DataFrame:
        """Get MBP (Market by Price) snapshots.

        Args:
            depth: 1 for BBO (top of book), 10 for top-10 levels.

        Returns DataFrame with bid/ask levels at each update.
        """
        schema = f"mbp-{depth}"
        return self._download(symbol, dt, schema)

    def get_ohlcv(
        self, symbol: str, dt: date, interval: str = "1m"
    ) -> pl.DataFrame:
        """Get OHLCV bars.

        Args:
            interval: '1s', '1m', '1h', or '1d'
        """
        schema = f"ohlcv-{interval}"
        return self._download(symbol, dt, schema)

    def get_definition(self, symbol: str, dt: date) -> pl.DataFrame:
        """Get instrument definitions (tick size, multiplier, etc.)."""
        return self._download(symbol, dt, "definition")

    # ── Multi-day helpers ────────────────────────────────────────────────

    def get_trades_range(
        self, symbol: str, start: date, end: date
    ) -> pl.DataFrame:
        """Download trades for a date range and concatenate."""
        frames = []
        current = start
        while current <= end:
            try:
                df = self.get_trades(symbol, current)
                frames.append(df)
            except Exception as e:
                logger.warning("Skipping %s %s: %s", symbol, current, e)
            current += timedelta(days=1)
        if not frames:
            raise ValueError(f"No data for {symbol} from {start} to {end}")
        return pl.concat(frames)

    def get_mbp_range(
        self, symbol: str, start: date, end: date, depth: int = 1
    ) -> pl.DataFrame:
        """Download MBP for a date range and concatenate."""
        frames = []
        current = start
        while current <= end:
            try:
                df = self.get_mbp(symbol, current, depth=depth)
                frames.append(df)
            except Exception as e:
                logger.warning("Skipping %s %s: %s", symbol, current, e)
            current += timedelta(days=1)
        if not frames:
            raise ValueError(f"No data for {symbol} from {start} to {end}")
        return pl.concat(frames)

    # ── Cost estimation ──────────────────────────────────────────────────

    def estimate_cost(
        self, symbol: str, start: date, end: date, schema: str = "trades"
    ) -> dict:
        """Estimate download cost in USD before committing."""
        raw_sym = SYMBOLS.get(symbol, symbol)
        return self.client.metadata.get_cost(
            dataset=DATASET,
            symbols=[raw_sym],
            schema=schema,
            stype_in="continuous",
            start=start.isoformat(),
            end=end.isoformat(),
        )
