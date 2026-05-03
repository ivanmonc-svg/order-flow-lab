"""Databento interface for downloading and caching MBO/MBP/trades data for ES & NQ futures.

Usage:
    loader = DataLoader(api_key="your-key", data_dir="./data")
    trades = loader.get_trades("ES", date(2026, 4, 28))
    mbp = loader.get_mbp("NQ", date(2026, 4, 28), depth=10)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
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
        start_hour: int | None = None,
        end_hour: int | None = None,
    ) -> pl.DataFrame:
        """Download data from Databento and cache as parquet.

        Args:
            start_hour: UTC hour to start (e.g., 13 for 9:30 AM ET).
                        If None, downloads from midnight.
            end_hour:   UTC hour to end (e.g., 15 for 11:00 AM ET).
                        If None, downloads until midnight next day.
        """
        # Build a cache key that includes the hour range
        hour_tag = ""
        if start_hour is not None or end_hour is not None:
            sh = start_hour or 0
            eh = end_hour or 24
            hour_tag = f"_h{sh}-{eh}"
        cache = self.data_dir / f"{symbol}_{dt.isoformat()}_{schema}{hour_tag}.parquet"

        if cache.exists():
            logger.info("Cache hit: %s", cache)
            return pl.read_parquet(cache)

        logger.info("Downloading %s %s %s from Databento...", symbol, dt, schema)
        raw_sym = SYMBOLS.get(symbol, symbol)

        # Build time range
        if start_hour is not None:
            start_ts = f"{dt.isoformat()}T{start_hour:02d}:00:00+00:00"
        else:
            start_ts = dt.isoformat()

        if end_hour is not None:
            end_ts = f"{dt.isoformat()}T{end_hour:02d}:00:00+00:00"
        else:
            end_ts = (dt + timedelta(days=1)).isoformat()

        data = self.client.timeseries.get_range(
            dataset=DATASET,
            symbols=[raw_sym],
            schema=schema,
            stype_in=stype_in,
            start=start_ts,
            end=end_ts,
        )

        # Convert DBN to DataFrame via to_df() then to polars
        pdf = data.to_df()
        df = pl.from_pandas(pdf.reset_index())

        # Write cache
        df.write_parquet(cache)
        logger.info("Cached %d rows → %s", len(df), cache)
        return df

    # ── Timestamp-based download (minute precision) ────────────────────

    def _download_ts(
        self,
        symbol: str,
        start_ts: str,
        end_ts: str,
        schema: str,
        stype_in: str = "continuous",
    ) -> pl.DataFrame:
        """Download data using exact ISO timestamps (minute precision).

        Args:
            start_ts: ISO 8601 start time, e.g. "2026-04-28T14:00:00+00:00".
            end_ts:   ISO 8601 end time,   e.g. "2026-04-28T14:15:00+00:00".
        """
        # Sanitize timestamps for cache filename
        safe_start = start_ts.replace(":", "").replace("+", "p")[:19]
        safe_end = end_ts.replace(":", "").replace("+", "p")[:19]
        cache = self.data_dir / f"{symbol}_{safe_start}_{safe_end}_{schema}.parquet"

        if cache.exists():
            logger.info("Cache hit: %s", cache)
            return pl.read_parquet(cache)

        logger.info(
            "Downloading %s %s [%s → %s] from Databento...",
            symbol, schema, start_ts, end_ts,
        )
        raw_sym = SYMBOLS.get(symbol, symbol)

        data = self.client.timeseries.get_range(
            dataset=DATASET,
            symbols=[raw_sym],
            schema=schema,
            stype_in=stype_in,
            start=start_ts,
            end=end_ts,
        )

        # Convert DBN to DataFrame via to_df() then to polars
        pdf = data.to_df()
        df = pl.from_pandas(pdf.reset_index())

        # Write cache
        df.write_parquet(cache)
        logger.info("Cached %d rows → %s", len(df), cache)
        return df

    def get_trades_ts(
        self, symbol: str, start_ts: str, end_ts: str,
    ) -> pl.DataFrame:
        """Get trades using exact ISO timestamps."""
        return self._download_ts(symbol, start_ts, end_ts, "trades")

    def get_mbp_ts(
        self, symbol: str, start_ts: str, end_ts: str, depth: int = 1,
    ) -> pl.DataFrame:
        """Get MBP data using exact ISO timestamps."""
        schema = f"mbp-{depth}"
        return self._download_ts(symbol, start_ts, end_ts, schema)

    def get_ohlcv_ts(
        self, symbol: str, start_ts: str, end_ts: str, interval: str = "1m",
    ) -> pl.DataFrame:
        """Get OHLCV bars using exact ISO timestamps."""
        schema = f"ohlcv-{interval}"
        return self._download_ts(symbol, start_ts, end_ts, schema)

    # ── Public API ───────────────────────────────────────────────────────

    def get_trades(
        self, symbol: str, dt: date,
        start_hour: int | None = None, end_hour: int | None = None,
    ) -> pl.DataFrame:
        """Get trades for a symbol on a given date.

        Args:
            start_hour: UTC hour to start (default: full day).
            end_hour:   UTC hour to end.
        """
        return self._download(symbol, dt, "trades", start_hour=start_hour, end_hour=end_hour)

    def get_mbp(
        self, symbol: str, dt: date, depth: int = 1,
        start_hour: int | None = None, end_hour: int | None = None,
    ) -> pl.DataFrame:
        """Get MBP (Market by Price) snapshots.

        Args:
            depth: 1 for BBO (top of book), 10 for top-10 levels.
            start_hour: UTC hour to start (default: full day).
            end_hour:   UTC hour to end.
        """
        schema = f"mbp-{depth}"
        return self._download(symbol, dt, schema, start_hour=start_hour, end_hour=end_hour)

    def get_ohlcv(
        self, symbol: str, dt: date, interval: str = "1m",
        start_hour: int | None = None, end_hour: int | None = None,
    ) -> pl.DataFrame:
        """Get OHLCV bars.

        Args:
            interval: '1s', '1m', '1h', or '1d'
            start_hour: UTC hour to start (default: full day).
            end_hour:   UTC hour to end.
        """
        schema = f"ohlcv-{interval}"
        return self._download(symbol, dt, schema, start_hour=start_hour, end_hour=end_hour)

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
