"""
Module 1: Data Loader & 3s Grid Aligner
========================================
Polars lazy-scan parquet, column pruning, 3s time bucketing,
SH+SZ market merge.

Trade/order aggregation onto the snap 3s grid is handled in
feature_factory.py (ceil bucketing + asof-join backward).
"""

from __future__ import annotations

import os
from typing import Optional

import polars as pl

from ml_framework.config import (
    DATA_ROOT, TRADING_HOURS, LUNCH_START, LUNCH_END,
    MARKET_OPEN, MARKET_CLOSE, SNAP_REQUIRED_COLS,
)


def _parse_time_to_3s_bucket(time_col: pl.Expr) -> pl.Expr:
    """Parse 'HH:MM:SS[.mmm]' string → 3s-bucket seconds since midnight (int32).

    The raw UpdateTime includes milliseconds (e.g. '09:30:03.123').
    We strip fractional seconds, then compute:
      hour * 3600 + minute * 60 + floor(second / 3) * 3.

    Explicit i32 casts prevent i8→i16 overflow in the arithmetic.
    """
    ts_parsed = pl.col("UpdateTime").str.slice(0, 8).str.to_time("%H:%M:%S")
    return (
        ts_parsed.dt.hour().cast(pl.Int32) * 3600
        + ts_parsed.dt.minute().cast(pl.Int32) * 60
        + (ts_parsed.dt.second().cast(pl.Int32) // 3) * 3
    ).cast(pl.Int32).alias("timestamp")


class SnapDataLoader:
    """Lazy-load snap parquet, align to 3s grid, merge SH + SZ.

    SH and SZ exchanges use different column names for the same concepts.
    We normalize SZ → SH on load so downstream code sees a unified schema.
    """

    # SZ column name → SH column name mapping
    _SZ_TO_SH_RENAME: dict[str, str] = {
        "TotalBidQty": "TotalBidVol",
        "TotalOfferQty": "TotalAskVol",
        "TurnNum": "TradNumber",
        "Volume": "TradVolume",
    }

    # Columns that exist in SH but not SZ (will be null-filled for SZ)
    _SH_ONLY_COLS: set[str] = {"MaxBidDur", "MaxSellDur", "TotBidNum", "TotSellNum"}

    def __init__(self, data_root: str = DATA_ROOT) -> None:
        self._data_root = data_root

    def _snap_path(self, date: str, market: str) -> str:
        return os.path.join(self._data_root, f"type=snap_{market}", f"date={date}", "data.parquet")

    def load_single_market(
        self, date: str, market: str, columns: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """Lazy-scan one market, collect with column pruning.

        SZ columns are renamed to SH naming convention.
        SH-only columns missing from SZ are filled with null.
        """
        path = self._snap_path(date, market)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing parquet: {path}")

        cols = columns or SNAP_REQUIRED_COLS
        # Ensure UpdateTime + SecurityID are always present
        if "UpdateTime" not in cols:
            cols = ["UpdateTime"] + cols
        if "SecurityID" not in cols:
            cols = ["SecurityID"] + cols

        # Get actual parquet schema so we don't request non-existent columns
        scan = pl.scan_parquet(path)
        parquet_cols = set(scan.collect_schema().names())

        # For SZ: translate SH column names to SZ equivalents
        if market == "sz":
            scan_cols: list[str] = []
            for c in cols:
                sz_c = self._SZ_TO_SH_RENAME.get(c, c)
                if sz_c in parquet_cols:
                    scan_cols.append(sz_c)
                # else: column not in this market's parquet → skip, will null-fill
            # Deduplicate
            scan_cols = list(dict.fromkeys(scan_cols))
        else:
            scan_cols = [c for c in cols if c in parquet_cols]

        lf = scan.select(scan_cols)

        # Filter to trading hours
        lf = lf.filter(
            pl.col("UpdateTime").is_between(
                pl.lit(TRADING_HOURS[0]), pl.lit(TRADING_HOURS[1])
            )
        )

        df = lf.collect()

        # Rename SZ columns → SH names
        if market == "sz":
            rename_map = {v: k for k, v in self._SZ_TO_SH_RENAME.items() if v in df.columns}
            if rename_map:
                df = df.rename(rename_map)

        # Null-fill any requested columns that don't exist in this market
        missing = [c for c in cols if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.lit(None).cast(pl.Float64).alias(c) for c in missing]
            )

        return df

    def load_day_merged(self, date: str, columns: Optional[list[str]] = None) -> pl.DataFrame:
        """Load SH + SZ, concat, add 3s timestamp, sort.

        Returns a Polars DataFrame with columns:
          timestamp (Int32), SecurityID (str), ... all selected columns.

        Note: SH and SZ parquet files use different column names for the
        same concepts.  SZ is normalized to SH naming on load.  SH-only
        columns (MaxBidDur, etc.) are null-filled for SZ.
        """
        frames: list[pl.DataFrame] = []
        for mkt in ("sh", "sz"):
            try:
                frames.append(self.load_single_market(date, mkt, columns))
            except FileNotFoundError:
                continue

        if not frames:
            raise RuntimeError(f"No data found for date={date}")

        # Ensure all frames have the same columns.
        # SH lacks HighLimitPrice/LowLimitPrice which SZ provides natively.
        # We union the column superset and null-fill missing columns per frame
        # so SZ's native limit prices survive the merge.
        all_columns = list(dict.fromkeys(
            col for frame in frames for col in frame.columns
        ))
        for i, frame in enumerate(frames):
            missing = [c for c in all_columns if c not in frame.columns]
            if missing:
                frames[i] = frame.with_columns(
                    [pl.lit(None).cast(pl.Float64).alias(c) for c in missing]
                )
            # Align column order
            if frame.columns != all_columns:
                frames[i] = frame.select(all_columns)

        # Align schemas: cast numeric columns to Float64
        target_schema = {}
        for frame in frames:
            for c, dt in zip(frame.columns, frame.dtypes):
                if dt.is_numeric():
                    target_schema[c] = pl.Float64
        for i, frame in enumerate(frames):
            casts = []
            for c in frame.columns:
                if c in target_schema and frame[c].dtype != target_schema[c]:
                    casts.append(pl.col(c).cast(target_schema[c]))
            if casts:
                frames[i] = frame.with_columns(casts)

        df = pl.concat(frames, how="vertical")

        # Build 3s timestamp key
        df = df.with_columns(_parse_time_to_3s_bucket(pl.col("UpdateTime")))

        # Sort by SecurityID then timestamp for deterministic row order
        df = df.sort(["SecurityID", "timestamp"])

        # Deduplicate: raw data may have multiple snapshots within one 3s
        # bucket for the same stock.  Keep the last one.
        # NOTE: unique() does NOT preserve sort order — must re-sort after.
        df = df.unique(subset=["SecurityID", "timestamp"], keep="last")
        df = df.sort(["SecurityID", "timestamp"])

        return df
