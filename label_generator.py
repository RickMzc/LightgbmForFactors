"""
Module 2: Label Generator
=========================
Strict mid-price forward log-returns with lunch-break,
day-boundary, and cross-day NaN enforcement.

All forward shifts use .over("SecurityID") — never bare shift
on the full-market DataFrame.
"""

from __future__ import annotations

import polars as pl

from ml_framework.config import (
    HORIZONS, HORIZON_NAMES, HORIZON_SHIFTS, HORIZON_SECONDS,
    LUNCH_START, LUNCH_END, MARKET_CLOSE, EPS,
)


class LabelGenerator:
    """Generate forward mid-price log-returns with microstructure-aware
    boundary handling.

    Parameters
    ----------
    horizons : dict[str, int]
        Mapping of label name → shift steps (each step ≈ 3s).
        Default from config.HORIZONS.
    """

    def __init__(self, horizons: dict[str, int] | None = None) -> None:
        self._horizons = horizons or HORIZONS
        self._horizon_names = list(self._horizons.keys())
        self._horizon_shifts = list(self._horizons.values())

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute forward returns on a Polars DataFrame.

        Expects columns: SecurityID, AskPrice1, BidPrice1, timestamp (Int32).

        Returns the input DataFrame with extra columns:
          mid_price (Float64), ret_15s, ret_30s, ret_60s, ret_180s, ret_300s.
        """
        df = self._compute_mid_price(df)
        df = self._forward_log_returns(df)
        df = self._mask_boundary_violations(df)
        return df

    def _compute_mid_price(self, df: pl.DataFrame) -> pl.DataFrame:
        """Mid-price = (AskPrice1 + BidPrice1) / 2 with limit-up/down handling.

        - If AskPrice1 <= 0 or null → use BidPrice1 only.
        - If BidPrice1 <= 0 or null → use AskPrice1 only.
        - If both invalid → NaN.
        """
        return df.with_columns(
            pl.when(
                (pl.col("AskPrice1").is_not_null())
                & (pl.col("BidPrice1").is_not_null())
                & (pl.col("AskPrice1") > 0)
                & (pl.col("BidPrice1") > 0)
            )
            .then((pl.col("AskPrice1") + pl.col("BidPrice1")) / 2.0)
            .when(
                (pl.col("AskPrice1").is_not_null()) & (pl.col("AskPrice1") > 0)
            )
            .then(pl.col("AskPrice1"))
            .when(
                (pl.col("BidPrice1").is_not_null()) & (pl.col("BidPrice1") > 0)
            )
            .then(pl.col("BidPrice1"))
            .otherwise(None)
            .alias("mid_price")
        )

    def _forward_log_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """ln(mid_price_{t+h}) - ln(mid_price_t), shifted per SecurityID.

        CRITICAL: .shift(-h).over("SecurityID") — without .over(), Polars
        shifts across the full DataFrame, contaminating stocks with each
        other's prices.

        Additionally validates that the actual time gap at the shifted rows
        equals the expected horizon seconds (±3s tolerance).  When a stock
        has missing snapshots, shift(-5) may traverse more than 15 seconds,
        producing a false horizon label.
        """
        ln_mid = pl.col("mid_price").log()
        ts = pl.col("timestamp")

        exprs = [ln_mid.alias("ln_mid")]
        for name, shift_h, horizon_secs in zip(
            self._horizon_names, self._horizon_shifts,
            [HORIZON_SECONDS[n] for n in self._horizon_names],
        ):
            fwd_ln = ln_mid.shift(-shift_h).over("SecurityID")
            fwd_ts = ts.shift(-shift_h).over("SecurityID")
            # Validate actual time gap; allow ±3s tolerance
            gap_ok = (fwd_ts - ts - pl.lit(horizon_secs)).abs() <= 3
            raw_ret = fwd_ln - ln_mid
            exprs.append(
                pl.when(gap_ok).then(raw_ret).otherwise(None).alias(name)
            )

        return df.with_columns(exprs)

    def _mask_boundary_violations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Set forward return to NaN when the look-ahead window crosses:

        1. Lunch break:  bucket[t] <= 41400 AND bucket[t+h] >= 46800
        2. End of day:   bucket[t+h] > 54000
        3. Cross-day:    bucket[t+h] < bucket[t]  (seconds wrap around)

        Each check uses the corresponding shifted timestamp to ensure
        the horizon boundary is compared to the target time, not the source.
        """
        ts = pl.col("timestamp")

        for name, shift in zip(self._horizon_names, self._horizon_shifts):
            ts_fwd = ts.shift(-shift).over("SecurityID")

            # Condition 1: spans lunch break
            crosses_lunch = (
                (ts <= LUNCH_START) & (ts_fwd >= LUNCH_END)
            )
            # Condition 2: target past market close
            past_close = ts_fwd > MARKET_CLOSE
            # Condition 3: wrapped to next day
            wrapped_day = ts_fwd < ts

            invalid = crosses_lunch | past_close | wrapped_day

            df = df.with_columns(
                pl.when(invalid | ts_fwd.is_null())
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
            )

        return df
