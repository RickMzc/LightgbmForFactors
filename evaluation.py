"""
Module 5: Cross-Sectional Rank-IC Evaluation
=============================================
Polars-native Spearman Rank-IC per 3s timestamp,
IC summary statistics, and IC decay analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

from ml_framework.config import HORIZON_NAMES, MIN_STOCKS, EPS


@dataclass
class ICSummary:
    """IC statistics for a single horizon."""
    mean_ic: float
    std_ic: float
    icir: float
    win_rate: float  # fraction of positive ICs
    n_buckets: int   # number of cross-sectional slices

    def __repr__(self) -> str:
        return (
            f"IC={self.mean_ic:+.4f}  std={self.std_ic:.4f}  "
            f"ICIR={self.icir:+.2f}  Win={self.win_rate:.1%}  "
            f"n={self.n_buckets}"
        )


@dataclass
class EvalResult:
    """Full evaluation result for one model/horizon."""
    horizon: str
    ic_series: pl.Series       # per-timestamp Rank-IC values
    summary: ICSummary
    ic_decay: dict[str, float]  # horizon → mean IC (for decay curve)
    daily_ic: pl.DataFrame      # date × mean IC

    def __repr__(self) -> str:
        return f"EvalResult({self.horizon}, {self.summary})"


class CrossSectionalEvaluator:
    """Compute cross-sectional Rank-IC (Spearman) per timestamp.

    Uses Polars-native corr(rank, rank, method='spearman') for
    efficient grouped computation, avoiding Python-loop groupby.
    """

    def __init__(self, min_stocks: int = MIN_STOCKS) -> None:
        self._min_stocks = min_stocks

    @staticmethod
    def rank_ic_per_timestamp(
        df: pl.DataFrame,
        pred_col: str = "prediction",
        label_col: str = "ret_15s",
        min_stocks: int = MIN_STOCKS,
    ) -> pl.DataFrame:
        """Compute Spearman Rank-IC for each 3s timestamp slice.

        Parameters
        ----------
        df : DataFrame with columns [timestamp, pred_col, label_col].
        pred_col : model prediction column.
        label_col : ground-truth forward return column.
        min_stocks : minimum stocks per timestamp to compute IC.

        Returns
        -------
        DataFrame with columns [timestamp, n_stocks, rank_ic].
        """
        ic_df = (
            df.select(["timestamp", pred_col, label_col])
            .drop_nulls()
            .group_by("timestamp")
            .agg([
                pl.len().alias("n_stocks"),
                pl.corr(pred_col, label_col, method="spearman").alias("rank_ic"),
            ])
            .filter(
                (pl.col("n_stocks") >= min_stocks)
                & pl.col("rank_ic").is_not_nan()
            )
            .sort("timestamp")
        )

        return ic_df

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        timestamps: pl.Series | np.ndarray,
        security_ids: pl.Series | np.ndarray,
        horizon: str = "ret_15s",
    ) -> EvalResult:
        """Compute full evaluation for one horizon.

        Parameters
        ----------
        predictions : model predictions (float array, shape n).
        labels : true forward returns (float array, shape n).
        timestamps : timestamp array (same length as predictions).
        security_ids : SecurityID array (same length as predictions).
        horizon : label name (e.g. "ret_15s").
        """
        n = len(predictions)
        assert len(labels) == n, f"Labels length {len(labels)} != preds {n}"

        eval_df = pl.DataFrame({
            "timestamp": pl.Series(timestamps) if not isinstance(timestamps, pl.Series) else timestamps,
            "SecurityID": pl.Series(security_ids) if not isinstance(security_ids, pl.Series) else security_ids,
            "prediction": pl.Series(predictions.astype(np.float64)),
            "label": pl.Series(labels.astype(np.float64)),
        })

        ic_df = self.rank_ic_per_timestamp(
            eval_df, pred_col="prediction", label_col="label",
            min_stocks=self._min_stocks,
        )

        ic_series = ic_df["rank_ic"].drop_nulls()

        if len(ic_series) == 0:
            summary = ICSummary(
                mean_ic=float("nan"), std_ic=float("nan"),
                icir=float("nan"), win_rate=float("nan"),
                n_buckets=0,
            )
        else:
            ics = ic_series.to_numpy()
            valid_ic = ics[~np.isnan(ics)]
            if len(valid_ic) == 0:
                summary = ICSummary(
                    mean_ic=float("nan"), std_ic=float("nan"),
                    icir=float("nan"), win_rate=float("nan"),
                    n_buckets=len(ics),
                )
            else:
                mean_ic = float(np.mean(valid_ic))
                std_ic = float(np.std(valid_ic))
                icir = mean_ic / (std_ic + EPS)
                win_rate = float(np.mean(valid_ic > 0))
            summary = ICSummary(
                mean_ic=mean_ic, std_ic=std_ic,
                icir=icir, win_rate=win_rate,
                n_buckets=len(ics),
            )

        return EvalResult(
            horizon=horizon,
            ic_series=ic_series,
            summary=summary,
            ic_decay={horizon: summary.mean_ic},
            daily_ic=ic_df,
        )

    def evaluate_multi_horizon(
        self,
        predictions: np.ndarray,
        labels_dict: dict[str, np.ndarray],
        timestamps: pl.Series,
        security_ids: pl.Series,
    ) -> dict[str, EvalResult]:
        """Evaluate predictions against multiple horizons simultaneously.

        Parameters
        ----------
        predictions : model predictions.
        labels_dict : {horizon_name: label_array}.
        timestamps, security_ids : metadata arrays.

        Returns
        -------
        {horizon_name: EvalResult}
        """
        results = {}
        for horizon, labels in labels_dict.items():
            results[horizon] = self.evaluate(
                predictions, labels, timestamps, security_ids, horizon,
            )
        return results

    @staticmethod
    def ic_decay_table(results: dict[str, EvalResult]) -> dict[str, float]:
        """Extract IC decay curve from multi-horizon results."""
        return {
            hn: r.summary.mean_ic
            for hn, r in results.items()
        }

    @staticmethod
    def print_summary(results: dict[str, EvalResult]) -> None:
        """Pretty-print evaluation results."""
        print()
        print("=" * 64)
        print("  Cross-Sectional Rank-IC Evaluation")
        print("=" * 64)
        print(f"  {'Horizon':<12s} {'Mean IC':>8s} {'ICIR':>7s} {'Win%':>7s} {'Buckets':>8s}")
        print("-" * 64)
        for hn in HORIZON_NAMES:
            if hn in results:
                r = results[hn]
                s = r.summary
                print(f"  {hn:<12s} {s.mean_ic:>+8.4f} {s.icir:>+7.2f} {s.win_rate:>6.1%} {s.n_buckets:>8d}")
            else:
                print(f"  {hn:<12s} {'--':>8s} {'--':>7s} {'--':>7s} {'--':>8s}")
        print("=" * 64)
