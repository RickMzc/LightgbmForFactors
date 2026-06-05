"""
Module 4: LightGBM Modeling Pipeline
====================================
Cross-sectional Z-score normalization, stock_id as categorical,
embargo-aware walk-forward cross-validation.

No random K-fold — splits respect calendar order to prevent
look-ahead leakage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
import lightgbm as lgb

from ml_framework.config import MIN_STOCKS, EPS


@dataclass
class WalkForwardSplit:
    """Calendar-based train/val/test split with embargo gap."""
    train_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]


def create_walk_forward_splits(
    dates: list[str],
    n_splits: int = 1,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    embargo_days: int = 5,
) -> list[WalkForwardSplit]:
    """Create embargo-aware walk-forward date splits.

    Parameters
    ----------
    dates : sorted list of date strings.
    n_splits : number of walk-forward folds.
    train_ratio : fraction for training.
    val_ratio : fraction for validation.
    embargo_days : gap days between train and val (excluded).

    Returns
    -------
    List of WalkForwardSplit, each with non-overlapping test sets.
    """
    n = len(dates)
    splits = []

    # For very small date ranges, embargo proportionally
    effective_embargo = min(embargo_days, max(1, n // 20))

    test_size = n - int(n * (train_ratio + val_ratio))
    if test_size < 1:
        test_size = max(1, n // 5)

    for fold in range(n_splits):
        # Expanding window: train grows, val+test slide forward
        test_end = n - fold * (test_size // max(n_splits, 1))
        test_start = max(test_end - test_size, 0)

        val_end = max(test_start - effective_embargo, 0)
        val_start = max(val_end - int(n * val_ratio), 0)

        train_end = max(val_start - effective_embargo, 0)
        train_start = 0

        if train_end <= train_start or val_end <= val_start or test_start >= test_end:
            continue

        splits.append(WalkForwardSplit(
            train_dates=dates[train_start:train_end],
            val_dates=dates[val_start:val_end],
            test_dates=dates[test_start:test_end],
        ))

    return splits


class AlphaModel:
    """LightGBM-based alpha model with walk-forward CV.

    Parameters
    ----------
    feature_cols : feature column names in the input DataFrame.
    target_col : label column name (e.g. "ret_15s").
    lgb_params : dict of LightGBM parameters (overrides defaults).
    """

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str = "ret_15s",
        lgb_params: dict | None = None,
        random_state: int = 42,
    ) -> None:
        self._feature_cols = list(feature_cols)
        self._target_col = target_col
        self._random_state = random_state

        # Default LightGBM params for regression
        self._lgb_params: dict = {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 100,
            "min_sum_hessian_in_leaf": 1e-3,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbosity": -1,
            "num_threads": 4,
            "seed": random_state,
        }
        if lgb_params:
            self._lgb_params.update(lgb_params)

        self._model: lgb.Booster | None = None
        self._feature_importance: dict[str, float] = {}

    # ---- Preprocessing ----

    def preprocess(
        self, df: pl.DataFrame, fit: bool = True,
        stats: dict | None = None,
    ) -> tuple[pl.DataFrame, dict]:
        """Cross-sectional Z-score normalization per timestamp.

        For each feature, within each timestamp slice:
          z = (x - mean) / (std + EPS)

        Parameters
        ----------
        df : DataFrame with timestamp, feature columns.
        fit : if True, compute stats from df. If False, apply given stats.
        stats : pre-computed {feature: {mean_col, std_col}} for transform.

        Returns
        -------
        (normalized_df, stats_dict)
        """
        if fit:
            stats = {}
            for col in self._feature_cols:
                grp = df.group_by("timestamp").agg([
                    pl.col(col).mean().alias("_mean"),
                    pl.col(col).std().alias("_std"),
                ])
                stats[col] = grp

            # Join means and stds back
            for col in self._feature_cols:
                df = df.join(stats[col], on="timestamp", how="left")
                df = df.with_columns(
                    ((pl.col(col) - pl.col("_mean")) / (pl.col("_std") + EPS)).alias(col)
                )
                df = df.drop(["_mean", "_std"])

            return df, stats
        else:
            if stats is None:
                raise ValueError("stats dict required when fit=False")
            for col in self._feature_cols:
                df = df.join(stats[col], on="timestamp", how="left")
                df = df.with_columns(
                    ((pl.col(col) - pl.col("_mean")) / (pl.col("_std") + EPS)).alias(col)
                )
                df = df.drop(["_mean", "_std"])
            return df, stats

    # ---- Data preparation ----

    def prepare_data(
        self, df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
        """Extract feature matrix, target vector, stock IDs, and filtered df.

        Caller should pass a DataFrame already filtered to non-null
        rows for features + target.  Returns aligned numpy arrays
        and the (subset) DataFrame for downstream evaluation.

        Returns
        -------
        (X, y, stock_ids, filtered_df) — X/y/stock_ids are aligned
        to filtered_df row order.
        """
        valid = df.drop_nulls(subset=[self._target_col] + self._feature_cols)

        X = valid.select(self._feature_cols).to_numpy().astype(np.float64)
        y = valid.select(self._target_col).to_numpy().ravel().astype(np.float64)

        stock_ids = (
            valid.select(pl.col("SecurityID").cast(pl.Categorical).to_physical())
            .to_numpy()
            .ravel()
            .astype(np.int32)
        )

        return X, y, stock_ids, valid

    # ---- Training ----

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
    ) -> lgb.Booster:
        """Train a single LightGBM model.

        If 'stock_id' is in feature_cols, it is treated as a categorical
        feature. Otherwise all features are continuous.
        """
        if "stock_id" in self._feature_cols:
            cat_idx = self._feature_cols.index("stock_id")
            cat_feature = [cat_idx]
        else:
            cat_feature = "auto"

        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_feature,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val, label=y_val,
                categorical_feature=cat_feature,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("val")

        self._model = lgb.train(
            params=self._lgb_params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(verbose_eval),
            ],
        )

        # Store feature importance
        importance = self._model.feature_importance(importance_type="gain")
        self._feature_importance = dict(
            zip(self._feature_cols, importance)
        )

        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the trained model."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self._model.predict(X).astype(np.float64)

    def fit_walk_forward(
        self,
        df: pl.DataFrame,
        splits: list[WalkForwardSplit],
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
    ) -> list[dict]:
        """Run full walk-forward CV.

        For each split: normalize, train, predict on test, store results.

        Returns a list of dicts, each with: fold, train_dates, val_dates,
        test_dates, model, predictions, test_timestamps, test_stock_ids.
        """
        results: list[dict] = []

        for fold_idx, split in enumerate(splits):
            t0 = time.time()

            # Prepare train/val/test DataFrames
            train_df = df.filter(pl.col("date").is_in(split.train_dates))
            val_df = df.filter(pl.col("date").is_in(split.val_dates))
            test_df = df.filter(pl.col("date").is_in(split.test_dates))

            if train_df.height == 0 or test_df.height == 0:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: empty split, skip", flush=True)
                continue

            # Preprocess (Z-score normalize)
            train_df, stats = self.preprocess(train_df, fit=True)
            val_df, _ = self.preprocess(val_df, fit=False, stats=stats)
            test_df, _ = self.preprocess(test_df, fit=False, stats=stats)

            # Extract arrays
            X_train, y_train, _, _ = self.prepare_data(train_df)
            X_val, y_val = None, None
            if val_df.height > 0:
                X_val, y_val, _, _ = self.prepare_data(val_df)
            X_test, y_test, _, test_df_aligned = self.prepare_data(test_df)

            # Train
            model = self.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0 if not verbose else 100,
            )

            # Predict
            preds = self.predict(X_test)

            fold_result = {
                "fold": fold_idx + 1,
                "train_dates": split.train_dates,
                "val_dates": split.val_dates,
                "test_dates": split.test_dates,
                "model": model,
                "predictions": preds,
                "labels": y_test,
                "X_test": X_test,
                "test_df": test_df_aligned,
            }
            results.append(fold_result)

            elapsed = time.time() - t0
            if verbose:
                n_train = len(split.train_dates)
                n_test = len(split.test_dates)
                print(
                    f"  Fold {fold_idx + 1}: train={n_train}d, test={n_test}d, "
                    f"{X_test.shape[0]:,} rows, {elapsed:.0f}s",
                    flush=True,
                )

        return results

    @property
    def feature_importance(self) -> dict[str, float]:
        return dict(self._feature_importance)

    @property
    def model(self) -> lgb.Booster | None:
        return self._model
