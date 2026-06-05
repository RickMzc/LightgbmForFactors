"""
End-to-end ML Alpha Pipeline
=============================
Wires DataLoader → LabelGenerator → FeatureFactory → AlphaModel → Evaluator.

Usage:
    python -m ml_framework.pipeline --date 20251201
    python -m ml_framework.pipeline --start 20251201 --end 20251205
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ml_framework.config import (
    HORIZONS, HORIZON_NAMES, SNAP_REQUIRED_COLS, FACTOR_CACHE_ROOT,
)
from ml_framework.data_loader import SnapDataLoader
from ml_framework.label_generator import LabelGenerator
from ml_framework.feature_factory import FeatureFactory, feature_registry
from ml_framework.modeling import AlphaModel, create_walk_forward_splits
from ml_framework.evaluation import CrossSectionalEvaluator


def get_trading_dates(start: str, end: str) -> list[str]:
    """Discover available trading dates from the data directory."""
    snap_dir = os.path.join("/fast1/user001/stock_data", "type=snap_sh")
    if not os.path.exists(snap_dir):
        return []

    date_dirs = sorted([
        d.replace("date=", "") for d in os.listdir(snap_dir)
        if d.startswith("date=")
    ])
    return [d for d in date_dirs if start <= d <= end]


def run_single_day(
    date: str,
    feature_names: list[str] | None = None,
    target_col: str = "ret_15s",
    use_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the full pipeline for a single trading day.

    Steps:
      1. Load SH+SZ snap data, align to 3s grid.
      2. Generate mid-price forward return labels.
      3. Compute (or load from cache) alpha features.

    Column requirements are gathered from the feature registry upfront
    so the parquet is scanned only once.
    """
    if feature_names is None:
        feature_names = sorted(feature_registry.keys())

    # Collect all required columns from feature registry
    all_required = set(SNAP_REQUIRED_COLS)
    for fn in feature_names:
        if fn in feature_registry:
            all_required.update(feature_registry[fn]["required_cols"])

    loader = SnapDataLoader()
    label_gen = LabelGenerator(horizons=HORIZONS)
    factory = FeatureFactory()

    t0 = time.time()

    # ---- Step 1: Load with all needed columns ----
    t1 = time.time()
    df = loader.load_day_merged(date, columns=sorted(all_required))
    load_time = time.time() - t1
    if verbose:
        print(f"[{date}] Load: {df.height:,} rows, {len(all_required)} cols ({load_time:.0f}s)", flush=True)

    # ---- Step 2: Labels ----
    t1 = time.time()
    df = label_gen.generate(df)
    label_time = time.time() - t1
    if verbose:
        valid_labels = df[target_col].drop_nulls().len()
        print(f"[{date}] Labels: {valid_labels:,}/{df.height:,} valid ({label_time:.0f}s)", flush=True)

    # ---- Step 3: Features ----
    t1 = time.time()
    df = factory.compute_many(df, feature_names, date, use_cache=use_cache)
    feat_time = time.time() - t1
    if verbose:
        print(f"[{date}] Features: {len(feature_names)} factors ({feat_time:.0f}s)", flush=True)

    n_stocks = df["SecurityID"].n_unique()
    total_time = time.time() - t0

    if verbose:
        print(f"[{date}] Done: {n_stocks} stocks, total={total_time:.0f}s", flush=True)

    return {
        "df": df,
        "feature_cols": feature_names,
        "date": date,
        "n_rows": df.height,
        "n_stocks": n_stocks,
    }


def run_baseline(
    start_date: str = "20251201",
    end_date: str = "20251201",
    feature_names: list[str] | None = None,
    target_col: str = "ret_15s",
    use_cache: bool = True,
) -> dict:
    """Run the full pipeline: load → features → labels → model → evaluate.

    This is the primary entry point for the day-1 (or multi-day) baseline.
    """
    if feature_names is None:
        feature_names = sorted(feature_registry.keys())

    dates = get_trading_dates(start_date, end_date)
    if not dates:
        print(f"No trading data found between {start_date} and {end_date}")
        sys.exit(1)

    print("=" * 64)
    print(f"  ML Alpha Pipeline: {dates[0]} → {dates[-1]} ({len(dates)} days)")
    print(f"  Features: {feature_names}")
    print(f"  Target: {target_col}")
    print("=" * 64)

    t_start = time.time()

    # ---- Stage 1-3: Data + Labels + Features ----
    all_dfs: list[pl.DataFrame] = []
    for date in dates:
        import polars as pl
        result = run_single_day(date, feature_names, target_col, use_cache)
        result["df"] = result["df"].with_columns(pl.lit(date).alias("date"))
        all_dfs.append(result["df"])

    import polars as pl
    full_df = pl.concat(all_dfs, how="vertical")
    n_total = full_df.height

    print(f"\nFull dataset: {n_total:,} rows, {full_df['date'].n_unique()} days")
    print(f"  Features: {feature_names}")
    print(f"  Target: {target_col}")

    # ---- Stage 4: Modeling ----
    print("\n--- Stage 4: Walk-Forward Training ---")
    splits = create_walk_forward_splits(dates, n_splits=1)
    if not splits:
        print("Not enough dates for walk-forward split.")
        sys.exit(1)

    model = AlphaModel(
        feature_cols=feature_names,
        target_col=target_col,
    )

    cv_results = model.fit_walk_forward(full_df, splits, verbose=True)

    # ---- Stage 5: Evaluation ----
    print("\n--- Stage 5: Evaluation ---")
    evaluator = CrossSectionalEvaluator()

    for fold_res in cv_results:
        preds = fold_res["predictions"]
        aligned_test = fold_res.get("test_df")
        if aligned_test is None:
            continue

        # Evaluate on all horizons using the aligned test DataFrame
        multi_labels = {}
        for hn in HORIZON_NAMES:
            if hn in aligned_test.columns:
                multi_labels[hn] = aligned_test[hn].to_numpy().astype(np.float64)

        results = evaluator.evaluate_multi_horizon(
            preds, multi_labels,
            aligned_test["timestamp"],
            aligned_test["SecurityID"],
        )
        evaluator.print_summary(results)

    total_time = time.time() - t_start
    print(f"\nTotal pipeline time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    return {
        "full_df": full_df,
        "cv_results": cv_results,
        "model": model,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Alpha Pipeline")
    parser.add_argument("--date", type=str, default=None,
                        help="Single date to process (YYYYMMDD)")
    parser.add_argument("--start", type=str, default="20251201",
                        help="Start date (default: 20251201)")
    parser.add_argument("--end", type=str, default="20251201",
                        help="End date (default: 20251201)")
    parser.add_argument("--target", type=str, default="ret_15s",
                        help="Target label column (default: ret_15s)")
    parser.add_argument("--features", type=str, nargs="*", default=None,
                        help="Feature names (default: all registered)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute features, ignore cache")
    args = parser.parse_args()

    start = args.date or args.start
    end = args.date or args.end

    import polars as pl  # ensure available in __main__ scope

    run_baseline(
        start_date=start,
        end_date=end,
        feature_names=args.features,
        target_col=args.target,
        use_cache=not args.no_cache,
    )
