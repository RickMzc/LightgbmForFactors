"""
Module 3: Feature Factory & Cache
==================================
Decorator-based feature registry + disk-backed cache.

Features are registered with @register(name, required_cols).
FeatureFactory orchestrates computation, caching, and joining.

Cache format (per factor per date):
  /fast1/user001/factor_values/{factor_name}/{date}.parquet
  Columns: [timestamp (Int32), SecurityID (str), value (Float64)]

Cache joins use exact match on [timestamp, SecurityID] — NEVER
join_asof on cached factors.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import polars as pl

from ml_framework.config import FACTOR_CACHE_ROOT


# ============================================================
# Global feature registry
# ============================================================

feature_registry: dict[str, dict[str, Any]] = {}


def register(name: str, required_cols: list[str]) -> Callable:
    """Decorator to register a feature computation function.

    Usage:

        @register("OBI", required_cols=["BidVolume1", "AskVolume1"])
        def compute_obi(df: pl.DataFrame) -> pl.DataFrame:
            ...
    """
    def decorator(func: Callable) -> Callable:
        feature_registry[name] = {
            "func": func,
            "required_cols": required_cols,
        }
        return func

    return decorator


# ============================================================
# Helpers
# ============================================================

def _time_group(df: pl.DataFrame) -> list[str]:
    """Group key for per-stock time-series ops.  Includes 'date' when
    available so that shift/ewm/rolling reset at day boundaries.  Single-day
    data (run_single_day) has no date column → group is just SecurityID."""
    return ["SecurityID", "date"] if "date" in df.columns else ["SecurityID"]


def _validate_cols(df: pl.DataFrame, factor_name: str, required: list[str]) -> None:
    """Raise if required columns are missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Factor '{factor_name}' missing required columns: {missing}. "
            f"Available: {sorted(df.columns)}"
        )


# ============================================================
# Pre-registered features
# ============================================================

@register("OBI", required_cols=["BidVolume1", "AskVolume1"])
def _compute_obi(df: pl.DataFrame) -> pl.DataFrame:
    """Order Book Imbalance: (BV1 - SV1) / (BV1 + SV1)."""
    return df.with_columns(
        pl.when((pl.col("BidVolume1") + pl.col("AskVolume1")) > 0)
        .then(
            (pl.col("BidVolume1") - pl.col("AskVolume1"))
            / (pl.col("BidVolume1") + pl.col("AskVolume1"))
        )
        .otherwise(0.0)
        .alias("OBI")
    )


# ============================================================
# OFI — Cont, Kukanov & Stoikov (2014), J. Financial Econometrics 12(1):47-88
# ============================================================
# Correct OFI MUST track price changes at each level.  A pure volume
# diff confuses price-improving orders with order cancellations.
#
# Per level i:
#   Bid:
#     Price up   → e = BV_t          (new orders at higher price)
#     Price flat → e = BV_t - BV_t-1 (net flow)
#     Price down → e = -BV_t-1       (old orders consumed)
#   Ask:
#     Price up   → f = -SV_t-1       (old offers consumed)
#     Price flat → f = SV_t - SV_t-1 (net flow)
#     Price down → f = SV_t          (new offers at lower price)
#   OFI_i = e - f
#   OFI    = Σ_i OFI_i / Σ_i (BV_i + SV_i)  [normalized]


def _ofi_per_level(bp: pl.Expr, bv: pl.Expr, ap: pl.Expr, sv: pl.Expr,
                   group: list[str]) -> pl.Expr:
    """Single-level price-aware OFI expression."""
    bp_prev = bp.shift(1).over(group)
    bv_prev = bv.shift(1).over(group)
    ap_prev = ap.shift(1).over(group)
    sv_prev = sv.shift(1).over(group)

    e = (
        pl.when(bp > bp_prev)
        .then(bv)
        .when(bp == bp_prev)
        .then(bv - bv_prev)
        .when(bp < bp_prev)
        .then(-bv_prev)
        .otherwise(0.0)
    )

    f = (
        pl.when(ap > ap_prev)
        .then(-sv_prev)
        .when(ap == ap_prev)
        .then(sv - sv_prev)
        .when(ap < ap_prev)
        .then(sv)
        .otherwise(0.0)
    )

    return e - f


_OFI5_P_COLS = [f"BidPrice{i}" for i in range(1, 6)] + \
               [f"AskPrice{i}" for i in range(1, 6)]
_OFI5_V_COLS = [f"BidVolume{i}" for i in range(1, 6)] + \
               [f"AskVolume{i}" for i in range(1, 6)]
_OFI5_COLS = _OFI5_P_COLS + _OFI5_V_COLS


@register("OFI", required_cols=_OFI5_COLS)
def _compute_ofi(df: pl.DataFrame) -> pl.DataFrame:
    """Price-aware OFI across 5 levels (Cont-Kukanov-Stoikov 2014).

    Sums per-level OFI, normalizes by total depth.
    """
    ofi_exprs = []
    total_vol_exprs = []
    for i in range(1, 6):
        bp = pl.col(f"BidPrice{i}")
        bv = pl.col(f"BidVolume{i}")
        ap = pl.col(f"AskPrice{i}")
        sv = pl.col(f"AskVolume{i}")
        g = _time_group(df)
        ofi_exprs.append(_ofi_per_level(bp, bv, ap, sv, g))
        total_vol_exprs.append(bv + sv)

    raw = pl.sum_horizontal(ofi_exprs)
    total = pl.sum_horizontal(total_vol_exprs)
    return df.with_columns(
        pl.when(total > 0).then(raw / total).otherwise(0.0).alias("OFI")
    )


# OFI_1: single-level version
_OFI1_COLS = ["BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]


@register("OFI_1", required_cols=_OFI1_COLS)
def _compute_ofi_1(df: pl.DataFrame) -> pl.DataFrame:
    g = _time_group(df)
    raw = _ofi_per_level(
        pl.col("BidPrice1"), pl.col("BidVolume1"),
        pl.col("AskPrice1"), pl.col("AskVolume1"), g,
    )
    total = pl.col("BidVolume1") + pl.col("AskVolume1")
    return df.with_columns(
        pl.when(total > 0).then(raw / total).otherwise(0.0).alias("OFI_1")
    )


# OFI_10: 10-level version
_OFI10_P_COLS = [f"BidPrice{i}" for i in range(1, 11)] + \
                [f"AskPrice{i}" for i in range(1, 11)]
_OFI10_V_COLS = [f"BidVolume{i}" for i in range(1, 11)] + \
                [f"AskVolume{i}" for i in range(1, 11)]
_OFI10_COLS = _OFI10_P_COLS + _OFI10_V_COLS


@register("OFI_10", required_cols=_OFI10_COLS)
def _compute_ofi_10(df: pl.DataFrame) -> pl.DataFrame:
    g = _time_group(df)
    ofi_exprs = []
    total_vol_exprs = []
    for i in range(1, 11):
        bp = pl.col(f"BidPrice{i}")
        bv = pl.col(f"BidVolume{i}")
        ap = pl.col(f"AskPrice{i}")
        sv = pl.col(f"AskVolume{i}")
        ofi_exprs.append(_ofi_per_level(bp, bv, ap, sv, g))
        total_vol_exprs.append(bv + sv)

    raw = pl.sum_horizontal(ofi_exprs)
    total = pl.sum_horizontal(total_vol_exprs)
    return df.with_columns(
        pl.when(total > 0).then(raw / total).otherwise(0.0).alias("OFI_10")
    )


# OFI_3: 3-level version
_OFI3_P_COLS = [f"BidPrice{i}" for i in range(1, 4)] + \
               [f"AskPrice{i}" for i in range(1, 4)]
_OFI3_V_COLS = [f"BidVolume{i}" for i in range(1, 4)] + \
               [f"AskVolume{i}" for i in range(1, 4)]
_OFI3_COLS = _OFI3_P_COLS + _OFI3_V_COLS


@register("OFI_3", required_cols=_OFI3_COLS)
def _compute_ofi_3(df: pl.DataFrame) -> pl.DataFrame:
    g = _time_group(df)
    ofi_exprs = []
    total_vol_exprs = []
    for i in range(1, 4):
        bp = pl.col(f"BidPrice{i}")
        bv = pl.col(f"BidVolume{i}")
        ap = pl.col(f"AskPrice{i}")
        sv = pl.col(f"AskVolume{i}")
        ofi_exprs.append(_ofi_per_level(bp, bv, ap, sv, g))
        total_vol_exprs.append(bv + sv)
    raw = pl.sum_horizontal(ofi_exprs)
    total = pl.sum_horizontal(total_vol_exprs)
    return df.with_columns(
        pl.when(total > 0).then(raw / total).otherwise(0.0).alias("OFI_3")
    )


# ---- Rolling OFI features (intraday EMA, reset per stock per day) ----

def _ensure_ofi(df: pl.DataFrame) -> pl.DataFrame:
    """Compute OFI if not already present."""
    if "OFI" not in df.columns:
        df = _compute_ofi(df)
    return df


@register("OFI_MA", required_cols=_OFI5_COLS)
def _compute_ofi_ma(df: pl.DataFrame) -> pl.DataFrame:
    """EMA of OFI (span=20 ≈ 60s), per stock per day.

    Group key includes date when available to prevent cross-day carry.
    In single-day mode (run_single_day), .over("SecurityID") alone suffices.
    """
    df = _ensure_ofi(df)
    alpha = 2.0 / (20.0 + 1.0)
    group = ["SecurityID", "date"] if "date" in df.columns else "SecurityID"
    return df.with_columns(
        pl.col("OFI").ewm_mean(alpha=alpha, adjust=False, min_periods=1)
        .over(group).alias("OFI_MA")
    )


@register("OFI_Z", required_cols=_OFI5_COLS)
def _compute_ofi_z(df: pl.DataFrame) -> pl.DataFrame:
    """Z-score: (OFI - OFI_MA) / rolling_std. EMA span=20."""
    df = _ensure_ofi(df)
    alpha = 2.0 / (20.0 + 1.0)
    group = ["SecurityID", "date"] if "date" in df.columns else "SecurityID"
    ema = pl.col("OFI").ewm_mean(alpha=alpha, adjust=False, min_periods=1).over(group)
    diff = pl.col("OFI") - ema
    emastd = diff.pow(2).ewm_mean(alpha=alpha, adjust=False, min_periods=1).over(group).sqrt()
    return df.with_columns(
        pl.when(emastd > 1e-8).then(diff / emastd).otherwise(0.0).alias("OFI_Z")
    )


@register("OFI_Decay", required_cols=_OFI5_COLS)
def _compute_ofi_decay(df: pl.DataFrame) -> pl.DataFrame:
    """Decay-weighted OFI: Σ_{k=0..4} OFI_{t−k} × exp(−k) (≈15s window)."""
    import numpy as np
    df = _ensure_ofi(df)
    ofi = pl.col("OFI")
    group = ["SecurityID", "date"] if "date" in df.columns else "SecurityID"
    wsum = pl.lit(0.0)
    for k in range(5):
        term = ofi.shift(k).over(group).fill_null(0.0) * np.exp(-k)
        wsum = wsum + term
    return df.with_columns(wsum.alias("OFI_Decay"))


_TS_COLS = ["BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]


@register("TS_Imbalance", required_cols=_TS_COLS)
def _compute_ts_imbalance(df: pl.DataFrame) -> pl.DataFrame:
    """Price-aware signed flow at level 1."""
    g = _time_group(df)
    e = (
        pl.when(pl.col("BidPrice1") > pl.col("BidPrice1").shift(1).over(g))
        .then(pl.col("BidVolume1"))
        .when(pl.col("BidPrice1") == pl.col("BidPrice1").shift(1).over(g))
        .then(pl.col("BidVolume1") - pl.col("BidVolume1").shift(1).over(g))
        .when(pl.col("BidPrice1") < pl.col("BidPrice1").shift(1).over(g))
        .then(-pl.col("BidVolume1").shift(1).over(g))
        .otherwise(0.0)
    )
    f = (
        pl.when(pl.col("AskPrice1") > pl.col("AskPrice1").shift(1).over(g))
        .then(-pl.col("AskVolume1").shift(1).over(g))
        .when(pl.col("AskPrice1") == pl.col("AskPrice1").shift(1).over(g))
        .then(pl.col("AskVolume1") - pl.col("AskVolume1").shift(1).over(g))
        .when(pl.col("AskPrice1") < pl.col("AskPrice1").shift(1).over(g))
        .then(pl.col("AskVolume1"))
        .otherwise(0.0)
    )

    raw = e - f
    denom = e.abs() + f.abs()
    return df.with_columns(
        pl.when(denom > 0).then(raw / denom).otherwise(0.0).alias("TS_Imbalance")
    )


@register("Vol_Spread", required_cols=[
    "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
    "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
])
def _compute_vol_spread(df: pl.DataFrame) -> pl.DataFrame:
    """Volume spread: ratio of total bid depth to total ask depth (log)."""
    total_bid = (
        pl.col("BidVolume1") + pl.col("BidVolume2") + pl.col("BidVolume3")
        + pl.col("BidVolume4") + pl.col("BidVolume5")
    )
    total_ask = (
        pl.col("AskVolume1") + pl.col("AskVolume2") + pl.col("AskVolume3")
        + pl.col("AskVolume4") + pl.col("AskVolume5")
    )

    return df.with_columns(
        pl.when((total_bid > 0) & (total_ask > 0))
        .then((total_bid / total_ask).log())
        .otherwise(None)
        .alias("Vol_Spread")
    )


_DEPTH_IMB_COLS = (
    [f"BidPrice{i}" for i in range(1, 6)]
    + [f"AskPrice{i}" for i in range(1, 6)]
    + [f"BidVolume{i}" for i in range(1, 6)]
    + [f"AskVolume{i}" for i in range(1, 6)]
)


@register("Depth_Imbalance", required_cols=_DEPTH_IMB_COLS)
def _compute_depth_imbalance(df: pl.DataFrame) -> pl.DataFrame:
    """Weighted depth imbalance with exponential distance decay.

    Bid side:  weight_i = exp(-κ × (BP1 - BP_i) / TickSize)
    Ask side:  weight_i = exp(-κ × (AP_i - AP1) / TickSize)

    Intuition: orders further from the best price are more likely to
    be spoofed / non-executable — their weight decays exponentially.

    κ = 1.0, TickSize = 0.01 (A-share standard).
    """
    kappa = pl.lit(1.0)
    tick = pl.lit(0.01)
    bp1 = pl.col("BidPrice1")
    ap1 = pl.col("AskPrice1")

    bid_w = pl.lit(0.0)
    ask_w = pl.lit(0.0)

    for i in range(1, 6):
        # Distance from best price (≥ 0)
        bid_dist = (bp1 - pl.col(f"BidPrice{i}")).clip(0) / tick
        ask_dist = (pl.col(f"AskPrice{i}") - ap1).clip(0) / tick

        w_bid = (-kappa * bid_dist).exp()
        w_ask = (-kappa * ask_dist).exp()

        bid_w = bid_w + pl.col(f"BidVolume{i}") * w_bid
        ask_w = ask_w + pl.col(f"AskVolume{i}") * w_ask

    return df.with_columns(
        pl.when((bid_w + ask_w) > 0)
        .then((bid_w - ask_w) / (bid_w + ask_w))
        .otherwise(0.0)
        .alias("Depth_Imbalance")
    )


# ============================================================
# Category ①: 盘口静态结构 (single-snapshot LOB features)
# ============================================================

# --- Spread (买卖价差) ---

_SPREAD_COLS = ["BidPrice1", "AskPrice1"]


@register("Spread", required_cols=_SPREAD_COLS)
def _compute_spread(df: pl.DataFrame) -> pl.DataFrame:
    """Absolute bid-ask spread: AskPrice1 - BidPrice1 (in price units)."""
    return df.with_columns(
        pl.when(
            (pl.col("BidPrice1") > 0) & (pl.col("AskPrice1") > 0)
        )
        .then(pl.col("AskPrice1") - pl.col("BidPrice1"))
        .otherwise(None)
        .alias("Spread")
    )


@register("SpreadRel", required_cols=_SPREAD_COLS)
def _compute_spread_rel(df: pl.DataFrame) -> pl.DataFrame:
    """Relative spread: (AskP1 - BidP1) / MidPrice, dimensionless."""
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2.0
    return df.with_columns(
        pl.when(
            (pl.col("BidPrice1") > 0) & (pl.col("AskPrice1") > 0) & (mid > 0)
        )
        .then((pl.col("AskPrice1") - pl.col("BidPrice1")) / mid)
        .otherwise(None)
        .alias("SpreadRel")
    )


# --- MicroPrice (Stoikov 2017, SSRN:2970694) ---

_MICRO_PRICE_COLS = ["BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]


@register("MicroPrice", required_cols=_MICRO_PRICE_COLS)
def _compute_micro_price(df: pl.DataFrame) -> pl.DataFrame:
    """Stoikov micro-price: (AskP1*BV1 + BidP1*SV1) / (BV1+SV1).

    Weights each price by the OPPOSING side's volume:
      - Ask price weighted by bid volume (bid depth → fair tilts toward ask)
      - Bid price weighted by ask volume (ask depth → fair tilts toward bid)
    """
    bp1 = pl.col("BidPrice1")
    ap1 = pl.col("AskPrice1")
    bv1 = pl.col("BidVolume1")
    sv1 = pl.col("AskVolume1")
    denom = bv1 + sv1
    return df.with_columns(
        pl.when((bp1 > 0) & (ap1 > 0) & (denom > 0))
        .then((ap1 * bv1 + bp1 * sv1) / denom)
        .otherwise(None)
        .alias("MicroPrice")
    )


@register("MicroPriceBias", required_cols=_MICRO_PRICE_COLS)
def _compute_micro_price_bias(df: pl.DataFrame) -> pl.DataFrame:
    """MicroPrice / MidPrice - 1.  Positive → fair value above mid (bullish)."""
    bp1 = pl.col("BidPrice1")
    ap1 = pl.col("AskPrice1")
    bv1 = pl.col("BidVolume1")
    sv1 = pl.col("AskVolume1")
    mid = (bp1 + ap1) / 2.0
    denom = bv1 + sv1
    mp = pl.when((bp1 > 0) & (ap1 > 0) & (denom > 0)).then(
        (ap1 * bv1 + bp1 * sv1) / denom
    ).otherwise(None)
    return df.with_columns(
        pl.when(mp.is_not_null() & (mid > 0))
        .then(mp / mid - 1.0)
        .otherwise(None)
        .alias("MicroPriceBias")
    )


# --- Order Count Imbalance (A-share specific: 笔数信息比金额更"干净") ---

_OCI_COLS_L1 = ["NumOrdersB1", "NumOrdersS1"]
_OCI_COLS_L5 = [f"NumOrdersB{i}" for i in range(1, 6)] + \
               [f"NumOrdersS{i}" for i in range(1, 6)]
_OCI_COLS_L10 = [f"NumOrdersB{i}" for i in range(1, 11)] + \
                [f"NumOrdersS{i}" for i in range(1, 11)]


def _oci_expr(bid_cols: list[str], ask_cols: list[str]) -> pl.Expr:
    """Generic OCI: (ΣNumB - ΣNumS) / (ΣNumB + ΣNumS)."""
    sum_b = sum(pl.col(c) for c in bid_cols)
    sum_s = sum(pl.col(c) for c in ask_cols)
    denom = sum_b + sum_s
    return pl.when(denom > 0).then((sum_b - sum_s) / denom).otherwise(0.0)


@register("OCIB_1", required_cols=_OCI_COLS_L1)
def _compute_ocib_1(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_oci_expr(["NumOrdersB1"], ["NumOrdersS1"]).alias("OCIB_1"))


@register("OCIB_5", required_cols=_OCI_COLS_L5)
def _compute_ocib_5(df: pl.DataFrame) -> pl.DataFrame:
    bc = [f"NumOrdersB{i}" for i in range(1, 6)]
    ac = [f"NumOrdersS{i}" for i in range(1, 6)]
    return df.with_columns(_oci_expr(bc, ac).alias("OCIB_5"))


@register("OCIB_10", required_cols=_OCI_COLS_L10)
def _compute_ocib_10(df: pl.DataFrame) -> pl.DataFrame:
    bc = [f"NumOrdersB{i}" for i in range(1, 11)]
    ac = [f"NumOrdersS{i}" for i in range(1, 11)]
    return df.with_columns(_oci_expr(bc, ac).alias("OCIB_10"))


# --- BookSlope: 累计量对价格的回归斜率, 刻画深度衰减 ---

_BOOK_SLOPE_COLS = (
    [f"BidPrice{i}" for i in range(1, 6)]
    + [f"AskPrice{i}" for i in range(1, 6)]
    + [f"BidVolume{i}" for i in range(1, 6)]
    + [f"AskVolume{i}" for i in range(1, 6)]
)


def _book_slope_5lev(price_cols: list[str], vol_cols: list[str]) -> pl.Expr:
    """Linear regression slope of cumulative volume vs price across 5 levels.

    Y_i = Σ_{j=1..i} Vol_j  (cumulative volume)
    X_i = Price_i

    slope = cov(X,Y)/var(X) = (5*ΣXY - ΣX*ΣY) / (5*ΣX² - (ΣX)²)

    For bid side: prices decrease (BP1 > BP2 > ...), slope is naturally negative.
    The caller should negate if comparing with ask side.
    """
    # Cumulative volumes
    cv = [sum(pl.col(vol_cols[j]) for j in range(i + 1)) for i in range(5)]

    # ΣY = Σ cumulative volumes
    sum_y = sum(cv)

    # ΣX = Σ prices
    sum_x = sum(pl.col(p) for p in price_cols)

    # ΣXY = Σ price_i * cum_vol_i
    sum_xy = sum(pl.col(price_cols[i]) * cv[i] for i in range(5))

    # ΣX² = Σ price_i²
    sum_x2 = sum(pl.col(p) ** 2 for p in price_cols)

    n = pl.lit(5.0)
    denom = n * sum_x2 - sum_x * sum_x

    return pl.when(denom.abs() > 1e-12).then(
        (n * sum_xy - sum_x * sum_y) / denom
    ).otherwise(None)


@register("BookSlope", required_cols=_BOOK_SLOPE_COLS)
def _compute_book_slope(df: pl.DataFrame) -> pl.DataFrame:
    """Bid slope (negated) minus ask slope, normalized by total depth.

    Positive → bid-side book is steeper (stronger support).
    Raw slope has extreme magnitude (volume/price units), so we divide
    by total depth × mid_price to get a dimensionless measure.
    """
    bid_p = [f"BidPrice{i}" for i in range(1, 6)]
    ask_p = [f"AskPrice{i}" for i in range(1, 6)]
    bid_v = [f"BidVolume{i}" for i in range(1, 6)]
    ask_v = [f"AskVolume{i}" for i in range(1, 6)]

    bid_slope = _book_slope_5lev(bid_p, bid_v)
    ask_slope = _book_slope_5lev(ask_p, ask_v)

    total_depth = sum(pl.col(v) for v in bid_v) + sum(pl.col(v) for v in ask_v)
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2.0

    # dimensionless: slope * mid / depth  (not slope / (depth * mid) which has 1/price² bias)
    return df.with_columns(
        pl.when(total_depth > 0)
        .then(((-bid_slope) - ask_slope) * mid / total_depth)
        .otherwise(None)
        .alias("BookSlope")
    )


# --- MaxDurPressure: 最长挂单持续时间的变化量 ---

_MAXDUR_COLS = ["MaxBidDur", "MaxSellDur"]


@register("MaxDurPressure", required_cols=_MAXDUR_COLS)
def _compute_max_dur_pressure(df: pl.DataFrame) -> pl.DataFrame:
    """Patient-capital flow: delta of MaxBidDur vs MaxSellDur.

    MaxBidDur / MaxSellDur are cumulative (monotonically increasing since open).
    ΔDur = Dur[t] - Dur[t-1] captures which side gained long-standing orders
    in this 3s window.

    Long-standing orders → patient capital (less likely to be noise).
    """
    g = _time_group(df)
    d_bid = pl.col("MaxBidDur") - pl.col("MaxBidDur").shift(1).over(g)
    d_sell = pl.col("MaxSellDur") - pl.col("MaxSellDur").shift(1).over(g)

    # SZ market has no MaxBidDur/MaxSellDur → output null, not 0.
    # 0 would imply "balanced patient capital" when it actually means "no data".
    is_sz = pl.col("MaxBidDur").is_null() | pl.col("MaxSellDur").is_null()
    denom = d_bid + d_sell
    return df.with_columns(
        pl.when(is_sz)
        .then(None)
        .when(denom > 0)
        .then((d_bid - d_sell) / denom)
        .otherwise(0.0)
        .alias("MaxDurPressure")
    )


# ============================================================
# ③ Multi-level OBI & Amount OBI
# ============================================================

# OBI_k: simple volume imbalance across k levels
# Amount_OBI_k: price×volume weighted across k levels

def _obi_k_expr(k: int) -> pl.Expr:
    """OBI across k levels: (ΣBV - ΣSV) / (ΣBV + ΣSV)."""
    sum_b = sum(pl.col(f"BidVolume{i}") for i in range(1, k + 1))
    sum_s = sum(pl.col(f"AskVolume{i}") for i in range(1, k + 1))
    denom = sum_b + sum_s
    return pl.when(denom > 0).then((sum_b - sum_s) / denom).otherwise(0.0)


def _amount_obi_k_expr(k: int) -> pl.Expr:
    """Amount-based OBI: (Σ BV×BP - Σ SV×AP) / (Σ BV×BP + Σ SV×AP)."""
    sum_b_amt = sum(pl.col(f"BidVolume{i}") * pl.col(f"BidPrice{i}") for i in range(1, k + 1))
    sum_s_amt = sum(pl.col(f"AskVolume{i}") * pl.col(f"AskPrice{i}") for i in range(1, k + 1))
    denom = sum_b_amt + sum_s_amt
    return pl.when(denom > 0).then((sum_b_amt - sum_s_amt) / denom).otherwise(0.0)


_OBI_K_COLS_1 = ["BidVolume1", "AskVolume1"]
_OBI_K_COLS_3 = [f"BidVolume{i}" for i in range(1, 4)] + [f"AskVolume{i}" for i in range(1, 4)]
_OBI_K_COLS_5 = [f"BidVolume{i}" for i in range(1, 6)] + [f"AskVolume{i}" for i in range(1, 6)]
_OBI_K_COLS_10 = [f"BidVolume{i}" for i in range(1, 11)] + [f"AskVolume{i}" for i in range(1, 11)]
_AMT_OBI_1_COLS = ["BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]
_AMT_OBI_3_COLS = [f"BidPrice{i}" for i in range(1, 4)] + [f"AskPrice{i}" for i in range(1, 4)] + _OBI_K_COLS_3
_AMT_OBI_5_COLS = [f"BidPrice{i}" for i in range(1, 6)] + [f"AskPrice{i}" for i in range(1, 6)] + _OBI_K_COLS_5
_AMT_OBI_10_COLS = [f"BidPrice{i}" for i in range(1, 11)] + [f"AskPrice{i}" for i in range(1, 11)] + _OBI_K_COLS_10


@register("OBI_3", required_cols=_OBI_K_COLS_3)
def _compute_obi_3(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_obi_k_expr(3).alias("OBI_3"))


@register("OBI_5", required_cols=_OBI_K_COLS_5)
def _compute_obi_5(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_obi_k_expr(5).alias("OBI_5"))


@register("OBI_10", required_cols=_OBI_K_COLS_10)
def _compute_obi_10(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_obi_k_expr(10).alias("OBI_10"))


@register("AmtOBI_1", required_cols=_AMT_OBI_1_COLS)
def _compute_amt_obi_1(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_amount_obi_k_expr(1).alias("AmtOBI_1"))


@register("AmtOBI_3", required_cols=_AMT_OBI_3_COLS)
def _compute_amt_obi_3(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_amount_obi_k_expr(3).alias("AmtOBI_3"))


@register("AmtOBI_5", required_cols=_AMT_OBI_5_COLS)
def _compute_amt_obi_5(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_amount_obi_k_expr(5).alias("AmtOBI_5"))


@register("AmtOBI_10", required_cols=_AMT_OBI_10_COLS)
def _compute_amt_obi_10(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_amount_obi_k_expr(10).alias("AmtOBI_10"))


# ============================================================
# ④ 盘口形状特征
# ============================================================

# --- DepthConcentration: 最优档深度占总深度的比例 ---

_DEPTH_CONC_COLS = (
    [f"BidVolume{i}" for i in range(1, 11)]
    + [f"AskVolume{i}" for i in range(1, 11)]
)


@register("DepthConcentration", required_cols=_DEPTH_CONC_COLS)
def _compute_depth_concentration(df: pl.DataFrame) -> pl.DataFrame:
    """(BV1/ΣBV_1..10) - (SV1/ΣSV_1..10).

    Positive → bid depth is more concentrated at the top (stronger
    immediate support). Negative → ask side more concentrated.
    """
    total_bid = sum(pl.col(f"BidVolume{i}") for i in range(1, 11))
    total_ask = sum(pl.col(f"AskVolume{i}") for i in range(1, 11))
    bid_conc = pl.when(total_bid > 0).then(pl.col("BidVolume1") / total_bid).otherwise(0.0)
    ask_conc = pl.when(total_ask > 0).then(pl.col("AskVolume1") / total_ask).otherwise(0.0)
    return df.with_columns((bid_conc - ask_conc).alias("DepthConcentration"))


@register("TopDepthRatio", required_cols=_DEPTH_CONC_COLS)
def _compute_top_depth_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """(BV1 + SV1) / (ΣBV_1..10 + ΣSV_1..10).

    最优档量占十档总量的比例。越高 → 盘口越集中，越容易被击穿。
    """
    top = pl.col("BidVolume1") + pl.col("AskVolume1")
    total = sum(pl.col(f"BidVolume{i}") + pl.col(f"AskVolume{i}") for i in range(1, 11))
    return df.with_columns(
        pl.when(total > 0).then(top / total).otherwise(0.0).alias("TopDepthRatio")
    )


# --- BookConvexity: 累计量曲线的二阶导（凸性） ---

_BOOK_CURVE_COLS = (
    [f"BidPrice{i}" for i in range(1, 6)]
    + [f"AskPrice{i}" for i in range(1, 6)]
    + [f"BidVolume{i}" for i in range(1, 6)]
    + [f"AskVolume{i}" for i in range(1, 6)]
)


def _cumvol_convexity(p_cols: list[str], v_cols: list[str],
                      p1: pl.Expr, p3: pl.Expr, p5: pl.Expr) -> pl.Expr:
    """Discrete 2nd derivative of cumulative volume vs price.

    CBV_i = Σ_{j=1..i} Vol_j

    Segment slope from level 1→3:  (CBV_3 - CBV_1) / (P1 - P3)
    Segment slope from level 3→5:  (CBV_5 - CBV_3) / (P3 - P5)

    Convexity = slope_35 - slope_13

    Positive → depth accumulation ACCELERATES at outer levels
    (concave shape: thin near top, thick at distance — "wall further out").
    Negative → depth accumulation DECELERATES (convex shape:
    thick near top, thin at distance — "wall right here").

    Price gaps are floored at 0.01 (1 tick) to avoid division blowup
    when multiple levels share the same price.
    """
    cbv1 = pl.col(v_cols[0])
    cbv3 = cbv1 + pl.col(v_cols[1]) + pl.col(v_cols[2])
    cbv5 = cbv3 + pl.col(v_cols[3]) + pl.col(v_cols[4])

    # Price gaps (≥ 1 tick).  MUST use .abs(): AskP1 < AskP3 so (P1-P3) < 0.
    gap_13 = pl.when((p1 - p3).abs() > 0.009).then((p1 - p3).abs()).otherwise(pl.lit(0.01))
    gap_35 = pl.when((p3 - p5).abs() > 0.009).then((p3 - p5).abs()).otherwise(pl.lit(0.01))

    slope_13 = (cbv3 - cbv1) / gap_13
    slope_35 = (cbv5 - cbv3) / gap_35

    total = sum(pl.col(v) for v in v_cols)
    convexity = (slope_35 - slope_13) / pl.when(total > 0).then(total).otherwise(pl.lit(1.0))

    return pl.when(total > 0).then(convexity).otherwise(None)


@register("BookConvexity", required_cols=_BOOK_CURVE_COLS)
def _compute_book_convexity(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulative-volume convexity: bid convexity minus ask convexity.

    Positive → bid side's depth accelerates outward more than ask side's
    (bid has a thicker "tail"). This is the opposite of DepthConcentration
    (which only sees the top).
    """
    bid_p1 = pl.col("BidPrice1")
    bid_p3 = pl.col("BidPrice3")
    bid_p5 = pl.col("BidPrice5")
    ask_p1 = pl.col("AskPrice1")
    ask_p3 = pl.col("AskPrice3")
    ask_p5 = pl.col("AskPrice5")

    bid_conv = _cumvol_convexity(
        [f"BidPrice{i}" for i in range(1, 6)],
        [f"BidVolume{i}" for i in range(1, 6)],
        bid_p1, bid_p3, bid_p5,
    )
    ask_conv = _cumvol_convexity(
        [f"AskPrice{i}" for i in range(1, 6)],
        [f"AskVolume{i}" for i in range(1, 6)],
        ask_p1, ask_p3, ask_p5,
    )
    return df.with_columns((bid_conv - ask_conv).alias("BookConvexity"))


# --- VWAP价格偏离 ---

_VWAP_COLS_5 = (
    [f"BidPrice{i}" for i in range(1, 6)]
    + [f"AskPrice{i}" for i in range(1, 6)]
    + [f"BidVolume{i}" for i in range(1, 6)]
    + [f"AskVolume{i}" for i in range(1, 6)]
)


@register("VWAP_Deviation", required_cols=_VWAP_COLS_5)
def _compute_vwap_deviation(df: pl.DataFrame) -> pl.DataFrame:
    """Weighted avg bid/ask price deviation from mid.

    BidVWAP = Σ(BP_i × BV_i) / ΣBV_i
    AskVWAP = Σ(AP_i × SV_i) / ΣSV_i

    Deviation = (AskVWAP - BidVWAP) / mid  — wide VWAP spread suggests
    real depth is further from mid than the top level suggests.
    """
    bv_sum = sum(pl.col(f"BidVolume{i}") for i in range(1, 6))
    sv_sum = sum(pl.col(f"AskVolume{i}") for i in range(1, 6))
    bvwap = sum(pl.col(f"BidPrice{i}") * pl.col(f"BidVolume{i}") for i in range(1, 6))
    svwap = sum(pl.col(f"AskPrice{i}") * pl.col(f"AskVolume{i}") for i in range(1, 6))
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2.0

    bid_vwap = pl.when(bv_sum > 0).then(bvwap / bv_sum).otherwise(None)
    ask_vwap = pl.when(sv_sum > 0).then(svwap / sv_sum).otherwise(None)

    return df.with_columns(
        pl.when(bid_vwap.is_not_null() & ask_vwap.is_not_null() & (mid > 0))
        .then((ask_vwap - bid_vwap) / mid)
        .otherwise(None)
        .alias("VWAP_Deviation")
    )


# --- AvgOrderSize: 每档平均单笔委托量 ---

_AVG_ORDER_COLS = (
    [f"BidVolume{i}" for i in range(1, 6)]
    + [f"AskVolume{i}" for i in range(1, 6)]
    + [f"NumOrdersB{i}" for i in range(1, 6)]
    + [f"NumOrdersS{i}" for i in range(1, 6)]
)


@register("AvgOrderSizeImb", required_cols=_AVG_ORDER_COLS)
def _compute_avg_order_size_imb(df: pl.DataFrame) -> pl.DataFrame:
    """Imbalance of average order size (volume per order count).

    Large avg size → institutional; small → retail.
    A-share: retail vs institutional signal is highly predictive.

    AvgSizeBid = ΣBV_i / ΣNumOrdersB_i
    AvgSizeAsk = ΣSV_i / ΣNumOrdersS_i
    Imbalance = (AvgBid - AvgAsk) / (AvgBid + AvgAsk)
    """
    sum_bv = sum(pl.col(f"BidVolume{i}") for i in range(1, 6))
    sum_sv = sum(pl.col(f"AskVolume{i}") for i in range(1, 6))
    sum_nb = sum(pl.col(f"NumOrdersB{i}") for i in range(1, 6))
    sum_ns = sum(pl.col(f"NumOrdersS{i}") for i in range(1, 6))

    avg_bid = pl.when(sum_nb > 0).then(sum_bv / sum_nb).otherwise(None)
    avg_ask = pl.when(sum_ns > 0).then(sum_sv / sum_ns).otherwise(None)

    return df.with_columns(
        pl.when((avg_bid + avg_ask) > 0)
        .then((avg_bid - avg_ask) / (avg_bid + avg_ask))
        .otherwise(0.0)
        .alias("AvgOrderSizeImb")
    )


# ============================================================
# ⑤ A股涨跌停特征
# ============================================================
# SH data lacks HighLimitPrice/LowLimitPrice columns.
# We compute limit prices from PreCloPrice when exchange-provided
# values are unavailable, using board-specific limit rates:
#   688xxx → STAR (科创板) → 20%
#   300xxx → ChiNext (创业板) → 20%
#   others → Main Board → 10%
# ST stocks (5%) are not detected automatically — flag is best-effort.

_LIMIT_COLS = ["PreCloPrice", "LastPrice", "BidPrice1", "AskPrice1",
               "BidVolume1", "AskVolume1",
               "HighLimitPrice", "LowLimitPrice"]
# HighLimitPrice/LowLimitPrice: SH parquet lacks them but loader null-fills safely


def _limit_prices(df_columns: list[str]) -> tuple[pl.Expr, pl.Expr]:
    """Build HighLimitPrice / LowLimitPrice expressions.

    Prefer native exchange-provided values (深市), fall back to estimation
    from PreCloPrice (沪市).  Native columns are present but null for SH
    rows after the loader's superset merge.
    """
    sid = pl.col("SecurityID")

    # Board detection for SH fallback: 688/300 → 20%, others → 10%
    rate = (
        pl.when(sid.str.starts_with("688") | sid.str.starts_with("300") | sid.str.starts_with("301"))
        .then(pl.lit(0.20))
        .otherwise(pl.lit(0.10))
    )

    hi_est = (pl.col("PreCloPrice") * (1.0 + rate)).round(2)
    lo_est = (pl.col("PreCloPrice") * (1.0 - rate)).round(2)

    if "HighLimitPrice" in df_columns:
        hi = (pl.when(pl.col("HighLimitPrice").is_not_null())
              .then(pl.col("HighLimitPrice"))
              .otherwise(hi_est))
    else:
        hi = hi_est

    if "LowLimitPrice" in df_columns:
        lo = (pl.when(pl.col("LowLimitPrice").is_not_null())
              .then(pl.col("LowLimitPrice"))
              .otherwise(lo_est))
    else:
        lo = lo_est

    return hi, lo


@register("LimitUpDist", required_cols=_LIMIT_COLS)
def _compute_limit_up_dist(df: pl.DataFrame) -> pl.DataFrame:
    """Distance to limit-up: (HighLimitPrice - LastPrice) / HighLimitPrice.

    ∈ [0, 1]; smaller → closer to hitting limit-up.
    """
    hi, _ = _limit_prices(df.columns)
    return df.with_columns(
        pl.when(hi > 0).then((hi - pl.col("LastPrice")) / hi)
        .otherwise(None).alias("LimitUpDist")
    )


@register("LimitDownDist", required_cols=_LIMIT_COLS)
def _compute_limit_down_dist(df: pl.DataFrame) -> pl.DataFrame:
    """Distance to limit-down: (LastPrice - LowLimitPrice) / LastPrice.

    ∈ [0, 1]; smaller → closer to hitting limit-down.
    """
    _, lo = _limit_prices(df.columns)
    return df.with_columns(
        pl.when(pl.col("LastPrice") > 0)
        .then((pl.col("LastPrice") - lo) / pl.col("LastPrice"))
        .otherwise(None).alias("LimitDownDist")
    )


@register("IsLimitUp", required_cols=_LIMIT_COLS)
def _compute_is_limit_up(df: pl.DataFrame) -> pl.DataFrame:
    """Binary: stock is sealed at limit-up.

    A-share撮合机制: when a stock hits limit-up, all sell orders at the
    limit price are consumed and AskPrice1 disappears (null or 0).  At the
    same time, BidPrice1 sits exactly at the limit price.

    We use |BidPrice1 - HighLimitPrice| < 0.005 as the equality check
    (half a tick).  This is a Float64 precision guard, NOT a "near-limit"
    tolerance — at 0.01 tick granularity, any price within 0.005 of the
    limit IS the limit price.

    Distinguishes real limit-up from suspension: both require BidPrice1 > 0
    (suspended stocks have BidPrice1 = AskPrice1 = 0).

    涨停价来源: 深市用原生 HighLimitPrice, 沪市从 PreCloPrice 推算
    (688=科创板20%, 300=创业板20%, 其余=主板10%).
    """
    hi, _ = _limit_prices(df.columns)
    return df.with_columns(
        (
            (pl.col("AskPrice1").is_null() | (pl.col("AskPrice1") <= 0))
            & (pl.col("BidPrice1") > 0)
            & ((pl.col("BidPrice1") - hi).abs() < 0.005)
        ).cast(pl.Int32).alias("IsLimitUp")
    )


@register("IsLimitDown", required_cols=_LIMIT_COLS)
def _compute_is_limit_down(df: pl.DataFrame) -> pl.DataFrame:
    """Binary: stock is sealed at limit-down.

    跌停: BidPrice1消失, AskPrice1精确等于跌停价.
    AskPrice1 > 0 排除停牌股 (停牌时两边都=0).
    """
    _, lo = _limit_prices(df.columns)
    return df.with_columns(
        (
            (pl.col("BidPrice1").is_null() | (pl.col("BidPrice1") <= 0))
            & (pl.col("AskPrice1") > 0)
            & ((pl.col("AskPrice1") - lo).abs() < 0.005)
        ).cast(pl.Int32).alias("IsLimitDown")
    )


@register("LimitBlockAmt", required_cols=_LIMIT_COLS)
def _compute_limit_block_amt(df: pl.DataFrame) -> pl.DataFrame:
    """Sealing amount at limit price (封单金额).

    At limit-up: BidVolume1 × BidPrice1  (buy orders sealing the ceiling)
    At limit-down: AskVolume1 × AskPrice1 (sell orders sealing the floor)
    Normal state: 0.

    Uses the same |price - limit| < 0.005 check as IsLimitUp/Down
    to ensure we're at the exact limit price, not just near it.
    """
    hi, lo = _limit_prices(df.columns)

    is_up = (
        (pl.col("AskPrice1").is_null() | (pl.col("AskPrice1") <= 0))
        & (pl.col("BidPrice1") > 0)
        & ((pl.col("BidPrice1") - hi).abs() < 0.005)
    )
    is_down = (
        (pl.col("BidPrice1").is_null() | (pl.col("BidPrice1") <= 0))
        & (pl.col("AskPrice1") > 0)
        & ((pl.col("AskPrice1") - lo).abs() < 0.005)
    )

    return df.with_columns(
        pl.when(is_up)
        .then(pl.col("BidVolume1") * pl.col("BidPrice1"))
        .when(is_down)
        .then(pl.col("AskVolume1") * pl.col("AskPrice1"))
        .otherwise(0.0)
        .alias("LimitBlockAmt")
    )


# --- Near-limit: approaching but not yet sealed ---

def _is_limit_up_cond(df_columns: list[str]) -> pl.Expr:
    """Inline IsLimitUp condition (avoids dependency on IsLimitUp column)."""
    hi, _ = _limit_prices(df_columns)
    return (
        (pl.col("AskPrice1").is_null() | (pl.col("AskPrice1") <= 0))
        & (pl.col("BidPrice1") > 0)
        & ((pl.col("BidPrice1") - hi).abs() < 0.005)
    )


def _is_limit_down_cond(df_columns: list[str]) -> pl.Expr:
    """Inline IsLimitDown condition."""
    _, lo = _limit_prices(df_columns)
    return (
        (pl.col("BidPrice1").is_null() | (pl.col("BidPrice1") <= 0))
        & (pl.col("AskPrice1") > 0)
        & ((pl.col("AskPrice1") - lo).abs() < 0.005)
    )


@register("NearLimitUp", required_cols=_LIMIT_COLS)
def _compute_near_limit_up(df: pl.DataFrame) -> pl.DataFrame:
    """Within 2% of limit-up but not yet sealed."""
    hi, _ = _limit_prices(df.columns)
    is_sealed = _is_limit_up_cond(df.columns)
    return df.with_columns(
        (
            ~is_sealed
            & (pl.col("LastPrice") > 0) & (hi > 0)
            & ((hi - pl.col("LastPrice")) / hi < 0.02)
        ).cast(pl.Int32).alias("NearLimitUp")
    )


@register("NearLimitDown", required_cols=_LIMIT_COLS)
def _compute_near_limit_down(df: pl.DataFrame) -> pl.DataFrame:
    """Within 2% of limit-down but not yet sealed."""
    _, lo = _limit_prices(df.columns)
    is_sealed = _is_limit_down_cond(df.columns)
    return df.with_columns(
        (
            ~is_sealed
            & (pl.col("LastPrice") > 0) & (lo > 0)
            & ((pl.col("LastPrice") - lo) / pl.col("LastPrice") < 0.02)
        ).cast(pl.Int32).alias("NearLimitDown")
    )


# --- 一字板: opened at limit and stayed ---

@register("IsGapLimitUp", required_cols=_LIMIT_COLS)
def _compute_is_gap_limit_up(df: pl.DataFrame) -> pl.DataFrame:
    """一字涨停: stock opened at limit-up (first bucket already sealed)."""
    is_up = _is_limit_up_cond(df.columns)
    g_key = _time_group(df)  # ["SecurityID"] or ["SecurityID","date"]
    first_up = (
        df.sort("timestamp").group_by(g_key)
        .agg(is_up.first().alias("_first_up"))
    )
    df = df.join(first_up, on=g_key, how="left")
    return df.with_columns(
        (is_up & pl.col("_first_up")).cast(pl.Int32).alias("IsGapLimitUp")
    ).drop("_first_up")


@register("IsGapLimitDown", required_cols=_LIMIT_COLS)
def _compute_is_gap_limit_down(df: pl.DataFrame) -> pl.DataFrame:
    """一字跌停: stock opened at limit-down."""
    is_down = _is_limit_down_cond(df.columns)
    g_key = _time_group(df)
    first_down = (
        df.sort("timestamp").group_by(g_key)
        .agg(is_down.first().alias("_first_down"))
    )
    df = df.join(first_down, on=g_key, how="left")
    return df.with_columns(
        (is_down & pl.col("_first_down")).cast(pl.Int32).alias("IsGapLimitDown")
    ).drop("_first_down")


# --- 涨跌停附近买卖盘稀缺程度 ---

_NEAR_LIMIT_COLS = _LIMIT_COLS + ["BidVolume1", "AskVolume1", "BidVolume2", "BidVolume3",
                                   "AskVolume2", "AskVolume3"]


@register("LimitAskScarcity", required_cols=_NEAR_LIMIT_COLS)
def _compute_limit_ask_scarcity(df: pl.DataFrame) -> pl.DataFrame:
    """When near limit-up: how thin is the ask side?

    = 1 - AskVol_1..3 / (BidVol_1..3 + AskVol_1..3). 高值 → 卖方稀缺 → 易封板.
    """
    ask_vol = pl.col("AskVolume1") + pl.col("AskVolume2") + pl.col("AskVolume3")
    bid_vol = pl.col("BidVolume1") + pl.col("BidVolume2") + pl.col("BidVolume3")
    hi, _ = _limit_prices(df.columns)
    is_sealed = _is_limit_up_cond(df.columns)
    near_up = ~is_sealed & (pl.col("LastPrice") > 0) & (hi > 0) & ((hi - pl.col("LastPrice")) / hi < 0.02)
    return df.with_columns(
        pl.when(near_up & (bid_vol + ask_vol > 0))
        .then(1.0 - ask_vol / (bid_vol + ask_vol))
        .otherwise(0.0).alias("LimitAskScarcity")
    )


@register("LimitBidScarcity", required_cols=_NEAR_LIMIT_COLS)
def _compute_limit_bid_scarcity(df: pl.DataFrame) -> pl.DataFrame:
    """When near limit-down: how thin is the bid side?"""
    ask_vol = pl.col("AskVolume1") + pl.col("AskVolume2") + pl.col("AskVolume3")
    bid_vol = pl.col("BidVolume1") + pl.col("BidVolume2") + pl.col("BidVolume3")
    _, lo = _limit_prices(df.columns)
    is_sealed = _is_limit_down_cond(df.columns)
    near_down = ~is_sealed & (pl.col("LastPrice") > 0) & (lo > 0) & ((pl.col("LastPrice") - lo) / pl.col("LastPrice") < 0.02)
    return df.with_columns(
        pl.when(near_down & (bid_vol + ask_vol > 0))
        .then(1.0 - bid_vol / (bid_vol + ask_vol))
        .otherwise(0.0).alias("LimitBidScarcity")
    )


# ============================================================
# ⑥ 逐笔成交 & 委托特征 (Section 2.3)
# ============================================================
# Trade/order data is tick-level and must be aggregated to the 3s
# snap grid before joining.  SH and SZ have different column names
# and encodings — normalization happens inside the aggregator.
#
# Corrections applied per data investigation:
#   - SZ ExecType=52 → cancellations in trade stream → filtered out
#   - SZ Order Price=0 → market orders → filtered out for price-based features
#   - SH TradeBSFlag: B/S only (no 'N' in this data)
#   - Sort by (time, TradeIndex/ApplSeqNum) for correct tick ordering
#   - Large threshold: 90th percentile of trade amount per stock per day

import time as _time


def _trade_path(date: str, market: str) -> str:
    return f"/fast1/user001/stock_data/type=trade_{market}/date={date}/data.parquet"


def _order_path(date: str, market: str) -> str:
    return f"/fast1/user001/stock_data/type=order_{market}/date={date}/data.parquet"


# ---- SH/SZ trade normalization ----

def _load_trade_sh(date: str) -> pl.DataFrame:
    """Load SH trade, keep only needed columns."""
    cols = ["SecurityID", "TradTime", "TradPrice", "TradVolume",
            "TradeMoney", "TradeBSFlag", "TradeIndex"]
    return pl.read_parquet(_trade_path(date, "sh")).select(cols).with_columns(
        pl.col("TradPrice", "TradVolume", "TradeMoney").cast(pl.Float64)
    )


def _load_trade_sz(date: str) -> pl.DataFrame:
    """Load SZ trade, normalize to SH column names, apply fixes.

    ExecType: 70='F'=成交, 52='4'=撤单(混入Trade流) → 只保留70.
    Direction by tick rule (no native B/S flag in SZ trade).
    """
    cols = ["SecurityID", "TransactTime", "LastPx", "LastQty",
            "ExecType", "ApplSeqNum"]
    df = pl.read_parquet(_trade_path(date, "sz")).select(cols)
    # Filter out cancellations (ExecType=52, Price=0)
    df = df.filter(pl.col("ExecType") == 70)
    # Compute amount = price × qty (SZ has no native TradeMoney)
    df = df.with_columns(
        (pl.col("LastPx").cast(pl.Float64) * pl.col("LastQty").cast(pl.Float64))
        .alias("TradeMoney")
    )
    # Rename to SH convention
    df = df.rename({
        "TransactTime": "TradTime",
        "LastPx": "TradPrice",
        "LastQty": "TradVolume",
        "ApplSeqNum": "TradeIndex",
    })
    df = df.with_columns(pl.lit("sz").alias("_mkt"))
    return df.select([
        pl.col("SecurityID"),
        pl.col("TradTime"),
        pl.col("TradPrice").cast(pl.Float64),
        pl.col("TradVolume").cast(pl.Float64),
        pl.col("TradeMoney").cast(pl.Float64),
        pl.col("TradeIndex"),
        pl.col("_mkt"),
    ])


def _apply_tick_rule(df: pl.DataFrame) -> pl.DataFrame:
    """Lee-Ready tick rule: classify trade direction from price sequence.

    TradPrice > prev_price → 'B' (buyer-initiated)
    TradPrice < prev_price → 'S' (seller-initiated)
    TradPrice == prev_price → forward-fill last determined direction.
    The first trade of each stock defaults to 'B'.

    CRITICAL: forward_fill the DETERMINED flags first, THEN fill the
    remaining null (first trade only) with 'B'.  The old approach
    of fill_null("B") on prev_flag systematically biased equal-price
    chains to 'B'.
    """
    # Group key: SecurityID + date (if present) to prevent cross-day contamination
    g = ["SecurityID", "date"] if "date" in df.columns else ["SecurityID"]
    prev_p = pl.col("TradPrice").shift(1).over(g)

    return df.with_columns(
        pl.when(pl.col("TradPrice") > prev_p)
        .then(pl.lit("B"))
        .when(pl.col("TradPrice") < prev_p)
        .then(pl.lit("S"))
        .otherwise(None)
        .alias("tick_flag")
    ).with_columns(
        pl.col("tick_flag")
        .forward_fill().over(g)
        .fill_null("B")
        .alias("TradeBSFlag")
    )


# ---- Trade aggregation to 3s ----

def _compute_consecutive_streak(df: pl.DataFrame) -> pl.DataFrame:
    """Per-tick: label each trade with its streak group and compute max
    consecutive same-direction streak per (SecurityID, 3s bucket).

    Sorts by (SecurityID, TradTime, TradeIndex).
    Streaks reset at bucket AND day boundaries.
    """
    # Base group key for per-stock + optionally per-date
    base_g = ["SecurityID", "date"] if "date" in df.columns else ["SecurityID"]
    bucket_g = base_g + ["timestamp"]  # streak resets within each bucket

    df = df.sort(["SecurityID", "TradTime", "TradeIndex"])

    df = df.with_columns(
        (pl.col("TradeBSFlag") != pl.col("TradeBSFlag").shift(1).over(bucket_g))
        .cast(pl.Int32).fill_null(0).alias("_dir_chg")
    )
    df = df.with_columns(
        pl.col("_dir_chg").cum_sum().over(bucket_g).alias("_streak_grp")
    )

    # Count streak lengths per (SecurityID, timestamp, streak group)
    streak_len = df.group_by(["SecurityID", "timestamp", "_streak_grp"]).agg([
        pl.len().alias("streak_len"),
        pl.col("TradeBSFlag").first().alias("streak_dir"),
    ])

    # Per (SecurityID, timestamp): max consecutive B and S streak length
    streak_agg = streak_len.group_by(["SecurityID", "timestamp"]).agg([
        pl.col("streak_len").filter(pl.col("streak_dir") == "B").max().alias("max_consec_buy"),
        pl.col("streak_len").filter(pl.col("streak_dir") == "S").max().alias("max_consec_sell"),
    ])

    return streak_agg.with_columns([
        pl.col("max_consec_buy").fill_null(0),
        pl.col("max_consec_sell").fill_null(0),
    ])


def _parse_time_float(time_col: pl.Expr) -> pl.Expr:
    """Parse 'HH:MM:SS.mmm' → seconds since midnight as Float64."""
    return (time_col.str.slice(0, 2).cast(pl.Float64) * 3600.0
            + time_col.str.slice(3, 2).cast(pl.Float64) * 60.0
            + time_col.str.slice(6, 2).cast(pl.Float64)
            + time_col.str.slice(9, 3).cast(pl.Float64) / 1000.0)


def _ceil_3s(secs: pl.Expr) -> pl.Expr:
    """CEIL to 3s grid: trades in (09:30:00, 09:30:03] → bucket 09:30:03."""
    return ((secs / 3.0).ceil() * 3).cast(pl.Int32)


def _aggregate_trades(df: pl.DataFrame, snap_prices: pl.DataFrame | None = None,
                      prev_thresholds: pl.DataFrame | None = None) -> pl.DataFrame:
    """Aggregate tick trade data to 3s buckets per stock.

    Corrected alignment:
    1. Asof-join each trade to the most recent snap (backward) → pre-trade book.
    2. Compute penetration at tick level.
    3. CEIL-bucket trades → no future information at prediction time.

    If prev_thresholds provided (from previous trading day), uses those
    for large-trade classification instead of same-day 90th percentile.
    """
    # Parse trade time to Float64 seconds + ceil to 3s
    trade_secs = _parse_time_float(pl.col("TradTime"))
    df = df.with_columns([
        trade_secs.alias("_trade_secs"),
        _ceil_3s(trade_secs).alias("timestamp"),
    ])

    # Large trade threshold: previous day only, no same-day fallback (future leak).
    if prev_thresholds is not None and prev_thresholds.height > 0:
        df = df.join(prev_thresholds, on="SecurityID", how="left")
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("large_thresh"))

    is_buy = pl.col("TradeBSFlag") == "B"
    is_sell = pl.col("TradeBSFlag") == "S"
    is_large = pl.col("TradeMoney") >= pl.col("large_thresh")

    # Base aggregation
    agg = df.group_by(["SecurityID", "timestamp"]).agg([
        pl.col("TradeMoney").filter(is_buy).sum().alias("trade_buy_amt"),
        pl.col("TradeMoney").filter(is_sell).sum().alias("trade_sell_amt"),
        pl.col("TradeMoney").filter(is_buy & is_large).sum().alias("trade_large_buy"),
        pl.col("TradeMoney").filter(is_sell & is_large).sum().alias("trade_large_sell"),
        (is_buy).cast(pl.Int32).sum().alias("trade_buy_cnt"),
        (is_sell).cast(pl.Int32).sum().alias("trade_sell_cnt"),
        pl.len().alias("trade_count"),
        (pl.col("TradPrice") * pl.col("TradVolume")).sum().alias("trade_vwap_num"),
        pl.col("TradVolume").sum().alias("trade_vwap_den"),
        # avg price deviation from weighted mid (approximate — uses trade-level prices)
        ((pl.col("TradPrice") - pl.col("TradPrice").mean()).abs() /
         pl.col("TradPrice").mean()).mean().alias("trade_price_dispersion"),
    ])

    agg = agg.with_columns(
        pl.when(pl.col("trade_vwap_den") > 0)
        .then(pl.col("trade_vwap_num") / pl.col("trade_vwap_den"))
        .otherwise(None).alias("trade_vwap")
    ).drop("trade_vwap_num")  # keep trade_vwap_den as VWAP weight for re-aggregation

    # Penetration & price deviation: asof-join to pre-trade snap, tick-level
    if snap_prices is not None:
        # Parse snap time for asof join
        snap_secs = _parse_time_float(pl.col("_snap_time"))
        sp = snap_prices.with_columns(snap_secs.alias("_snap_secs"))
        # Asof-join: each trade gets the most recent snap BEFORE it (backward)
        df = df.sort(["SecurityID", "_trade_secs"])
        sp_s = sp.select([
            "SecurityID", "_snap_secs", "BidPrice1", "AskPrice1", "mid_price"
        ])
        sp_sorted = sp_s.sort(["SecurityID", "_snap_secs"])
        df = df.join_asof(
            sp_sorted,
            left_on="_trade_secs", right_on="_snap_secs",
            by="SecurityID", strategy="backward"
        )
        # Tick-level penetration: compare to PRE-TRADE book
        is_pen_buy = is_buy & (pl.col("TradPrice") > pl.col("AskPrice1"))
        is_pen_sell = is_sell & (pl.col("TradPrice") < pl.col("BidPrice1"))
        price_dev = ((pl.col("TradPrice") - pl.col("mid_price")).abs()
                     / pl.col("mid_price")).fill_null(0.0)

        pen = df.group_by(["SecurityID", "timestamp"]).agg([
            (is_pen_buy).cast(pl.Int32).sum().alias("pen_buy_cnt"),
            (is_pen_sell).cast(pl.Int32).sum().alias("pen_sell_cnt"),
            pl.col("TradeMoney").filter(is_pen_buy).sum().alias("pen_buy_amt"),
            pl.col("TradeMoney").filter(is_pen_sell).sum().alias("pen_sell_amt"),
            price_dev.mean().alias("trade_price_dev"),
        ])
        agg = agg.join(pen, on=["SecurityID", "timestamp"], how="left")
        agg = agg.with_columns([
            pl.col("pen_buy_cnt", "pen_sell_cnt", "pen_buy_amt", "pen_sell_amt").fill_null(0),
        ])
        # Keep trade_price_dev from pen agg, not separately
    else:
        agg = agg.with_columns(pl.lit(None).alias("trade_price_dev"))

    # Consecutive BS streak
    streak = _compute_consecutive_streak(df)
    agg = agg.join(streak, on=["SecurityID", "timestamp"], how="left")
    agg = agg.with_columns([
        pl.col("max_consec_buy", "max_consec_sell").fill_null(0),
    ])

    return agg


def _load_and_agg_trade(date: str, snap_prices: pl.DataFrame | None = None,
                        prev_thresholds: pl.DataFrame | None = None) -> pl.DataFrame:
    """Load SH+SZ trade data, normalize, aggregate to 3s."""
    frames = []

    # SH
    sh = _load_trade_sh(date)
    frames.append(_aggregate_trades(sh, snap_prices, prev_thresholds))

    # SZ: sort first, then apply tick rule (shift needs correct order!)
    sz = _load_trade_sz(date)
    sz = sz.sort(["SecurityID", "TradTime", "TradeIndex"])
    sz = _apply_tick_rule(sz)
    frames.append(_aggregate_trades(sz, snap_prices, prev_thresholds))

    # Concat + re-aggregate across markets
    result = pl.concat(frames, how="vertical")
    key_cols = {"SecurityID", "timestamp"}

    # Columns to sum vs columns to take max of (streak lengths)
    max_cols = {"max_consec_buy", "max_consec_sell"}
    weighted_cols = {"trade_vwap", "trade_price_dispersion", "trade_price_dev"}
    sum_cols = [c for c, d in zip(result.columns, result.dtypes)
                if d.is_numeric() and c not in key_cols
                and c not in max_cols and c not in weighted_cols]
    # Exclude timestamp from weighted_cols check
    weighted_present = [c for c in weighted_cols if c in result.columns]

    agg_exprs = [pl.col(c).sum() for c in sum_cols]
    agg_exprs += [pl.col(c).max() for c in max_cols if c in result.columns]

    if "trade_vwap" in weighted_present and "trade_vwap_den" in result.columns:
        agg_exprs.append((pl.col("trade_vwap") * pl.col("trade_vwap_den")).sum().alias("vwap_w"))
        agg_exprs.append(pl.col("trade_vwap_den").sum().alias("vwap_den_sum"))
    elif "trade_vwap" in weighted_present:
        agg_exprs.append((pl.col("trade_vwap") * pl.col("trade_count")).sum().alias("vwap_w"))
    if "trade_price_dispersion" in weighted_present:
        agg_exprs.append((pl.col("trade_price_dispersion") * pl.col("trade_count")).sum().alias("disp_w"))
    if "trade_price_dev" in weighted_present:
        agg_exprs.append((pl.col("trade_price_dev") * pl.col("trade_count")).sum().alias("pdev_w"))

    result = result.group_by(["SecurityID", "timestamp"]).agg(agg_exprs)
    if "vwap_w" in result.columns:
        denom_col = "vwap_den_sum" if "vwap_den_sum" in result.columns else "trade_count"
        result = result.with_columns(
            pl.when(pl.col(denom_col) > 0).then(pl.col("vwap_w") / pl.col(denom_col))
            .otherwise(None).alias("trade_vwap")
        ).drop(["vwap_w"] + (["vwap_den_sum"] if "vwap_den_sum" in result.columns else []))
    if "disp_w" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("trade_count") > 0).then(pl.col("disp_w") / pl.col("trade_count"))
            .otherwise(None).alias("trade_price_dispersion")
        ).drop("disp_w")
    if "pdev_w" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("trade_count") > 0).then(pl.col("pdev_w") / pl.col("trade_count"))
            .otherwise(None).alias("trade_price_dev")
        ).drop("pdev_w")

    return result


# ---- Trade-derived registered features ----

# Trade features need snap prices for penetration/asof-join and mid_price for VWAP dev
_TRADE_REQUIRED_COLS = ["UpdateTime", "BidPrice1", "AskPrice1", "mid_price"]
# Order features need snap prices for depth computation and mid_price for aggressiveness
_ORDER_REQUIRED_COLS = ["BidPrice1", "AskPrice1", "mid_price"]

def _get_prev_date(date: str) -> str | None:
    """Get previous trading date from available snap directories."""
    import os as _os
    snap_dir = "/fast1/user001/stock_data/type=snap_sh"
    if not _os.path.exists(snap_dir):
        return None
    all_dates = sorted([
        d.replace("date=", "") for d in _os.listdir(snap_dir) if d.startswith("date=")
    ])
    idx = all_dates.index(date) if date in all_dates else -1
    return all_dates[idx - 1] if idx > 0 else None


def _load_prev_trade_thresholds(date: str) -> pl.DataFrame | None:
    """Load previous day's per-stock 90th percentile trade amount thresholds."""
    prev = _get_prev_date(date)
    if prev is None:
        return None
    path = f"/fast1/user001/factor_values/_trade_thresh/{prev}.parquet"
    import os as _os
    if not _os.path.exists(path):
        return None
    return pl.read_parquet(path)


def _save_trade_thresholds(df: pl.DataFrame, date: str) -> None:
    """Save per-stock 90th percentile trade thresholds for next day's use."""
    import os as _os
    out_dir = "/fast1/user001/factor_values/_trade_thresh"
    _os.makedirs(out_dir, exist_ok=True)
    thresh = df.group_by("SecurityID").agg(
        pl.col("TradeMoney").quantile(0.90).alias("large_thresh")
    )
    thresh.write_parquet(f"{out_dir}/{date}.parquet", compression="zstd")


def _load_prev_order_thresholds(date: str) -> pl.DataFrame | None:
    """Load previous day's per-stock 90th percentile order amount thresholds."""
    prev = _get_prev_date(date)
    if prev is None:
        return None
    path = f"/fast1/user001/factor_values/_order_thresh/{prev}.parquet"
    import os as _os
    if not _os.path.exists(path):
        return None
    return pl.read_parquet(path)


def _save_order_thresholds(df: pl.DataFrame, date: str) -> None:
    """Save per-stock 90th percentile order amount thresholds for next day's use."""
    import os as _os
    out_dir = "/fast1/user001/factor_values/_order_thresh"
    _os.makedirs(out_dir, exist_ok=True)
    thresh = df.group_by("SecurityID").agg(
        (pl.col("OrderPrice") * pl.col("Balance")).quantile(0.90).alias("large_thresh")
    )
    thresh.write_parquet(f"{out_dir}/{date}.parquet", compression="zstd")


# Backward compatibility aliases
_load_prev_thresholds = _load_prev_trade_thresholds
_save_thresholds = _save_trade_thresholds


def _join_trade_stats(df: pl.DataFrame, date: str) -> pl.DataFrame:
    """Load & aggregate trade, join onto snap df (idempotent).

    Uses previous trading day's large-trade thresholds to avoid look-ahead.
    Tick-level penetration computed via asof-join (backward) to pre-trade snap.
    """
    if "trade_buy_amt" in df.columns:
        return df

    cache_key = f"_trade_agg_{date}"
    if not hasattr(_join_trade_stats, "_cache"):
        _join_trade_stats._cache = {}
    if cache_key not in _join_trade_stats._cache:
        t0 = _time.time()
        # Snap prices with original UpdateTime for asof-join (not 3s bucket).
        # UpdateTime is a declared required_col — missing means caller error.
        if "UpdateTime" not in df.columns:
            raise KeyError(
                "_join_trade_stats requires 'UpdateTime' column for asof-join. "
                "Ensure _TRADE_REQUIRED_COLS is in the feature registry."
            )
        snap_prices = df.select([
            "SecurityID", "timestamp", "UpdateTime", "BidPrice1", "AskPrice1", "mid_price"
        ]).unique(subset=["SecurityID", "timestamp"]).with_columns(
            pl.col("UpdateTime").alias("_snap_time")
        )
        prev_thresh = _load_prev_trade_thresholds(date)
        trade_agg = _load_and_agg_trade(date, snap_prices, prev_thresh)
        # Save thresholds for next day
        sh_path = f"/fast1/user001/stock_data/type=trade_sh/date={date}/data.parquet"
        if os.path.exists(sh_path):
            raw_sh = pl.read_parquet(sh_path).select([
                pl.col("SecurityID"),
                pl.col("TradeMoney").cast(pl.Float64),
            ])
            raw_sz_path = f"/fast1/user001/stock_data/type=trade_sz/date={date}/data.parquet"
            if os.path.exists(raw_sz_path):
                raw_sz = pl.read_parquet(raw_sz_path).filter(
                    pl.col("ExecType") == 70
                ).select([
                    pl.col("SecurityID"),
                    (pl.col("LastPx").cast(pl.Float64) * pl.col("LastQty").cast(pl.Float64)).alias("TradeMoney"),
                ])
                raw_trade = pl.concat([raw_sh, raw_sz], how="vertical")
            else:
                raw_trade = raw_sh
            _save_trade_thresholds(raw_trade, date)
        _join_trade_stats._cache = {cache_key: trade_agg}
        elapsed = _time.time() - t0
        print(f"  Trade agg: {trade_agg.height:,} rows ({elapsed:.0f}s)", flush=True)

    trade_agg = _join_trade_stats._cache[cache_key]
    return df.join(trade_agg, on=["SecurityID", "timestamp"], how="left")


@register("TradeImb", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_imb(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Trade imbalance: (BuyAmt - SellAmt) / (BuyAmt + SellAmt) per 3s bucket."""
    df = _join_trade_stats(df, date)
    buy = pl.col("trade_buy_amt").fill_null(0.0)
    sell = pl.col("trade_sell_amt").fill_null(0.0)
    denom = buy + sell
    return df.with_columns(
        pl.when(denom > 0).then((buy - sell) / denom).otherwise(0.0).alias("TradeImb")
    )


@register("TradeVWAPDev", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_vwap_dev(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Trade VWAP deviation from mid-price (snap mid_price must already exist)."""
    df = _join_trade_stats(df, date)
    return df.with_columns(
        pl.when(pl.col("trade_vwap").is_not_null() & (pl.col("mid_price") > 0))
        .then(pl.col("trade_vwap") / pl.col("mid_price") - 1.0)
        .otherwise(None)
        .alias("TradeVWAPDev")
    )


@register("TradeIntensity", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_intensity(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Raw trade count per 3s bucket."""
    df = _join_trade_stats(df, date)
    return df.with_columns(
        pl.col("trade_count").fill_null(0).alias("TradeIntensity")
    )


@register("LargeTradeRatio", required_cols=_TRADE_REQUIRED_COLS)
def _compute_large_trade_ratio(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Large trade imbalance: (LargeBuy - LargeSell) / (LargeBuy + LargeSell).

    Large = trade amount ≥ 90th percentile within the stock on this day.
    """
    df = _join_trade_stats(df, date)
    lb = pl.col("trade_large_buy").fill_null(0.0)
    ls = pl.col("trade_large_sell").fill_null(0.0)
    denom = lb + ls
    return df.with_columns(
        pl.when(denom > 0).then((lb - ls) / denom).otherwise(0.0).alias("LargeTradeRatio")
    )


@register("TradePriceDev", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_price_dev(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Mean |TradePrice - mid| / mid within 3s bucket."""
    df = _join_trade_stats(df, date)
    return df.with_columns(pl.col("trade_price_dev").alias("TradePriceDev"))


@register("TradePriceDispersion", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_price_dispersion(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """CV of trade prices within 3s bucket."""
    df = _join_trade_stats(df, date)
    return df.with_columns(pl.col("trade_price_dispersion").alias("TradePriceDispersion"))


@register("TradePenetration", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_penetration(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Net penetration: (penBuyAmt - penSellAmt) / totalTradeAmt."""
    df = _join_trade_stats(df, date)
    pb = pl.col("pen_buy_amt").fill_null(0.0)
    ps = pl.col("pen_sell_amt").fill_null(0.0)
    total = pl.col("trade_buy_amt").fill_null(0.0) + pl.col("trade_sell_amt").fill_null(0.0)
    return df.with_columns(
        pl.when(total > 0).then((pb - ps) / total).otherwise(0.0).alias("TradePenetration")
    )


@register("TradeIntensityZ", required_cols=_TRADE_REQUIRED_COLS)
def _compute_trade_intensity_z(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Z-score of TradeCount per (date, timestamp) cross-section."""
    df = _join_trade_stats(df, date)
    tc = pl.col("trade_count").fill_null(0.0)
    g_cols = ["date", "timestamp"] if "date" in df.columns else ["timestamp"]
    grp = df.group_by(g_cols).agg([
        tc.median().alias("_med"),
        (tc - tc.median()).abs().median().alias("_mad"),
    ])
    df = df.join(grp, on=g_cols, how="left")
    return df.with_columns(
        pl.when(pl.col("_mad") > 0)
        .then((tc - pl.col("_med")) / (pl.col("_mad") * 1.4826))
        .otherwise(0.0).alias("TradeIntensityZ")
    ).drop(["_med", "_mad"])


@register("ConsecutiveBS", required_cols=_TRADE_REQUIRED_COLS)
def _compute_consecutive_bs(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """(maxConsecBuy - maxConsecSell) / (maxConsecBuy + maxConsecSell)."""
    df = _join_trade_stats(df, date)
    mb = pl.col("max_consec_buy").cast(pl.Float64).fill_null(0)
    ms = pl.col("max_consec_sell").cast(pl.Float64).fill_null(0)
    denom = mb + ms
    return df.with_columns(
        pl.when(denom > 0).then((mb - ms) / denom).otherwise(0.0).alias("ConsecutiveBS")
    )


@register("BuySellCountImb", required_cols=_TRADE_REQUIRED_COLS)
def _compute_buy_sell_count_imb(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """(BuyCnt - SellCnt) / (BuyCnt + SellCnt)."""
    df = _join_trade_stats(df, date)
    bc = pl.col("trade_buy_cnt").cast(pl.Float64).fill_null(0)
    sc = pl.col("trade_sell_cnt").cast(pl.Float64).fill_null(0)
    denom = bc + sc
    return df.with_columns(
        pl.when(denom > 0).then((bc - sc) / denom).otherwise(0.0).alias("BuySellCountImb")
    )


# ---- Order aggregation ----

def _load_order_sh(date: str) -> pl.DataFrame:
    """Load SH order data."""
    cols = ["SecurityID", "OrderTime", "OrderPrice", "Balance",
            "OrderBSFlag", "OrderType", "OrderIndex"]
    return pl.read_parquet(_order_path(date, "sh")).select(cols).with_columns(
        pl.col("OrderPrice", "Balance").cast(pl.Float64)
    )


def _load_order_sz(date: str) -> pl.DataFrame:
    """Load SZ order data, normalize to SH names.

    Side: 49='1'=买, 50='2'=卖.
    OrdType: 49='1'=新增, 50='2'=撤单, 85='U'=修改(暂忽略).
    Price=0 → market order → excluded from price-based stats.
    """
    cols = ["SecurityID", "TransactTime", "Price", "OrderQty",
            "Side", "OrdType", "ApplSeqNum"]
    df = pl.read_parquet(_order_path(date, "sz")).select(cols)

    # Decode: Side 49→'B', 50→'S'
    # OrdType 49→'A'(新增), 50→'D'(撤单)
    df = df.with_columns([
        pl.when(pl.col("Side") == 49).then(pl.lit("B"))
        .when(pl.col("Side") == 50).then(pl.lit("S"))
        .otherwise(None).alias("OrderBSFlag"),
        pl.when(pl.col("OrdType") == 49).then(pl.lit("A"))
        .when(pl.col("OrdType") == 50).then(pl.lit("D"))
        .otherwise(None).alias("OrderType"),
    ]).filter(pl.col("OrderBSFlag").is_not_null() & pl.col("OrderType").is_not_null())

    return df.select([
        pl.col("SecurityID"),
        pl.col("TransactTime").alias("OrderTime"),
        pl.col("Price").cast(pl.Float64).alias("OrderPrice"),
        pl.col("OrderQty").cast(pl.Float64).alias("Balance"),
        pl.col("OrderBSFlag"),
        pl.col("OrderType"),
        pl.col("ApplSeqNum").alias("OrderIndex"),
    ])


def _aggregate_orders(df: pl.DataFrame, snap_prices: pl.DataFrame | None = None,
                      prev_thresholds: pl.DataFrame | None = None) -> pl.DataFrame:
    """Aggregate tick order data to 3s buckets per stock.

    Uses ceil-bucketing (right-endpoint, matching trade convention).
    Large-order threshold: previous day only (no same-day fallback).

    If snap_prices provided (SecurityID, timestamp, BidPrice1, AskPrice1),
    also computes OrderDepthPos (which level the order sits at).
    Depth stats are computed on NEW ORDERS ONLY (is_new), not cancels.
    """
    # Parse OrderTime to seconds with millisecond precision, then ceil to 3s
    order_secs = _parse_time_float(pl.col("OrderTime"))
    df = df.with_columns([
        order_secs.alias("_order_secs"),
        _ceil_3s(order_secs).alias("timestamp"),
    ])

    # Order amount = Price × Qty (金额口径).  SZ market orders (Price=0) → amt=0.
    order_amt = pl.col("OrderPrice") * pl.col("Balance")

    # Large order threshold: previous day only, no same-day fallback
    if prev_thresholds is not None and prev_thresholds.height > 0:
        df = df.join(prev_thresholds, on="SecurityID", how="left")
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("large_thresh"))

    is_new = pl.col("OrderType") == "A"
    is_cancel = pl.col("OrderType") == "D"
    is_buy = pl.col("OrderBSFlag") == "B"
    is_sell = pl.col("OrderBSFlag") == "S"
    is_large = order_amt >= pl.col("large_thresh")
    has_price = pl.col("OrderPrice") > 0
    is_market = (pl.col("OrderPrice").is_null()) | (pl.col("OrderPrice") <= 0)

    agg = df.group_by(["SecurityID", "timestamp"]).agg([
        # Amount columns: Price × Qty (金额口径)
        order_amt.filter(is_new & is_buy).sum().alias("new_buy_amt"),
        order_amt.filter(is_new & is_sell).sum().alias("new_sell_amt"),
        order_amt.filter(is_cancel & is_buy).sum().alias("cancel_buy_amt"),
        order_amt.filter(is_cancel & is_sell).sum().alias("cancel_sell_amt"),
        # Count columns (for cancel rate + market order fraction)
        (is_new & is_market).cast(pl.Int64).sum().alias("market_order_cnt"),
        (is_new & is_buy).cast(pl.Int64).sum().alias("new_buy_cnt"),
        (is_new & is_sell).cast(pl.Int64).sum().alias("new_sell_cnt"),
        (is_cancel & is_buy).cast(pl.Int64).sum().alias("cancel_buy_cnt"),
        (is_cancel & is_sell).cast(pl.Int64).sum().alias("cancel_sell_cnt"),
        # Large order amounts (金额口径)
        order_amt.filter(is_new & is_buy & is_large).sum().alias("large_new_buy"),
        order_amt.filter(is_new & is_sell & is_large).sum().alias("large_new_sell"),
        order_amt.filter(is_cancel & is_buy & is_large).sum().alias("large_cancel_buy"),
        order_amt.filter(is_cancel & is_sell & is_large).sum().alias("large_cancel_sell"),
        (is_new & is_buy & is_large).cast(pl.Int64).sum().alias("large_new_buy_cnt"),
        (is_new & is_sell & is_large).cast(pl.Int64).sum().alias("large_new_sell_cnt"),
        (is_cancel & is_buy & is_large).cast(pl.Int64).sum().alias("large_cancel_buy_cnt"),
        (is_cancel & is_sell & is_large).cast(pl.Int64).sum().alias("large_cancel_sell_cnt"),
        # Volume-weighted average price
        (pl.col("OrderPrice") * pl.col("Balance")).filter(is_new & is_buy & has_price).sum().alias("vwap_num_buy"),
        pl.col("Balance").filter(is_new & is_buy & has_price).sum().alias("vwap_den_buy"),
        (pl.col("OrderPrice") * pl.col("Balance")).filter(is_new & is_sell & has_price).sum().alias("vwap_num_sell"),
        pl.col("Balance").filter(is_new & is_sell & has_price).sum().alias("vwap_den_sell"),
    ]).with_columns([
        pl.when(pl.col("vwap_den_buy") > 0).then(pl.col("vwap_num_buy") / pl.col("vwap_den_buy")).otherwise(None).alias("avg_price_new_buy"),
        pl.when(pl.col("vwap_den_sell") > 0).then(pl.col("vwap_num_sell") / pl.col("vwap_den_sell")).otherwise(None).alias("avg_price_new_sell"),
    ]).drop(["vwap_num_buy", "vwap_den_buy", "vwap_num_sell", "vwap_den_sell"])

    # OrderDepthPos: computed on NEW ORDERS ONLY, with tick-level asof join to snap
    if snap_prices is not None:
        # Asof-join: each order gets the most recent snap BEFORE it (backward)
        # using the order's raw event time (_order_secs), not the ceil bucket.
        sp = snap_prices.select([
            "SecurityID", "timestamp", "BidPrice1", "AskPrice1"
        ]).with_columns(
            pl.col("timestamp").cast(pl.Float64).alias("_snap_secs")
        )
        df = df.sort(["SecurityID", "_order_secs"])
        sp_sorted = sp.sort(["SecurityID", "_snap_secs"])
        df = df.join_asof(
            sp_sorted,
            left_on="_order_secs", right_on="_snap_secs",
            by="SecurityID", strategy="backward"
        )
        tick = pl.lit(0.01)
        # NEW ORDERS ONLY for depth stats (not cancels)
        valid_buy = is_new & is_buy & has_price & (pl.col("BidPrice1") > 0)
        valid_sell = is_new & is_sell & has_price & (pl.col("AskPrice1") > 0)
        buy_depth = pl.when(valid_buy).then(
            (pl.col("BidPrice1") - pl.col("OrderPrice")) / tick
        ).otherwise(None)
        sell_depth = pl.when(valid_sell).then(
            (pl.col("OrderPrice") - pl.col("AskPrice1")) / tick
        ).otherwise(None)
        # Clip to [-500, 500] ticks — beyond that is data error
        buy_depth = buy_depth.clip(-500, 500)
        sell_depth = sell_depth.clip(-500, 500)
        depth_agg = df.group_by(["SecurityID", "timestamp"]).agg([
            buy_depth.mean().alias("avg_depth_buy"),
            sell_depth.mean().alias("avg_depth_sell"),
            # Depth distribution: fraction at best (≤1 tick), near (1-5), deep (>5)
            (buy_depth.abs() <= 1).cast(pl.Int64).sum().alias("depth_best_buy"),
            (buy_depth.abs().is_between(1.01, 5)).cast(pl.Int64).sum().alias("depth_near_buy"),
            (buy_depth.abs() > 5).cast(pl.Int64).sum().alias("depth_deep_buy"),
            (sell_depth.abs() <= 1).cast(pl.Int64).sum().alias("depth_best_sell"),
            (sell_depth.abs().is_between(1.01, 5)).cast(pl.Int64).sum().alias("depth_near_sell"),
            (sell_depth.abs() > 5).cast(pl.Int64).sum().alias("depth_deep_sell"),
            buy_depth.is_not_null().cast(pl.Int64).sum().alias("depth_total_buy"),
            sell_depth.is_not_null().cast(pl.Int64).sum().alias("depth_total_sell"),
        ])
        agg = agg.join(depth_agg, on=["SecurityID", "timestamp"], how="left")

    return agg


def _load_and_agg_order(date: str, snap_prices: pl.DataFrame | None = None,
                        prev_thresholds: pl.DataFrame | None = None) -> pl.DataFrame:
    """Load SH+SZ order data, normalize, aggregate to 3s."""
    frames = []
    for mkt, loader in [("sh", _load_order_sh), ("sz", _load_order_sz)]:
        try:
            o = loader(date)
            o = o.sort(["SecurityID", "OrderTime", "OrderIndex"])
            frames.append(_aggregate_orders(o, snap_prices, prev_thresholds))
        except FileNotFoundError:
            continue

    result = pl.concat(frames, how="vertical")
    key_cols = {"SecurityID", "timestamp"}
    num_cols = [c for c, d in zip(result.columns, result.dtypes)
                if d.is_numeric() and c not in key_cols]
    result = result.group_by(["SecurityID", "timestamp"]).agg(
        [pl.col(c).sum() for c in num_cols]
    )
    return result


def _join_order_stats(df: pl.DataFrame, date: str) -> pl.DataFrame:
    """Load & aggregate order data, join onto snap df (idempotent).

    Uses previous trading day's large-order thresholds to avoid look-ahead.
    Saves current day's thresholds for next day's use.
    """
    if "new_buy_amt" in df.columns:
        return df
    cache_key = f"_order_agg_{date}"
    if not hasattr(_join_order_stats, "_cache"):
        _join_order_stats._cache = {}
    if cache_key not in _join_order_stats._cache:
        t0 = _time.time()
        snap_prices = df.select([
            "SecurityID", "timestamp", "BidPrice1", "AskPrice1"
        ]).unique(subset=["SecurityID", "timestamp"])
        prev_thresh = _load_prev_order_thresholds(date)
        _join_order_stats._cache = {cache_key: _load_and_agg_order(date, snap_prices, prev_thresh)}
        # Save current day's order thresholds for next day
        sh_path = f"/fast1/user001/stock_data/type=order_sh/date={date}/data.parquet"
        if os.path.exists(sh_path):
            raw_sh = pl.read_parquet(sh_path).select([
                pl.col("SecurityID"),
                (pl.col("OrderPrice").cast(pl.Float64) * pl.col("Balance").cast(pl.Float64)).alias("OrderAmount"),
            ])
            raw_sz_path = f"/fast1/user001/stock_data/type=order_sz/date={date}/data.parquet"
            if os.path.exists(raw_sz_path):
                raw_sz = pl.read_parquet(raw_sz_path).select([
                    pl.col("SecurityID"),
                    (pl.col("OrderPrice").cast(pl.Float64) * pl.col("Balance").cast(pl.Float64)).alias("OrderAmount"),
                ])
                raw_order = pl.concat([raw_sh, raw_sz], how="vertical")
            else:
                raw_order = raw_sh
            _save_order_thresholds(raw_order, date)
        elapsed = _time.time() - t0
        print(f"  Order agg: {_join_order_stats._cache[cache_key].height:,} rows ({elapsed:.0f}s)", flush=True)
    return df.join(_join_order_stats._cache[cache_key], on=["SecurityID", "timestamp"], how="left")


# ---- Order-derived registered features ----

@register("CancelRateImb", required_cols=_ORDER_REQUIRED_COLS)
def _compute_cancel_rate_imb(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Cancel rate imbalance: BuyCancelRate - SellCancelRate.

    CancelRate = cancel_cnt / (new_cnt + cancel_cnt).
    Higher cancel rate on one side → that side's orders are more "fake".
    """
    df = _join_order_stats(df, date)
    nbc = pl.col("new_buy_cnt").fill_null(0)
    nsc = pl.col("new_sell_cnt").fill_null(0)
    cbc = pl.col("cancel_buy_cnt").fill_null(0)
    csc = pl.col("cancel_sell_cnt").fill_null(0)

    buy_rate = pl.when(nbc + cbc > 0).then(cbc / (nbc + cbc)).otherwise(0.0)
    sell_rate = pl.when(nsc + csc > 0).then(csc / (nsc + csc)).otherwise(0.0)
    return df.with_columns((buy_rate - sell_rate).alias("CancelRateImb"))


@register("OrderImb", required_cols=_ORDER_REQUIRED_COLS)
def _compute_order_imb(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """New order imbalance: (NewBuy - NewSell) / (NewBuy + NewSell)."""
    df = _join_order_stats(df, date)
    nb = pl.col("new_buy_amt").fill_null(0.0)
    ns = pl.col("new_sell_amt").fill_null(0.0)
    denom = nb + ns
    return df.with_columns(
        pl.when(denom > 0).then((nb - ns) / denom).otherwise(0.0).alias("OrderImb")
    )


@register("LargeOrderImb", required_cols=_ORDER_REQUIRED_COLS)
def _compute_large_order_imb(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Large new order imbalance: (LargeNewBuy - LargeNewSell) / (LargeNewBuy + LargeNewSell)."""
    df = _join_order_stats(df, date)
    lb = pl.col("large_new_buy").fill_null(0.0)
    ls = pl.col("large_new_sell").fill_null(0.0)
    denom = lb + ls
    return df.with_columns(
        pl.when(denom > 0).then((lb - ls) / denom).otherwise(0.0).alias("LargeOrderImb")
    )


@register("OrderAggress", required_cols=_ORDER_REQUIRED_COLS)
def _compute_order_aggress(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Order aggressiveness: how far new order prices are from mid.

    AggrBuy = (avg_buy_price - mid) / mid  → positive = aggressive buy
    AggrSell = (mid - avg_sell_price) / mid → positive = aggressive sell
    OrderAggress = AggrBuy - AggrSell.

    Market orders (Price=0) are already excluded during aggregation;
    they are inherently the most aggressive — no price anchor.
    """
    df = _join_order_stats(df, date)
    mid = pl.col("mid_price")
    ab = pl.col("avg_price_new_buy")
    as_ = pl.col("avg_price_new_sell")

    aggr_buy = pl.when((mid > 0) & ab.is_not_null()).then((ab - mid) / mid).otherwise(None)
    aggr_sell = pl.when((mid > 0) & as_.is_not_null()).then((mid - as_) / mid).otherwise(None)
    return df.with_columns((aggr_buy - aggr_sell).alias("OrderAggress"))


@register("LargeCancelImb", required_cols=_ORDER_REQUIRED_COLS)
def _compute_large_cancel_imb(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Large cancel imbalance: (LargeCancelBuy - LargeCancelSell) / sum."""
    df = _join_order_stats(df, date)
    lb = pl.col("large_cancel_buy").fill_null(0.0)
    ls = pl.col("large_cancel_sell").fill_null(0.0)
    denom = lb + ls
    return df.with_columns(
        pl.when(denom > 0).then((lb - ls) / denom).otherwise(0.0).alias("LargeCancelImb")
    )


@register("OrderDepthPos", required_cols=_ORDER_REQUIRED_COLS)
def _compute_order_depth_pos(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Avg depth position: (avgDepthBuy - avgDepthSell) / (avgDepthBuy + avgDepthSell).

    Depth = (BidP1 - OrderPrice)/TickSize for buys, (OrderPrice - AskP1)/TickSize for sells.
    Larger depth → order placed further from best price (more passive).
    """
    df = _join_order_stats(df, date)
    db = pl.col("avg_depth_buy").cast(pl.Float64).fill_null(0.0)
    ds = pl.col("avg_depth_sell").cast(pl.Float64).fill_null(0.0)
    denom = db.abs() + ds.abs()
    return df.with_columns(
        pl.when(denom > 0).then((db - ds) / denom).otherwise(0.0).alias("OrderDepthPos")
    )


# --- Standalone cancel rates ---

@register("BuyCancelRate", required_cols=_ORDER_REQUIRED_COLS)
def _compute_buy_cancel_rate(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Buy-side cancel rate: CancelBuyCnt / (NewBuyCnt + CancelBuyCnt)."""
    df = _join_order_stats(df, date)
    n = pl.col("new_buy_cnt").cast(pl.Float64).fill_null(0)
    c = pl.col("cancel_buy_cnt").cast(pl.Float64).fill_null(0)
    return df.with_columns(
        pl.when(n + c > 0).then(c / (n + c)).otherwise(0.0).alias("BuyCancelRate")
    )


@register("SellCancelRate", required_cols=_ORDER_REQUIRED_COLS)
def _compute_sell_cancel_rate(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Sell-side cancel rate."""
    df = _join_order_stats(df, date)
    n = pl.col("new_sell_cnt").cast(pl.Float64).fill_null(0)
    c = pl.col("cancel_sell_cnt").cast(pl.Float64).fill_null(0)
    return df.with_columns(
        pl.when(n + c > 0).then(c / (n + c)).otherwise(0.0).alias("SellCancelRate")
    )


@register("LargeBuyCancelRate", required_cols=_ORDER_REQUIRED_COLS)
def _compute_large_buy_cancel_rate(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Large-order buy cancel rate."""
    df = _join_order_stats(df, date)
    n = pl.col("large_new_buy_cnt").cast(pl.Float64).fill_null(0)
    c = pl.col("large_cancel_buy_cnt").cast(pl.Float64).fill_null(0)
    return df.with_columns(
        pl.when(n + c > 0).then(c / (n + c)).otherwise(0.0).alias("LargeBuyCancelRate")
    )


@register("LargeSellCancelRate", required_cols=_ORDER_REQUIRED_COLS)
def _compute_large_sell_cancel_rate(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Large-order sell cancel rate."""
    df = _join_order_stats(df, date)
    n = pl.col("large_new_sell_cnt").cast(pl.Float64).fill_null(0)
    c = pl.col("large_cancel_sell_cnt").cast(pl.Float64).fill_null(0)
    return df.with_columns(
        pl.when(n + c > 0).then(c / (n + c)).otherwise(0.0).alias("LargeSellCancelRate")
    )


# --- Order depth distribution ---

@register("OrderDepthBestFrac", required_cols=_ORDER_REQUIRED_COLS)
def _compute_order_depth_best_frac(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Fraction of new orders at ≤1 tick from best price (most aggressive/passive boundary)."""
    df = _join_order_stats(df, date)
    b_best = pl.col("depth_best_buy").cast(pl.Float64).fill_null(0)
    s_best = pl.col("depth_best_sell").cast(pl.Float64).fill_null(0)
    b_tot = pl.col("depth_total_buy").cast(pl.Float64).fill_null(1)
    s_tot = pl.col("depth_total_sell").cast(pl.Float64).fill_null(1)
    return df.with_columns(
        ((b_best / b_tot) - (s_best / s_tot)).alias("OrderDepthBestFrac")
    )


@register("OrderDepthDeepFrac", required_cols=_ORDER_REQUIRED_COLS)
def _compute_order_depth_deep_frac(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Fraction of new orders >5 ticks from best price (deep/passive)."""
    df = _join_order_stats(df, date)
    b_deep = pl.col("depth_deep_buy").cast(pl.Float64).fill_null(0)
    s_deep = pl.col("depth_deep_sell").cast(pl.Float64).fill_null(0)
    b_tot = pl.col("depth_total_buy").cast(pl.Float64).fill_null(1)
    s_tot = pl.col("depth_total_sell").cast(pl.Float64).fill_null(1)
    return df.with_columns(
        ((b_deep / b_tot) - (s_deep / s_tot)).alias("OrderDepthDeepFrac")
    )


# ============================================================
# ⑦ 已实现矩 (Realized Moments) — mid-price snap-to-snap returns
# ============================================================
# RVol / RSkew / RKurt / UpVolRatio are computed from intra-stock
# adjacent-snapshot log-returns of mid_price, with rolling windows.
#
# r_i = ln(MidPrice_i) - ln(MidPrice_{i-1})  within each stock
# Window N = 300 snapshots (~15 min) by default.
# Zero-mean assumption per Amaya et al. (2015): r̄ ≈ 0 at tick level,
# which simplifies the formulas and improves robustness.

_REALIZED_WINDOW: int = 300  # ~15 minutes at 3s resolution
_REALIZED_MIN_PERIODS: int = 30  # min 90s of data before computing


def _realized_moment_features(df: pl.DataFrame, N: int = _REALIZED_WINDOW) -> pl.DataFrame:
    """Add realized volatility, skewness, kurtosis, and up/down vol ratio."""
    if "mid_price" not in df.columns:
        return df

    # Only compute on valid mid (排除停牌/涨跌停单边无价/0价)
    valid_mid = pl.col("mid_price") > 0
    ln_mid = pl.when(valid_mid).then(pl.col("mid_price").log()).otherwise(None)
    # Set first return of each day to 0 to avoid overnight gap contamination.
    g = _time_group(df)
    r_raw = ln_mid - ln_mid.shift(1).over(g)
    is_first = pl.col("timestamp") == pl.col("timestamp").min().over(g)
    r = pl.when(is_first).then(0.0).otherwise(r_raw)

    r2 = r.pow(2)
    r3 = r.pow(3)
    r4 = r.pow(4)
    r2_up = pl.when(r > 0).then(r2).otherwise(0.0)
    r2_down = pl.when(r < 0).then(r2).otherwise(0.0)

    # Rolling sums over N snapshots + effective sample count (per stock per day)
    is_valid = r.is_not_null().cast(pl.Float64)
    RVol = r2.rolling_sum(window_size=N, min_periods=_REALIZED_MIN_PERIODS).over(g)
    sum_r3 = r3.rolling_sum(window_size=N, min_periods=_REALIZED_MIN_PERIODS).over(g)
    sum_r4 = r4.rolling_sum(window_size=N, min_periods=_REALIZED_MIN_PERIODS).over(g)
    RVol_up = r2_up.rolling_sum(window_size=N, min_periods=_REALIZED_MIN_PERIODS).over(g)
    RVol_down = r2_down.rolling_sum(window_size=N, min_periods=_REALIZED_MIN_PERIODS).over(g)
    n_eff = is_valid.rolling_sum(window_size=N, min_periods=_REALIZED_MIN_PERIODS).over(g)

    eps = pl.lit(1e-8)

    return df.with_columns([
        RVol.alias("_rvol"),
        sum_r3.alias("_sr3"),
        sum_r4.alias("_sr4"),
        RVol_up.alias("_rvol_up"),
        RVol_down.alias("_rvol_down"),
        n_eff.alias("_n_eff"),
    ]).with_columns([
        pl.col("_rvol").alias("RVol"),
        # RSkew: sqrt(n_eff) * Σr³ / RVol^(3/2)
        pl.when((pl.col("_rvol") > eps) & (pl.col("_n_eff") > 0))
        .then(pl.col("_n_eff").sqrt() * pl.col("_sr3") / pl.col("_rvol").pow(1.5))
        .otherwise(None).alias("RSkew"),
        # RKurt: n_eff * Σr⁴ / RVol²
        pl.when((pl.col("_rvol") > eps) & (pl.col("_n_eff") > 0))
        .then(pl.col("_n_eff") * pl.col("_sr4") / pl.col("_rvol").pow(2))
        .otherwise(None).alias("RKurt"),
        pl.when(pl.col("_rvol") > eps)
        .then(((pl.col("_rvol_up") + eps) / (pl.col("_rvol_down") + eps)).log())
        .otherwise(None).alias("UpVolRatio"),
    ]).drop(["_sr3", "_sr4", "_rvol_up", "_rvol_down", "_n_eff"])


_REALIZED_COLS: list[str] = []  # only needs mid_price which LabelGenerator adds


@register("RVol", required_cols=_REALIZED_COLS)
def _compute_rvol(df: pl.DataFrame) -> pl.DataFrame:
    if "_rvol" not in df.columns:
        df = _realized_moment_features(df)
    return df


@register("RSkew", required_cols=_REALIZED_COLS)
def _compute_rskew(df: pl.DataFrame) -> pl.DataFrame:
    if "_rvol" not in df.columns:
        df = _realized_moment_features(df)
    return df


@register("RKurt", required_cols=_REALIZED_COLS)
def _compute_rkurt(df: pl.DataFrame) -> pl.DataFrame:
    if "_rvol" not in df.columns:
        df = _realized_moment_features(df)
    return df


@register("UpVolRatio", required_cols=_REALIZED_COLS)
def _compute_up_vol_ratio(df: pl.DataFrame) -> pl.DataFrame:
    if "_rvol" not in df.columns:
        df = _realized_moment_features(df)
    return df


# --- AvgTradeSize: 单笔均额 ---

@register("AvgTradeSize", required_cols=_TRADE_REQUIRED_COLS)
def _compute_avg_trade_size(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    df = _join_trade_stats(df, date)
    total_amt = pl.col("trade_buy_amt").fill_null(0.0) + pl.col("trade_sell_amt").fill_null(0.0)
    cnt = pl.col("trade_count").fill_null(1)
    return df.with_columns(
        pl.when(cnt > 0).then(total_amt / cnt).otherwise(0.0).alias("AvgTradeSize")
    )


# --- OrderArrivalIntensity: 委托到达强度 (笔/秒) ---

@register("MarketOrderFrac", required_cols=_ORDER_REQUIRED_COLS)
def _compute_market_order_frac(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """Fraction of new orders that are market orders (Price=0 or null)."""
    df = _join_order_stats(df, date)
    mkt = pl.col("market_order_cnt").fill_null(0).cast(pl.Float64)
    total = (pl.col("new_buy_cnt").fill_null(0) + pl.col("new_sell_cnt").fill_null(0)).cast(pl.Float64)
    return df.with_columns(
        pl.when(total > 0).then(mkt / total).otherwise(0.0).alias("MarketOrderFrac")
    )


@register("OrderArrivalIntensity", required_cols=_ORDER_REQUIRED_COLS)
def _compute_order_arrival_intensity(df: pl.DataFrame, date: str = "") -> pl.DataFrame:
    """New orders per second: (new_buy_cnt + new_sell_cnt) / 3."""
    df = _join_order_stats(df, date)
    n = pl.col("new_buy_cnt").fill_null(0).cast(pl.Float64) + pl.col("new_sell_cnt").fill_null(0).cast(pl.Float64)
    return df.with_columns((n / 3.0).alias("OrderArrivalIntensity"))


# ============================================================
# ⑧ 跨标的 OFI PCA (Cont, Cucuringu & Zhang 2023)
# ============================================================
# For each 3s timestamp, build n_stocks × 3 matrix of [OFI_1, OFI, OFI_10],
# compute PCA, and extract:
#   OFI_PC1     — each stock's projection onto PC1 (market-wide OFI component)
#   OFI_Residual — ‖OFI_vec − PC1_proj‖ (idiosyncratic OFI not explained by market)
#   OFI_PC1_Var — % variance explained by PC1 (market co-movement strength)

import numpy as _np


def _ofi_pca_per_timestamp(group: pl.DataFrame) -> pl.DataFrame:
    """PCA of OFI across stocks within a single (date, timestamp) cross-section.

    Input group has columns: SecurityID, [date,] timestamp, OFI_1, OFI, OFI_10.
    Returns: SecurityID, [date,] timestamp, OFI_PC1, OFI_Residual, OFI_PC1_Var.
    """
    n = group.height
    has_date = "date" in group.columns
    base_cols = ["SecurityID", "date", "timestamp"] if has_date else ["SecurityID", "timestamp"]
    if n < 20:  # too few stocks for meaningful PCA
        return group.select(base_cols).with_columns([
            pl.lit(None).alias("OFI_PC1"),
            pl.lit(None).alias("OFI_Residual"),
            pl.lit(None).alias("OFI_PC1_Var"),
        ])

    cols = ["OFI_1", "OFI", "OFI_10"]
    X = group.select(cols).to_numpy().astype(_np.float64)

    # Drop rows with any NaN
    valid = ~_np.isnan(X).any(axis=1)
    if valid.sum() < 20:
        return group.select(base_cols).with_columns([
            pl.lit(None).alias("OFI_PC1"),
            pl.lit(None).alias("OFI_Residual"),
            pl.lit(None).alias("OFI_PC1_Var"),
        ])

    Xv = X[valid]
    sec_ids = group["SecurityID"].to_numpy()[valid]

    # Z-score standardize
    mean = Xv.mean(axis=0)
    std = Xv.std(axis=0)
    std[std < 1e-10] = 1.0
    Z = (Xv - mean) / std

    # 3×3 covariance, eigendecomposition
    cov = Z.T @ Z / max(len(Z) - 1, 1)
    eigvals, eigvecs = _np.linalg.eigh(cov)
    # eigh returns ascending order; PC1 is the last eigenvector
    pc1_vec = eigvecs[:, -1].copy()  # 3-vector, loadings on [OFI_1, OFI, OFI_10]
    # Anchor sign: force OFI (index 1, the 5-level version) loading positive.
    # Prevents arbitrary sign flips between adjacent 3s buckets.
    if pc1_vec[1] < 0:
        pc1_vec = -pc1_vec
    total_var = max(float(eigvals.sum()), 1e-12)
    pc1_var = max(float(eigvals[-1]), 0.0) / total_var

    # Project each stock onto PC1
    pc1_proj = Z @ pc1_vec          # n-vector
    residual = Z - pc1_proj[:, None] * pc1_vec[None, :]
    resid_norm = _np.sqrt((residual ** 2).sum(axis=1))  # n-vector

    out = {
        "SecurityID": sec_ids,
        "timestamp": [group["timestamp"][0]] * len(sec_ids),
        "OFI_PC1": pc1_proj,
        "OFI_Residual": resid_norm,
        "OFI_PC1_Var": [pc1_var] * len(sec_ids),
    }
    if has_date:
        out["date"] = [group["date"][0]] * len(sec_ids)
    return pl.DataFrame(out)


def _compute_ofi_pca(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cross-sectional OFI PCA for all timestamps.

    Requires OFI_1, OFI, OFI_10 to exist in df.
    """
    # Ensure OFI variants exist
    for fn in ["OFI_1", "OFI", "OFI_10"]:
        if fn not in df.columns:
            need_cols = feature_registry[fn]["required_cols"]
            if all(c in df.columns for c in need_cols):
                df = feature_registry[fn]["func"](df)

    if not all(c in df.columns for c in ["OFI_1", "OFI", "OFI_10"]):
        return df  # can't compute

    # Use date-aware grouping to prevent cross-day section mixing
    has_date = "date" in df.columns
    g_cols = ["date", "timestamp"] if has_date else ["timestamp"]
    select_cols = ["SecurityID"] + g_cols + ["OFI_1", "OFI", "OFI_10"]

    # Keep only rows with all three OFI values
    sub = df.select(select_cols).drop_nulls()

    # Per-timestamp PCA via group_by + map_groups
    result = sub.group_by(g_cols, maintain_order=True).map_groups(
        _ofi_pca_per_timestamp
    )

    # Join back to main df
    join_on = ["SecurityID"] + g_cols
    result_cols = ["SecurityID"] + g_cols + ["OFI_PC1", "OFI_Residual", "OFI_PC1_Var"]
    return df.join(
        result.select(result_cols),
        on=join_on, how="left"
    )


_OFI_PCA_COLS = _OFI1_COLS + _OFI5_COLS + _OFI10_COLS  # all OFI price+volume columns


@register("OFI_PC1", required_cols=_OFI_PCA_COLS)
def _compute_ofi_pc1(df: pl.DataFrame) -> pl.DataFrame:
    if "_ofi_pca_done" in df.columns:
        return df
    df = _compute_ofi_pca(df)
    return df.with_columns(pl.lit(1).alias("_ofi_pca_done"))


@register("OFI_Residual", required_cols=_OFI_PCA_COLS)
def _compute_ofi_residual(df: pl.DataFrame) -> pl.DataFrame:
    if "_ofi_pca_done" in df.columns:
        return df
    df = _compute_ofi_pca(df)
    return df.with_columns(pl.lit(1).alias("_ofi_pca_done"))


@register("OFI_PC1_Var", required_cols=_OFI_PCA_COLS)
def _compute_ofi_pc1_var(df: pl.DataFrame) -> pl.DataFrame:
    if "_ofi_pca_done" in df.columns:
        return df
    df = _compute_ofi_pca(df)
    return df.with_columns(pl.lit(1).alias("_ofi_pca_done"))


# ============================================================
# Feature Factory
# ============================================================

class FeatureFactory:
    """Orchestrate feature computation with disk caching.

    Cache key: [timestamp (Int32), SecurityID (str), value (Float64)].
    Join uses exact match on [timestamp, SecurityID] — never join_asof.
    """

    def __init__(self, cache_root: str = FACTOR_CACHE_ROOT) -> None:
        self._cache_root = cache_root

    # ---- Cache paths ----

    def _cache_dir(self, factor_name: str) -> str:
        return os.path.join(self._cache_root, factor_name)

    def _cache_path(self, factor_name: str, date: str) -> str:
        return os.path.join(self._cache_dir(factor_name), f"{date}.parquet")

    # ---- Cache read/write ----

    def cache_exists(self, factor_name: str, date: str) -> bool:
        return os.path.exists(self._cache_path(factor_name, date))

    def read_cache(self, factor_name: str, date: str) -> pl.DataFrame:
        """Read a cached factor parquet. Returns [timestamp, SecurityID, value]."""
        path = self._cache_path(factor_name, date)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache miss: {path}")
        return pl.read_parquet(path)

    def write_cache(self, df: pl.DataFrame, factor_name: str, date: str) -> None:
        """Write factor values to cache.

        Expects df to have columns [timestamp, SecurityID, {factor_name}].
        Writes [timestamp, SecurityID, value] to parquet.
        """
        os.makedirs(self._cache_dir(factor_name), exist_ok=True)

        out = df.select([
            pl.col("timestamp"),
            pl.col("SecurityID"),
            pl.col(factor_name).alias("value"),
        ]).drop_nulls(subset=["value"])

        out.write_parquet(
            self._cache_path(factor_name, date),
            compression="zstd",
        )

    def join_cache(self, df: pl.DataFrame, factor_name: str, date: str) -> pl.DataFrame:
        """Join a single cached factor onto df using exact [timestamp, SecurityID] match."""
        cache_df = self.read_cache(factor_name, date)
        return df.join(
            cache_df,
            on=["timestamp", "SecurityID"],
            how="left",
        ).rename({"value": factor_name})

    # ---- Computation ----

    def compute_single(
        self, df: pl.DataFrame, factor_name: str, date: str,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """Compute or load one factor, optionally caching.

        Parameters
        ----------
        df : DataFrame with all required columns + timestamp, SecurityID.
        factor_name : registered factor name.
        date : date string for cache.
        use_cache : if True and cache exists, skip computation.

        Returns
        -------
        df with the factor column added.
        """
        if use_cache and self.cache_exists(factor_name, date):
            return self.join_cache(df, factor_name, date)

        if factor_name not in feature_registry:
            raise KeyError(
                f"Factor '{factor_name}' not registered. "
                f"Available: {list(feature_registry.keys())}"
            )

        entry = feature_registry[factor_name]
        compute_fn = entry["func"]

        # ---- Defensive sort: guarantee ordered rows for shift/ewm/rolling ----
        sort_keys = ["SecurityID", "date", "timestamp"] if "date" in df.columns else ["SecurityID", "timestamp"]
        df = df.sort(sort_keys)

        # ---- Validate required columns ----
        req_cols = entry.get("required_cols", [])
        if req_cols:
            missing = [c for c in req_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Factor '{factor_name}' missing required columns: {missing}"
                )

        # Try passing date for features that need to load trade/order data.
        try:
            df = compute_fn(df, date=date)
        except TypeError:
            df = compute_fn(df)

        # Ensure the factor column exists
        if factor_name not in df.columns:
            raise RuntimeError(
                f"Factor function for '{factor_name}' did not produce column '{factor_name}'"
            )

        self.write_cache(df, factor_name, date)
        return df

    def compute_many(
        self, df: pl.DataFrame, factor_names: list[str], date: str,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """Compute or load multiple factors, each cached independently."""
        for fn in factor_names:
            df = self.compute_single(df, fn, date, use_cache=use_cache)

        return df

    def list_available(self) -> list[str]:
        return sorted(feature_registry.keys())
