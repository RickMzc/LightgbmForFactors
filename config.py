"""
Shared configuration for the ML alpha mining framework.

All time values are in 3s-bucket seconds (seconds since midnight, floored to 3s).
Key reference points:
  9:30 = 34200
  11:30 = 41400
  13:00 = 46800
  15:00 = 54000
"""

import os

# ---- Thread control (must precede numpy import in callers) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---- Horizon definitions ----
# shift steps (each step ≈ 3s) → label column name
HORIZONS: dict[str, int] = {
    "ret_15s":  5,
    "ret_30s":  10,
    "ret_60s":  20,
    "ret_180s": 60,
    "ret_300s": 100,
}
HORIZON_NAMES: list[str] = list(HORIZONS.keys())
HORIZON_SHIFTS: list[int] = list(HORIZONS.values())

# ---- Trading calendar (3s-bucket seconds) ----
MARKET_OPEN: int  = 34200   # 09:30:00
LUNCH_START: int  = 41400   # 11:30:00
LUNCH_END: int    = 46800   # 13:00:00
MARKET_CLOSE: int = 54000   # 15:00:00

TRADING_HOURS: tuple[str, str] = ("09:30:00", "15:00:00")
TRADING_HOURS_SEC: tuple[int, int] = (MARKET_OPEN, MARKET_CLOSE)

# ---- Data paths ----
DATA_ROOT: str = "/fast1/user001/stock_data"
FACTOR_CACHE_ROOT: str = "/fast1/user001/factor_values"

# ---- Constants ----
MIN_STOCKS: int = 10
EPS: float = 1e-8

# ---- Snap base columns (always loaded) ----
SNAP_REQUIRED_COLS: list[str] = [
    "UpdateTime", "SecurityID",
    "BidPrice1", "AskPrice1",
    "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
    "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
]

# ---- Feature column list (bid/ask full 10-level for feature computation) ----
FEATURE_PRICE_COLS: list[str] = [f"BidPrice{i}" for i in range(1, 11)] + \
                                  [f"AskPrice{i}" for i in range(1, 11)]
FEATURE_VOLUME_COLS: list[str] = [f"BidVolume{i}" for i in range(1, 11)] + \
                                   [f"AskVolume{i}" for i in range(1, 11)]
FEATURE_ORDER_COLS: list[str] = [f"NumOrdersB{i}" for i in range(1, 11)] + \
                                  [f"NumOrdersS{i}" for i in range(1, 11)]
