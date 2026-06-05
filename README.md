# ML Framework for A-Share High-Frequency Alpha Mining

> 基于 3s 快照 + 逐笔成交/委托的 75 个高频 Alpha 因子，支持 LightGBM 训练与 Walk-Forward 截面 IC 评估。

---

## 目录结构

```
ml_framework/
├── __init__.py           # 包入口，导出核心类
├── config.py             # 全局常量、时间定义、数据路径
├── data_loader.py        # 数据加载（快照 + 3s网格对齐 + SH/SZ归一化）
├── label_generator.py    # 前向收益标签生成（防跨午休/跨日/涨跌停）
├── feature_factory.py    # 特征注册表 + 75个特征实现 + 缓存
├── modeling.py           # LightGBM 训练 + Walk-Forward 拆分
├── evaluation.py         # 截面 Rank-IC 评估
├── pipeline.py           # 端到端流水线入口
├── FEATURE_SPEC.md       # 75个特征的公式文档
└── README.md             # 本文件
```

---

## 数据流

```
[Snap SH+SW parquet]          [Trade SH+SW parquet]       [Order SH+SW parquet]
        │                             │                           │
        ▼                             ▼                           ▼
  SnapDataLoader               _load_trade_sh/sz            _load_order_sh/sz
  (3s bucket, sort, dedup)     (normalize, ceil bucket)     (decode SZ encoding)
        │                             │                           │
        ▼                             ▼                           ▼
  LabelGenerator               _aggregate_trades            _aggregate_orders
  (mid_price → fwd returns)    (asof-join snap, tick agg)   (amount, count, depth)
        │                             │                           │
        └──────────┬──────────────────┴───────────────┬──────────┘
                   ▼                                  ▼
             FeatureFactory.compute_many    _join_trade_stats / _join_order_stats
                   │                                  │
                   ▼                                  ▼
              [75 features in df] ──────► AlphaModel.train / predict
                                                  │
                                                  ▼
                                         CrossSectionalEvaluator
                                         (per-timestamp Spearman IC)
```

---

## 子文件详细说明

### 1. `config.py` — 全局配置

| 常量 | 值 | 说明 |
|------|-----|------|
| `HORIZONS` | 15s/30s/60s/180s/300s | 前向收益窗口 (shift steps: 5/10/20/60/100) |
| `MARKET_OPEN` | 34200 | 09:30:00 的 3s-bucket 秒数 |
| `LUNCH_START` | 41400 | 11:30:00 |
| `LUNCH_END` | 46800 | 13:00:00 (修正：非45000) |
| `MARKET_CLOSE` | 54000 | 15:00:00 |
| `DATA_ROOT` | `/fast1/user001/stock_data` | |
| `FACTOR_CACHE_ROOT` | `/fast1/user001/factor_values` | |

**线程控制**：`config.py` 导入时自动设置 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`，必须在 `import numpy` 之前执行。

---

### 2. `data_loader.py` — 数据加载

#### `SnapDataLoader` 类

| 方法 | 功能 |
|------|------|
| `load_single_market(date, market, columns)` | 懒加载单个市场快照，列裁剪，过滤交易时段 |
| `load_day_merged(date, columns)` | 加载 SH+SZ，合并，schema 对齐（Float32→Float64），3s bucket timestamp，排序，去重（同 SecurityID+timestamp 保留最后一条） |

**沪深深差异处理**：

| 概念 | 沪市列名 | 深市列名 | 处理 |
|------|---------|---------|------|
| 总买量 | `TotalBidVol` | `TotalBidQty` | loader 自动 rename |
| 总卖量 | `TotalAskVol` | `TotalOfferQty` | loader 自动 rename |
| 成交笔数 | `TradNumber` | `TurnNum` | loader 自动 rename |
| 成交量 | `TradVolume` | `Volume` | loader 自动 rename |
| 挂单量类型 | Float32 | Int32 → 统一 cast Float64 | |
| 涨停价 | 无 → 从 PreCloPrice 推算 | 原生 `HighLimitPrice` | 合并时保留原生列 |
| 最长挂单时长 | `MaxBidDur/MaxSellDur` | **无此列** → null | |

**时间戳**：原始 `UpdateTime` ("HH:MM:SS.mmm") → 解析为 3s-bucket 秒数（Int32），如 `09:30:03.123` → `34203`。该整数作为后续所有 join 的主键。

#### `TradeAggregator` 类

| 方法 | 功能 |
|------|------|
| `aggregate_trades_to_3s(trade_lf, snap_timestamps)` | 逐笔成交按 (SecurityID, 3s-bucket) 聚合，asof-join backward 到快照网格 |

**⚠️ 注意**：`df.unique()` 在 Polars 中不保持排序。Loader 已在 `unique()` 后重新 `sort(["SecurityID", "timestamp"])`。

---

### 3. `label_generator.py` — 标签生成

#### `LabelGenerator` 类

| 方法 | 功能 |
|------|------|
| `generate(df)` | 主入口：计算 mid_price + 前向收益 + 边界 mask |

**计算逻辑**：

```
mid_price = (AskPrice1 + BidPrice1) / 2
// 涨跌停处理: AskP1≤0 → 用 BidP1; BidP1≤0 → 用 AskP1; 都≤0 → NaN
// 前向收益:
ret_h = ln(mid_price.shift(-h).over("SecurityID")) - ln(mid_price)
```

**边界 NaN 规则** (3 条)：

1. **跨午休**：`ts[t] ≤ 41400` 且 `ts.shift(-h) ≥ 46800` → NaN
2. **跨日**：`ts.shift(-h) < ts[t]` → NaN
3. **超收盘**：`ts.shift(-h) > 54000` → NaN

**⚠️ 关键**：所有 `shift()` 必须加 `.over("SecurityID")`。裸 `shift(-h)` 会把茅台价格错位到平安银行。

---

### 4. `feature_factory.py` — 特征工厂（核心文件，2434 行）

#### 架构

```
@register(name, required_cols) → feature_registry 全局字典
FeatureFactory.compute_single → 查缓存 / 防排 / 校验列 / 调计算函数 / 写缓存
```

#### 全局函数

| 函数 | 功能 |
|------|------|
| `_time_group(df)` | 返回分组键：含 `date` 列则 `["SecurityID","date"]`，否则 `"SecurityID"`。所有 shift/ewm/rolling 必须用此键，防跨日污染 |
| `_parse_time_float(col)` | "HH:MM:SS.mmm" → Float64 秒数（保留毫秒，用于 asof-join） |
| `_ceil_3s(secs)` | `ceil(s/3)*3`：成交分桶用上取整，消除 floor 的未来函数 |
| `_ofi_per_level(bp,bv,ap,sv,group)` | 单档价格感知 OFI (Cont 2014) |
| `_limit_prices(cols)` | 涨停价/跌停价：深市取原生列，沪市从 PreCloPrice 推算（688/300/301→20%其余→10%） |

#### FeatureFactory 类

| 方法 | 功能 |
|------|------|
| `compute_single(df, name, date, use_cache)` | **入口防排**：`df.sort(["SecurityID","timestamp"])`；**列校验**：检查 required_cols；调用计算函数；写缓存（`[timestamp, SecurityID, value]`） |
| `compute_many(df, names, date, use_cache)` | 依次调用 compute_single |
| `read_cache(name, date)` | 读 `/fast1/user001/factor_values/{name}/{date}.parquet` |
| `write_cache(df, name, date)` | 写 parquet，压缩 zstd |
| `join_cache(df, name, date)` | **精确 join** `on=["timestamp","SecurityID"]`，不用 join_asof |

**⚠️ 缓存键**：`[timestamp (Int32), SecurityID (str), value (Float64)]`。timestamp 是 3s 网格秒数，不是原始 UpdateTime。join 用精确匹配，禁止 join_asof。

#### 75 个特征分类

| 类别 | 数量 | 代表性特征 |
|------|------|-----------|
| ① 盘口静态 | 12 | Spread/Rel, OBI, MicroPrice/Bias, OCIB×3, Depth_Imbalance, MaxDurPressure |
| ② 订单流 (OFI) | 8 | OFI×4 (价格感知), TS_Imbalance, OFI_MA/Z/Decay |
| ③ 多档 OBI | 8 | OBI_3/5/10, AmtOBI×4 |
| ④ 盘口形状 | 6 | BookSlope/Convexity, DepthConcentration, TopDepthRatio, VWAP_Deviation, AvgOrderSizeImb |
| ⑤ 涨跌停 | 12 | 距涨停/跌停, 接近/封板/一字板, 盘口稀缺, 封单金额 |
| ⑥ 逐笔成交 | 12 | TradeImb, VWAPDev, Penetration, ConsecutiveBS, LargeTradeRatio, Intensity/Z, BuySellCountImb, AvgTradeSize, TradePriceDev/Dispersion |
| ⑦ 逐笔委托 | 15 | CancelRate×5, OrderImb, LargeOrderImb/CancelImb, OrderAggress, OrderDepthPos/BestFrac/DeepFrac, MarketOrderFrac, OrderArrivalIntensity |
| ⑧ PCA+矩 | 7 | OFI_PC1/Residual/Var, RVol, RSkew, RKurt, UpVolRatio |
| 其他 | 1 | Vol_Spread |

**详细公式见 `FEATURE_SPEC.md`。**

#### 逐笔数据处理关键修正

| 问题 | 修正前 | 修正后 |
|------|--------|--------|
| SZ ExecType 混入撤单 | 不区分 | `ExecType==70` 只保留真实成交 |
| SZ 无 BS 标志 | — | tick rule (Lee-Ready)，同价 forward_fill |
| SZ OrderPrice=0 | 进入均价计算 | Price=0 的市价单单独输出 `MarketOrderFrac`，均价计算排除 |
| 成交分桶未来函数 | `floor(s/3)*3` | `ceil(s/3)*3` |
| 穿透判断错位 | 同 bucket join | `join_asof(strategy="backward")` 用成交前最近快照 |
| 大单阈值未来函数 | 同日 90% 分位 | 前日阈值缓存 `/fast1/user001/factor_values/_trade_thresh/` |
| 委托金额错当数量 | `sum(Balance)` | `sum(OrderPrice × Balance)` |
| OFI 忽略价格跳变 | 纯量 diff | 价格跳变前置判断 (Cont 2014) |
| BookConvexity Ask 间距恒=0.01 | `(P1-P3) > 0.009` | `abs(P1-P3) > 0.009` |
| ConsecutiveBS 跨桶 streak | 全日分组 | 3s 桶内独立分组 |
| OFI_Decay 开头 null | shift 无 fill | `fill_null(0.0)` |
| MaxDurPressure SZ=0 | 误判为均衡 | SZ 显式输出 null |
| OFI PCA 符号翻转 | 特征向量方向随机 | 锚定 OFI loading > 0 |
| RVol 隔夜跳空 | 首个 r 包含隔夜 | 首笔 r 置 0 |
| 跨日污染 | `over("SecurityID")` | `_time_group(df)` 含 date |

---

### 5. `modeling.py` — LightGBM 建模

#### `AlphaModel` 类

| 方法 | 功能 |
|------|------|
| `__init__(feature_cols, target_col, lgb_params)` | 默认参数：gbdt, num_leaves=63, lr=0.05, 早停=50 |
| `preprocess(df, fit, stats)` | 截面 Z-score 标准化：每 timestamp 内 `(x-mean)/std` |
| `prepare_data(df)` | 提取 X, y, stock_ids, 过滤 NaN |
| `train(X_train, y_train, X_val, y_val)` | 训练 LightGBM；若 feature 含 `stock_id` 则作 categorical |
| `predict(X)` | 推理 |
| `fit_walk_forward(df, splits)` | Walk-Forward CV，按 split 逐折训练+评估 |

#### `create_walk_forward_splits(dates, ...)`

- 日期按时间排序，训练/验证/测试按比例切分
- embargo_days：训练与验证间插入 N 天间隔
- 对短日期范围自动缩短 embargo

#### `WalkForwardSplit` dataclass

```python
train_dates: list[str]
val_dates: list[str]
test_dates: list[str]
```

---

### 6. `evaluation.py` — 截面评估

#### `CrossSectionalEvaluator` 类

| 方法 | 功能 |
|------|------|
| `rank_ic_per_timestamp(df, pred_col, label_col)` | 每个 3s bocket 内计算 Spearman ρ。Polars 原生 `pl.corr(method="spearman")` |
| `evaluate(predictions, labels, timestamps, security_ids)` | 完整评估 |
| `evaluate_multi_horizon(...)` | 多 horizon 同时评估 |
| `ic_decay_table(results)` | IC 衰减曲线 |
| `print_summary(results)` | 打印 IC/ICIR/Win%/Buckets |

#### `ICSummary` dataclass

```python
mean_ic: float    # 平均 Rank-IC
std_ic: float     # IC 标准差
icir: float       # IC / std_IC
win_rate: float   # IC > 0 的比例
n_buckets: int    # 有效截面数
```

**⚠️ NaN 处理**：Polars `drop_nulls()` 不过滤 `NaN`。代码加了 `is_not_nan()` 显式过滤。

---

### 7. `pipeline.py` — 流水线入口

#### `run_single_day(date, feature_names, target_col, use_cache)`

```
1. 根据特征声明的 required_cols 收集所需列
2. SnapDataLoader.load_day_merged(date, columns)  → 4.4M 行
3. LabelGenerator.generate(df)                    → 加 mid_price + 5 个 horizon 标签
4. FeatureFactory.compute_many(df, features, date) → 加 75 个特征列
5. 返回 df + 元信息
```

**每次只处理一天数据**，保证时序操作不跨日。

#### `run_baseline(start_date, end_date, ...)`

```
1. 逐日调 run_single_day
2. 加 date 列，concat 为全量 df
3. create_walk_forward_splits(dates)
4. AlphaModel.fit_walk_forward(df, splits)
5. CrossSectionalEvaluator 评估每个 fold
```

#### CLI 命令

```bash
python -m ml_framework.pipeline --date 20251201
python -m ml_framework.pipeline --start 20251201 --end 20251205 --features OBI OFI SpreadRel --target ret_15s
python -m ml_framework.pipeline --date 20251201 --no-cache
```

---

## 常见问题

### 时间系统
- 所有时间统一用 3s-bucket 秒数 (Int32)
- 基准点：9:30=34200, 11:30=41400, 13:00=46800 (注意不是45000), 15:00=54000
- 原始 `UpdateTime` 字符串仅在 loader 中解析一次

### 排序依赖
- `.shift()`, `.ewm_mean()`, `.rolling_sum()` 都依赖行顺序
- `compute_single` 入口强制 `df.sort(["SecurityID","timestamp"])`
- `unique()` 后必须重新 sort

### 跨股防污染
- 所有时序操作必须加 `.over("SecurityID")` 或用 `_time_group(df)`
- 含 `date` 列时自动扩为 `.over(["SecurityID","date"])`，防跨日

### 沪深深差异
- 挂单量类型不同 (Float32 vs Int32) → loader 统一 cast Float64
- 列名不同 (TotalBidQty vs TotalBidVol) → loader 自动 rename
- `MaxBidDur/MaxSellDur` 仅沪市有 → 深市 null
- `HighLimitPrice/LowLimitPrice` 仅深市有 → 沪市从 PreCloPrice 推算
- SZ 成交无 BS 标志 → tick rule
- SZ 委托 Side/OrdType 是 Int32 编码 (49/50) → loader 解码

### 缓存系统
- 磁盘：`/fast1/user001/factor_values/{特征名}/{日期}.parquet`
- 内存：trade_agg / order_agg 单日缓存，避免重复加载
- 缓存键不含代码版本 — 修改特征逻辑后需手动删除旧缓存
- `use_cache=False` 可强制重算

### 性能
- 单日快照加载：~2s
- 单日特征计算（75个）：~60-90s（含 trade/order 聚合）
- 瓶颈在逐笔数据聚合（trade~35s, order~12s）
- 建议 `nice -n 19` 降低 CPU 优先级

### 已修复的严重 Bug（备忘）
1. `unique()` 后未重排 → 已加 sort
2. OFI 未做价格跳变判断 → 已按 Cont 2014 修正
3. BookConvexity Ask 间距恒 0.01 → 已加 abs()
4. ConsecutiveBS u32 减法溢出 → 已 cast Float64
5. IsLimitUp 0.995 容差 → 已改为 `|price-limit|<0.005`
6. SZ tick rule `fill_null("B")` 偏置 → 已改 forward_fill
7. OFI PCA 符号翻转 → 已锚定
8. RVol 隔夜跳空 → 首笔 r 置 0
9. 委托数量当金额 → 已改 Price×Balance
10. 大单全天阈值 → 已改前日缓存
11. 成交 floor 分桶 → 已改 ceil
12. 穿透同桶 join → 已改 asof backward
