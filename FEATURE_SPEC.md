# 高频 Alpha 因子特征清单 — 计算逻辑文档

> 共 75 个特征，覆盖 8 大类。所有时序操作均使用 `_time_group(df)` 防跨股/跨日数据污染。

---

## 数据环境

### 目录结构

```
/fast1/user001/stock_data/
├── type=snap_sh/date={YYYYMMDD}/data.parquet    # 沪市 L2 快照, 77 列
├── type=snap_sz/date={YYYYMMDD}/data.parquet    # 深市 L2 快照, 76 列
├── type=trade_sh/date={YYYYMMDD}/data.parquet   # 沪市逐笔成交, 10 列
├── type=trade_sz/date={YYYYMMDD}/data.parquet   # 深市逐笔成交, 9 列
├── type=order_sh/date={YYYYMMDD}/data.parquet   # 沪市逐笔委托, 9 列
└── type=order_sz/date={YYYYMMDD}/data.parquet   # 深市逐笔委托, 8 列

/fast1/user001/factor_values/{factor_name}/{YYYYMMDD}.parquet  # 特征缓存

/home/user001/data/univ_20260514.csv             # 选股池 (1000只)
```

- 数据范围：2025.12.01 ~ 2026.05.29（114 个交易日）
- 选股池：1000 只（无 ST，涨跌幅全部 10% 或 20%）
- 快照粒度：~3 秒，全天约 4000 个截面/股
- 所有快照、成交、委托按交易所分文件存储，加载时 SH+SZ 合并

---

### 快照数据 (Snap) 字段

#### 沪市 (SH) 独有列 — 8 个

| 字段 | 类型 | 说明 |
|------|------|------|
| `MaxBidDur` | Int32 | 买侧最长挂单持续时间（开盘累计，近似单调递增） |
| `MaxSellDur` | Int32 | 卖侧最长挂单持续时间 |
| `TotBidNum` | Int32 | 买侧总委托笔数 |
| `TotSellNum` | Int32 | 卖侧总委托笔数 |
| `TotalBidVol` | Float32 | 买侧总委托量 |
| `TotalAskVol` | Float32 | 卖侧总委托量 |
| `TradNumber` | Int32 | 成交笔数 |
| `TradVolume` | Float32 | 成交量 |

#### 深市 (SZ) 独有列 — 3 个

| 字段 | 类型 | 说明 |
|------|------|------|
| `HighLimitPrice` | Float32 | 涨停价（交易所原生计算） |
| `LowLimitPrice` | Float32 | 跌停价（交易所原生计算） |
| `TradingPhaseCode` | String | 交易阶段代码 |

#### 两市共有列 — 69 个

| 字段 | 类型 | 说明 |
|------|------|------|
| `UpdateTime` | String | 快照时间 `HH:MM:SS.mmm` |
| `SecurityID` | String | 6 位股票代码 |
| `PreCloPrice` | Float32 | 昨收盘价 |
| `OpenPrice` | Float32 | 开盘价 |
| `HighPrice` | Float32 | 日内最高价 |
| `LowPrice` | Float32 | 日内最低价 |
| `LastPrice` | Float32 | 最新成交价 |
| `Turnover` | Float32 | 成交额 |
| `BidPrice1` ~ `BidPrice10` | Float32 | 买 1 ~ 买 10 价格 |
| `BidVolume1` ~ `BidVolume10` | Float32/Int32 | 买 1 ~ 买 10 挂单量（SH=Float32, SZ=Int32） |
| `AskPrice1` ~ `AskPrice10` | Float32 | 卖 1 ~ 卖 10 价格 |
| `AskVolume1` ~ `AskVolume10` | Float32/Int32 | 卖 1 ~ 卖 10 挂单量 |
| `NumOrdersB1` ~ `NumOrdersB10` | Int32 | 买 1 ~ 买 10 委托笔数 |
| `NumOrdersS1` ~ `NumOrdersS10` | Int32 | 卖 1 ~ 卖 10 委托笔数 |

#### 沪深深快照列名映射 (loader 自动处理)

| 概念 | 沪市列名 | 深市列名 |
|------|---------|---------|
| 总买量 | `TotalBidVol` | `TotalBidQty` |
| 总卖量 | `TotalAskVol` | `TotalOfferQty` |
| 成交笔数 | `TradNumber` | `TurnNum` |
| 成交量 | `TradVolume` | `Volume` |

#### 关键处理

- 深市涨跌停价保留原生列。沪市无该列，loader 自动从 `PreCloPrice` 推算（688/300→20%, 其余→10%）。
- 沪市挂单量为 Float32，深市为 Int32 → loader 统一 cast 为 Float64。
- 深市独有的 `MaxBidDur`/`MaxSellDur` 相关特征，深市股票值为 null。
- 原始 `UpdateTime` 只在加载时解析一次，转为 3s 网格秒数 `timestamp`（Int32）作为后续所有 join 的主键。

---

### 逐笔成交 (Trade) 字段

#### 沪市 (SH)

| 字段 | 类型 | 说明 |
|------|------|------|
| `SecurityID` | String | 股票代码 |
| `TradTime` | String | 成交时间 `HH:MM:SS.mmm` |
| `TradPrice` | Float32 | 成交价 |
| `TradVolume` | Float32 | 成交量（股） |
| `TradeMoney` | Float32 | 成交额（元） |
| `TradeBSFlag` | String | 买卖方向：`B`=主动买, `S`=主动卖 |
| `TradeIndex` | Int32 | 全市场递增成交序号（用于精确排序） |
| `TradeBuyNo` | Int32 | 买方委托编号 |
| `TradeSellNo` | Int32 | 卖方委托编号 |
| `LocalTime` | String | 本地接收时间 |

#### 深市 (SZ)

| 字段 | 类型 | 说明 |
|------|------|------|
| `SecurityID` | String | 股票代码 |
| `TransactTime` | String | 成交时间 `HH:MM:SS.mmm` |
| `LastPx` | Float32 | 成交价 |
| `LastQty` | Int32 | 成交量（股） |
| `ExecType` | Int32 | **含撤单！** `70`(F)=真实成交, `52`(4)=撤单混入 |
| `ApplSeqNum` | Int32 | 全市场递增序号（用于精确排序） |
| `BidApplSeqNum` | Int32 | 买方委托序号 |
| `OfferApplSeqNum` | Int32 | 卖方委托序号 |
| `LocalTime` | String | 本地接收时间 |

#### 关键处理

| 问题 | 处理 |
|------|------|
| SZ `ExecType=52` 是撤单混入成交流 | 全部过滤，只保留 `ExecType=70` |
| SZ 无原生 `TradeMoney` | 用 `LastPx × LastQty` 计算 |
| SZ 无 `TradeBSFlag` | 用 Lee-Ready tick rule 判方向 |
| SZ `Price=0` (ExecType=52) | 撤单无成交价，已一并过滤 |
| 排序不可靠 | 按 `(TradTime/TransactTime, TradeIndex/ApplSeqNum)` 联合排序 |

---

### 逐笔委托 (Order) 字段

#### 沪市 (SH)

| 字段 | 类型 | 说明 |
|------|------|------|
| `SecurityID` | String | 股票代码 |
| `OrderTime` | String | 委托时间 `HH:MM:SS.mmm` |
| `OrderPrice` | Float32 | 委托价格 |
| `Balance` | Float32 | 委托数量（股） |
| `OrderBSFlag` | String | 买卖方向：`B`=买, `S`=卖 |
| `OrderType` | String | 委托类型：`A`=新增, `D`=撤单 |
| `OrderIndex` | Int32 | 全市场递增委托序号 |
| `OrderNO` | Int32 | 委托编号 |
| `LocalTime` | String | 本地接收时间 |

#### 深市 (SZ)

| 字段 | 类型 | 说明 |
|------|------|------|
| `SecurityID` | String | 股票代码 |
| `TransactTime` | String | 委托时间 `HH:MM:SS.mmm` |
| `Price` | Float32 | 委托价格 |
| `OrderQty` | Int32 | 委托数量（股） |
| `Side` | Int32 | **编码**：`49`(ASCII '1')=买, `50`(ASCII '2')=卖 |
| `OrdType` | Int32 | **编码**：`49`(ASCII '1')=新增, `50`(ASCII '2')=撤单, `85`(ASCII 'U')=修改 |
| `ApplSeqNum` | Int32 | 全市场递增委托序号 |
| `LocalTime` | String | 本地接收时间 |

#### 关键处理

| 问题 | 处理 |
|------|------|
| SZ `Side`/`OrdType` 是 Int32 编码 | loader 解码为 'B'/'S' 和 'A'/'D'/'U' |
| SZ `Price=0` (64,237 笔, 0.25%) | 市价单，从价格敏感特征中剔除 |
| SZ `OrdType=85`(修改单) | 数量极少(9,533 笔)，当前忽略 |
| SH 无 `Price=0` | 沪市数据干净，无需过滤 |

---

## 特征清单

### ① 盘口静态结构（单快照，无时序运算）— 12 个

### Spread — 绝对价差
```
Spread = AskPrice1 − BidPrice1
```
- 条件：BP1 > 0 且 AP1 > 0，否则 NaN。单位：元。

### SpreadRel — 相对价差
```
SpreadRel = (AskPrice1 − BidPrice1) / MidPrice,  MidPrice = (AskPrice1 + BidPrice1) / 2
```
- 无量纲，跨股票可比。

### OBI — 订单簿不平衡（一档 QI）
```
OBI = (BV1 − SV1) / (BV1 + SV1)
```
- 分母 = 0 时返回 0。值域 [−1, 1]，正 = 买盘压倒卖盘。

### MicroPrice — Stoikov 微观价格 (2017, SSRN:2970694)
```
MicroPrice = (AskP1 × BV1 + BidP1 × SV1) / (BV1 + SV1)
```
- 用对手方量做权重。条件：BP1>0, AP1>0, BV1+SV1>0。

### MicroPriceBias — 微观价格偏离
```
MicroPriceBias = MicroPrice / MidPrice − 1
```
- 正 → 公允价高于中价 → 买方压力大。

### OCIB_1 / OCIB_5 / OCIB_10 — 委托笔数不平衡
```
OCIB_k = (Σ_{i=1..k} NumOrdersB_i − Σ_{i=1..k} NumOrdersS_i)
       / (Σ_{i=1..k} NumOrdersB_i + Σ_{i=1..k} NumOrdersS_i)
```
- k = 1, 5, 10。A 股特色：笔数比委托量更"干净"。

### Depth_Imbalance — 多档加权 QI（指数衰减版）
```
Bid: weight_i = exp(−κ × (BP1 − BP_i) / TickSize)
Ask: weight_i = exp(−κ × (AP_i − AP1) / TickSize)

DepthImb = (WeightedBid − WeightedAsk) / (WeightedBid + WeightedAsk)
```
- κ = 1.0, TickSize = 0.01。偏离最优价越远 → 权重指数衰减。

### MaxDurPressure — 耐心资本流向
```
ΔBidDur  = MaxBidDur[t] − MaxBidDur[t−1]
ΔSellDur = MaxSellDur[t] − MaxSellDur[t−1]

MaxDurPressure = (ΔBidDur − ΔSellDur) / (ΔBidDur + ΔSellDur)
```
- ⚠️ 深交所无此列，深市股票全为 null。

---

## ② 订单流不平衡（跨快照增量）— 价格感知 OFI — 8 个

### OFI / OFI_1 / OFI_3 / OFI_10 — Cont-Kukanov-Stoikov (2014)

每档独立判断价格跳变后再算量差：

**买侧（Bid）:**
```
BidP_i[t] > BidP_i[t−1] → e = BV_i[t]           （价格推高，全是新买单）
BidP_i[t] = BidP_i[t−1] → e = BV_i[t] − BV_i[t−1] （价格不变，净流入）
BidP_i[t] < BidP_i[t−1] → e = −BV_i[t−1]         （价格溃退，旧单全撤）
```

**卖侧（Ask）:**
```
AskP_i[t] > AskP_i[t−1] → f = −SV_i[t−1]         （价格推高，旧卖单被吃）
AskP_i[t] = AskP_i[t−1] → f = SV_i[t] − SV_i[t−1] （价格不变，净流入）
AskP_i[t] < AskP_i[t−1] → f = SV_i[t]            （价格下压，全是新卖单）
```

**汇总:**
```
OFI_i = e − f
OFI_k = Σ_{i=1..k} OFI_i / Σ_{i=1..k} (BV_i + SV_i)
```
- k = 1 (`OFI_1`), 3 (`OFI_3`), 5 (`OFI`), 10 (`OFI_10`)

### TS_Imbalance — 一档价格感知流（变化量归一化）
```
同 OFI_1 的 e − f，分母用 |e| + |f| 替代存量归一化。对短期突发流更敏感。
```

### OFI_MA — OFI 指数移动均线
```
OFI_MA = EMA(OFI, span=20) ≈ 60s 窗口, alpha = 2/(20+1)
```
- 每天开盘 EMA 自动重置。

### OFI_Z — OFI 滚动 Z-score
```
OFI_diff = OFI − OFI_MA
OFI_std  = sqrt(EMA(OFI_diff², span=20))
OFI_Z = OFI_diff / OFI_std
```

### OFI_Decay — OFI 指数衰减加权
```
OFI_Decay = Σ_{k=0..4} OFI_{t−k} × exp(−k)  ≈ 15s 窗口
```

---

## ③ 多档 OBI & 金额加权 OBI — 8 个

### OBI_3 / OBI_5 / OBI_10 — 多档量不平衡
```
OBI_k = (Σ_{i=1..k} BV_i − Σ_{i=1..k} SV_i) / (Σ_{i=1..k} BV_i + Σ_{i=1..k} SV_i)
```

### AmtOBI_1 / AmtOBI_3 / AmtOBI_5 / AmtOBI_10 — 金额加权不平衡
```
AmtOBI_k = (Σ BV_i×BidP_i − Σ SV_i×AskP_i) / (Σ BV_i×BidP_i + Σ SV_i×AskP_i)
```
- 用金额替代裸量，大盘股和小盘股在因子层面更可比。

---

## ④ 盘口形状特征 — 6 个

### BookSlope — 累计量回归斜率
```
CBV_i = Σ_{j=1..i} BV_j,  X = Price_i,  Y = CBV_i
slope = (5×ΣXY − ΣX×ΣY) / (5×ΣX² − (ΣX)²)
BookSlope = (−BidSlope − AskSlope) / (总深度 × MidPrice)
```
- 正 → 买侧深度衰减更陡（近端支撑更强）。

### BookConvexity — 累计量凸性（二阶导）
```
slope_13 = (CBV_3 − CBV_1) / max(BP1−BP3, 0.01)
slope_35 = (CBV_5 − CBV_3) / max(BP3−BP5, 0.01)
BookConvexity = (Bid_slope_35 − Bid_slope_13) − (Ask_slope_35 − Ask_slope_13)
```
- 价格间距归一化。正 → 买侧深度在远档加速增厚。

### DepthConcentration — 最优档集中度
```
= (BV1 / ΣBV_1..10) − (SV1 / ΣSV_1..10)
```

### TopDepthRatio — 最优档占十档比
```
= (BV1 + SV1) / Σ_{i=1..10} (BV_i + SV_i)
```

### VWAP_Deviation — 加权价差偏离
```
BidVWAP = Σ(BP_i × BV_i) / ΣBV_i,  AskVWAP = Σ(AP_i × SV_i) / ΣSV_i  (i=1..5)
VWAP_Deviation = (AskVWAP − BidVWAP) / MidPrice
```

### AvgOrderSizeImb — 平均单笔委托量不平衡
```
AvgBidSize = ΣBV_1..5 / ΣNumOrdersB_1..5
AvgAskSize = ΣSV_1..5 / ΣNumOrdersS_1..5
AvgOrderSizeImb = (AvgBidSize − AvgAskSize) / (AvgBidSize + AvgAskSize)
```
- 单笔均量区分散户 vs 机构。

---

## ⑤ A 股涨跌停特征 — 12 个

### 涨停价/跌停价来源
- **深市**：原生 `HighLimitPrice` / `LowLimitPrice` 列 → loader 保留
- **沪市**：无原生列，从 `PreCloPrice` 推算：688→±20%, 300→±20%, 其余→±10%
- ST 股票（±5%）无法从代码自动识别（选股池已剔除）

### LimitUpDist / LimitDownDist — 距涨跌停距离
```
LimitUpDist   = (HighLimitPrice − LastPrice) / HighLimitPrice    ∈ [0, 1]
LimitDownDist = (LastPrice − LowLimitPrice)  / LastPrice         ∈ [0, 1]
```

### NearLimitUp / NearLimitDown — 接近涨跌停
```
NearLimitUp   = (LimitUpDist < 0.02) AND NOT IsLimitUp
NearLimitDown = (LimitDownDist < 0.02) AND NOT IsLimitDown
```
- 2% 阈值。"接近"和"已封死"是不同市场状态。

### IsLimitUp / IsLimitDown — 封板判定
```
IsLimitUp:   AskPrice1 消失 AND BidPrice1 > 0 AND |BidPrice1 − 涨停价| < 0.005
IsLimitDown: BidPrice1 消失 AND AskPrice1 > 0 AND |AskPrice1 − 跌停价| < 0.005
```
- 0.005 是浮点精度保护，非容差系数。BidPrice1 > 0 排除停牌。

### IsGapLimitUp / IsGapLimitDown — 一字板
```
IsGapLimitUp   = IsLimitUp AND (该股票第一个 3s 桶已封板)
IsGapLimitDown = IsLimitDown AND (该股票第一个 3s 桶已封板)
```

### LimitAskScarcity / LimitBidScarcity — 涨跌停附近盘口稀缺
```
NearLimitUp 时:  1 − AskVol_1..3 / (BidVol_1..3 + AskVol_1..3)
NearLimitDown 时: 1 − BidVol_1..3 / (BidVol_1..3 + AskVol_1..3)
正常时: 0
```
- 高值 → 反向挂单极度稀缺 → 容易封板/难以开板。

### LimitBlockAmt — 封单金额
```
涨停: BidVolume1 × BidPrice1,  跌停: AskVolume1 × AskPrice1,  正常: 0
```

---

## ⑥ 逐笔成交特征 — 11 个

> SH 用 `TradeBSFlag`（B/S），SZ 用 tick rule（Lee-Ready）判方向。SZ 的 `ExecType=52`（Price=0）是混入成交流的撤单，已过滤。大单阈值 = 该股票当日成交额 90% 分位。

### TradeImb — 成交流不平衡
```
TradeImb = (BuyAmt − SellAmt) / (BuyAmt + SellAmt)
```
- 深市无 TradeMoney，用 `LastPx × LastQty` 计算。

### TradeVWAPDev — 成交均价偏离
```
TradeVWAPDev = Σ(TradPrice × TradVolume) / ΣTradVolume / MidPrice − 1
```
- 正 → 成交价高于中价（买方在追）。

### TradePriceDev — 成交价离散度
```
= mean(|TradPrice − MidPrice| / MidPrice) per 3s bucket
```

### TradePenetration — 盘口穿透
```
PenBuy  = Σ TradeMoney where BS=买 AND TradPrice > AskP1
PenSell = Σ TradeMoney where BS=卖 AND TradPrice < BidP1
TradePenetration = (PenBuy − PenSell) / (BuyAmt + SellAmt)
```
- 正 → 买方激进穿透卖一价成交。

### TradeIntensity — 成交笔数
```
TradeIntensity = 3s bucket 内成交笔数（原始值）
```

### TradeIntensityZ — 成交笔数 Z-score
```
TradeIntensityZ = (TradeIntensity − 该桶中位数) / (MAD × 1.4826)
```
- 每 timestamp 跨股票截面标准化。不受开盘脉冲影响。

### ConsecutiveBS — 连续同向成交
```
按 (TradTime, TradeIndex) 排序，追踪方向翻转分组。
maxConsecBuy  = 3s 桶内最长连续 B 笔数
maxConsecSell = 3s 桶内最长连续 S 笔数
ConsecutiveBS = (maxConsecBuy − maxConsecSell) / (maxConsecBuy + maxConsecSell)
```

### BuySellCountImb — 买卖笔数不平衡
```
= (BuyCount − SellCount) / (BuyCount + SellCount)
```

### LargeTradeRatio — 大单成交不平衡
```
LargeTradeRatio = (LargeBuy − LargeSell) / (LargeBuy + LargeSell)
```

### AvgTradeSize — 单笔成交均额
```
= (BuyAmt + SellAmt) / TradeCount
```

---

## ⑦ 逐笔委托特征 — 14 个

> SH: `OrderBSFlag` (B/S), `OrderType` (A=新增/D=撤单)。SZ: `Side` (49=买/50=卖), `OrdType` (49=新增/50=撤单/85=修改)。SZ 的 `Price=0`（市价单）已从价格敏感特征中剔除。大单阈值 = 该股票当日委托量 90% 分位。

### CancelRateImb — 撤单率不平衡
```
BuyCancelRate  = CancelBuyCnt  / (NewBuyCnt  + CancelBuyCnt)
SellCancelRate = CancelSellCnt / (NewSellCnt + CancelSellCnt)
CancelRateImb  = BuyCancelRate − SellCancelRate
```

### BuyCancelRate / SellCancelRate — 买卖撤单率（单独）
```
BuyCancelRate  = CancelBuyCnt  / (NewBuyCnt  + CancelBuyCnt)
SellCancelRate = CancelSellCnt / (NewSellCnt + CancelSellCnt)
```

### LargeBuyCancelRate / LargeSellCancelRate — 大单撤单率（单独）
```
LargeBuyCancelRate  = LargeCancelBuyCnt  / (LargeNewBuyCnt  + LargeCancelBuyCnt)
LargeSellCancelRate = LargeCancelSellCnt / (LargeNewSellCnt + LargeCancelSellCnt)
```

### OrderImb — 新增委托不平衡
```
= (NewBuyAmt − NewSellAmt) / (NewBuyAmt + NewSellAmt)
```

### LargeOrderImb — 大单新增不平衡
```
= (LargeNewBuy − LargeNewSell) / (LargeNewBuy + LargeNewSell)
```

### LargeCancelImb — 大单撤单不平衡
```
= (LargeCancelBuy − LargeCancelSell) / (LargeCancelBuy + LargeCancelSell)
```

### OrderAggress — 委托激进程度
```
AggrBuy  = (avg 新买价 − MidPrice) / MidPrice
AggrSell = (MidPrice − avg 新卖价) / MidPrice
OrderAggress = AggrBuy − AggrSell
```
- 正 → 买方委托更激进。剔除 Price=0 市价单。

### OrderDepthPos — 委托深度位置
```
BuyDepth  = (BidP1 − OrderPrice) / TickSize （0=最优价, >0=深档）
SellDepth = (OrderPrice − AskP1) / TickSize
OrderDepthPos = (avgBuyDepth − avgSellDepth) / (|avgBuyDepth| + |avgSellDepth|)
```
- 剔除 Price=0 市价单，clip 到 [−500, 500] ticks。

### OrderDepthBestFrac / OrderDepthDeepFrac — 委托深度分布
```
Best = (buy 最优价占比 − sell 最优价占比)   // ≤1 tick
Deep = (buy 深档占比 − sell 深档占比)       // >5 ticks
```

### OrderArrivalIntensity — 委托到达强度
```
= (NewBuyCnt + NewSellCnt) / 3   // 笔/秒
```

---

## ⑧ 跨标的 OFI PCA & 已实现矩 — 7 个

### OFI_PC1 / OFI_Residual / OFI_PC1_Var — 跨标的 OFI PCA
```
每个 3s 截面: 取所有股票的 [OFI_1, OFI, OFI_10] 向量 → z-score → 3×3 协方差矩阵
→ 特征分解 → PC1 = 最大特征向量
OFI_PC1 = 每只股票在 PC1 上的投影（市场型 OFI）
OFI_Residual = ‖OFI_vec − PC1_proj‖（特异性 OFI，不能被市场解释的部分）
OFI_PC1_Var = PC1 解释的方差比
```
- Cont, Cucuringu & Zhang (2023)。PC1 平均解释 ~73% 方差。

### RVol — 已实现波动（Realized Volatility）
```
r_i = ln(MidPrice_i) − ln(MidPrice_{i−1})  同股票相邻快照
RVol = Σ_{i=1..N} r_i²    N=300 (~15min)
```
- 零均值假设 (Amaya et al., 2015)。窗口 N=300 快照，最少 30 点。

### RSkew — 已实现偏度
```
RSkew = sqrt(N) × Σr_i³ / RVol^(3/2)
```
- 正偏度 → 日内有极端向上拉升（A 股彩票偏好 → 负向 IC）。

### RKurt — 已实现峰度
```
RKurt = N × Σr_i⁴ / RVol²
```
- 高峰度 → 偶尔极端跳跃（Jump），微观结构脆弱 → 负向 IC。

### UpVolRatio — 上下行波动比
```
r_i⁺ = max(r_i, 0),  r_i⁻ = min(r_i, 0)
RVol_up = Σ(r_i⁺)², RVol_down = Σ(r_i⁻)²
UpVolRatio = ln((RVol_up + ε) / (RVol_down + ε))
```
- 对称 epsilon 防除零。正 → 上涨波动 > 下跌波动。

---

## 其他

### Vol_Spread — 深度比对数
```
Vol_Spread = ln(ΣBV_1..5 / ΣSV_1..5)
```
- 正 → 买方总深度大于卖方。
