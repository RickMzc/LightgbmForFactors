# 高频特征代码审查报告

审查对象：

- `FEATURE_SPEC.md`
- `feature_factory.py`

审查视角：高频 A 股 Alpha 特征的研究、回测、仿真和生产可用性。重点检查计算口径、时间对齐、未来函数、订单簿/逐笔数据聚合、缓存一致性和文档一致性。

结论：这份特征体系覆盖面较完整，核心静态盘口特征和 CKS OFI 的方向性实现大体正确。但当前版本不建议直接进入严肃回测或实盘生产。主要风险集中在时间排序未强制、逐笔数据与 3 秒快照对齐不明、日内大单阈值使用全天数据、委托金额口径实际使用数量、连续成交分桶错误、缓存键过粗以及若干边界条件。若不修复，这些问题会造成未来函数、信号失真或缓存污染。

## 严重问题

### 1. 时序特征依赖输入排序，但代码没有全局强制排序

位置：

- `feature_factory.py:91-118`
- `feature_factory.py:230-269`
- `feature_factory.py:275-307`
- `feature_factory.py:575-595`
- `feature_factory.py:1902-1948`

问题：

大量特征使用 `shift(1).over("SecurityID")`、`ewm_mean(...).over(...)`、`rolling_sum(...).over(...)`。Polars 的窗口表达式按当前行顺序运算，并不会自动按 `timestamp` 排序。只要输入 DataFrame 没有严格按 `SecurityID, timestamp` 排好，OFI、TS_Imbalance、MaxDurPressure、RVol/RSkew/RKurt/UpVolRatio 都会用错上一条记录。

影响：

- OFI 的买卖压力方向可能反转。
- realized moments 会把非相邻快照当作相邻收益。
- EMA/Z-score 会在乱序数据上失真。
- 回测结果可能高度不稳定，换一次上游读取顺序就变。

建议：

- 在 `FeatureFactory.compute_single/compute_many` 入口强制排序，至少 `df = df.sort(["SecurityID", "timestamp"])`。
- 如果允许多日数据，排序键应包含 `date`：`["SecurityID", "date", "timestamp"]`。
- 对所有时序表达式统一使用包含日期的分组键，不能只靠注释说明上游已排好序。

### 2. 多日数据会跨日污染

位置：

- `feature_factory.py:91-118`
- `feature_factory.py:575-595`
- `feature_factory.py:1072-1098`
- `feature_factory.py:1902-1948`

问题：

部分 OFI 派生特征用 `["SecurityID", "date"] if "date" in df.columns else "SecurityID"`，但底层 OFI、TS_Imbalance、MaxDurPressure、IsGapLimitUp/Down、realized moments 仍只按 `SecurityID`。如果一次处理多日数据，上一交易日收盘快照会污染下一交易日开盘。

影响：

- 开盘第一笔 OFI/收益/持续时间变化可能包含隔夜信息。
- 一字板判断只取该股票全样本第一个快照，不是每天第一个快照。
- rolling realized moments 会跨日带入上一日窗口。

建议：

- 统一引入 `_time_group(df)` helper，存在 `date` 时总是按 `["SecurityID", "date"]`。
- 一字板应按 `["SecurityID", "date"]` 取首个快照。
- realized moments 的 rolling window 必须按日重置。

### 3. 逐笔成交/委托与 3 秒快照的时间对齐存在未来函数风险

位置：

- `feature_factory.py:1280-1363`
- `feature_factory.py:1599-1678`
- `feature_factory.py:1435-1439`
- `feature_factory.py:1711-1714`

问题：

逐笔数据用 `floor(second / 3) * 3` 聚合到 bucket 起点，然后和同一 `timestamp` 的盘口快照精确 join。这里缺少一个关键定义：快照 timestamp 表示 bucket 起点、bucket 终点，还是交易所推送时刻？

如果快照 `timestamp=09:30:03` 表示 09:30:03 时刻的盘口状态，而成交 09:30:03-09:30:05 被 floor 到 09:30:03，则这些成交发生在快照之后，直接产生未来函数。反过来，如果快照是 bucket 结束状态，用同一 bucket 内成交和结束盘口比较，也会把成交后的盘口拿来解释成交。

影响：

- `TradeVWAPDev`、`TradePriceDev`、`TradePenetration`、`OrderAggress`、`OrderDepthPos` 最敏感。
- 穿透类特征尤其不能用 bucket 级别的单一快照替代每笔成交前的盘口。

建议：

- 明确定义快照语义：bar open、bar close、还是事件时间点。
- 逐笔成交/委托特征若作为 t 时刻可交易特征，应只使用 t 之前已发生的数据。
- 穿透/深度位置应优先用每笔事件发生前最近一笔盘口 `join_asof`，而不是同 bucket 快照。
- 如果最终只在 bucket close 之后下单，应将标签和交易决策时间整体后移，避免同一 bucket 同时用成交和未来盘口。

### 4. 大单阈值使用全天 90% 分位，属于日内未来信息

位置：

- `feature_factory.py:1295-1304`
- `feature_factory.py:1613-1623`
- `FEATURE_SPEC.md:227`
- `FEATURE_SPEC.md:292`

问题：

成交大单阈值和委托大单阈值都使用当日全量数据的 90% 分位。对于日内高频预测，在 09:35 不能知道全天 90% 分位。

影响：

- `LargeTradeRatio`
- `LargeOrderImb`
- `LargeCancelImb`
- `LargeBuyCancelRate`
- `LargeSellCancelRate`

这类特征会在回测中显著高估表现，尤其对放量尾盘、事件日和异常成交日。

建议：

- 使用前一交易日同股票分位数。
- 或使用截至当前时刻的 expanding/rolling 分位数，并至少 lag 一个 bucket。
- 若研究目标是日终解释变量，需要在文档中明确这些特征不可用于实时预测。

### 5. 委托金额口径与实现不一致：代码实际用的是数量

位置：

- `FEATURE_SPEC.md:313-325`
- `feature_factory.py:1557-1596`
- `feature_factory.py:1626-1644`
- `feature_factory.py:1740-1794`

问题：

文档写 `NewBuyAmt`、`NewSellAmt`、`LargeNewBuy`、`LargeCancelBuy`，但代码对沪市 `Balance`、深市 `OrderQty` 直接求和，并命名为 `new_buy_amt` 等。也就是说当前实现是委托数量不平衡，不是委托金额不平衡。

影响：

- 高价股和低价股不可比。
- 与成交侧 `TradeMoney` 口径不一致。
- `OrderImb`、`LargeOrderImb`、`LargeCancelImb` 的经济含义和文档不同。

建议：

- 若目标是金额口径：新增 `OrderMoney = OrderPrice * Balance`，价格为 0 的市价单需要单独处理或用可执行价估算。
- 若目标是数量口径：文档和变量名应改成 `qty`/`volume`，避免研究和生产误读。

### 6. 连续同向成交分桶逻辑错误

位置：

- `feature_factory.py:1243-1277`

问题：

代码先按全日方向变化生成 streak，再把一个完整 streak 归到“第一笔成交所在的 timestamp”。如果一个连续买入序列跨越多个 3 秒桶，后续 bucket 不会得到对应的连续成交长度。

例子：

一个 B streak 从 09:30:01 持续到 09:30:08，代码会把整个 streak 记在 09:30:00 桶，09:30:03 和 09:30:06 桶反而没有该 streak。

影响：

- `ConsecutiveBS` 在强趋势或扫单场景下会错配到较早 bucket。
- 这会同时造成未来信息和当前 bucket 信息缺失。

建议：

- streak 应在 `["SecurityID", "timestamp"]` 内重新计算。
- 或者先按全日计算 streak，再按 bucket 切分每个 streak 的重叠长度。

### 7. OrderDepthBestFrac/DeepFrac 的分母把无效订单也算进去了

位置：

- `feature_factory.py:1647-1675`
- `feature_factory.py:1861-1884`

问题：

`depth_total_buy = buy_depth.len()` 和 `depth_total_sell = sell_depth.len()` 统计的是组内总行数，不是有效买/卖深度记录数。`buy_depth` 对非买单、无价格单、无有效盘口时为 null，但 `len()` 仍计入分母。

影响：

- 买/卖最优价占比和深档占比会被系统性压低。
- 买卖两侧订单数量不对称时，差值也会偏。

建议：

- 用 `buy_depth.is_not_null().sum()` 和 `sell_depth.is_not_null().sum()` 做分母。
- 最好同时输出无效价格/市价单占比，否则深度分布缺了一块最激进流量。

### 8. 缓存键过粗，容易产生静默污染

位置：

- `feature_factory.py:1425-1444`
- `feature_factory.py:1702-1717`
- `feature_factory.py:2140-2243`

问题：

磁盘缓存只按 `factor_name/date`，内存逐笔聚合缓存只按 `date`。这些 key 没有包含：

- universe
- 输入快照版本
- 对齐方式
- `mid_price` 计算版本
- 参数版本
- 代码版本

更严重的是，`_join_trade_stats` 和 `_join_order_stats` 用第一次传入的 `snap_prices` 计算穿透、价格偏离和深度位置；同一进程内后续同日期但不同 df 会复用旧聚合结果。

影响：

- 用小股票池先算一次，再用全市场算同一天，穿透/深度类特征可能沿用小股票池的 snap lookup。
- 修改特征逻辑后旧 parquet 仍会被读入，造成回测结果不可复现。

建议：

- 缓存 key 增加数据版本、universe id、特征参数 hash、代码版本 hash。
- 内存缓存应区分 `snap_prices` 依赖；至少把 price-dependent 聚合拆出来，不要只按 date 缓存。
- 对研究环境，默认 `use_cache=False` 或提供强制 invalidation。

## 高优先级问题

### 9. FeatureFactory 没有检查 required_cols，且很多特征 required_cols 为空但实际依赖盘口列

位置：

- `feature_factory.py:35-51`
- `feature_factory.py:1447-1544`
- `feature_factory.py:1722-1998`
- `feature_factory.py:2200-2243`

问题：

注册器保存了 `required_cols`，但 `compute_single` 从未校验缺失列。许多成交/委托特征声明 `required_cols=[]`，实际却依赖 `BidPrice1`、`AskPrice1`、`mid_price`、`timestamp` 等列。

影响：

- 缺列时可能抛出不清晰的运行时错误。
- 更糟的是某些函数会先聚合并缓存，再在后续步骤失败，留下不完整缓存。

建议：

- 在 `compute_single` 中统一校验 `timestamp`、`SecurityID` 和每个因子的必需列。
- 对依赖逐笔数据但还需要盘口参考价的特征，显式声明 `["BidPrice1", "AskPrice1", "mid_price"]`。
- 对 `HighLimitPrice/LowLimitPrice` 这类可选列，拆分 required 和 optional。

### 10. OFI_Decay 前几行会变成 null，不是可用历史的衰减和

位置：

- `feature_factory.py:259-269`

问题：

`wsum = wsum + ofi.shift(k).over(group) * exp(-k)`。前 4 个 bucket 中，较大 k 的 shift 为 null，Polars 算术会把整条表达式传染为 null。

影响：

- 每只股票开盘前几个 bucket 的 `OFI_Decay` 为 null，而不是使用已有历史做部分衰减和。
- 写缓存时 `drop_nulls(subset=["value"])` 会直接丢掉这些行。

建议：

- 对每个 lag 后 `.fill_null(0.0)`。
- 如需可比尺度，可除以已可用权重和。

### 11. MaxDurPressure 对深市 null 行返回 0，与文档不一致

位置：

- `FEATURE_SPEC.md:55-62`
- `feature_factory.py:575-595`

问题：

文档写“深市股票全为 null”，但代码在 duration 为 null 时，`denom > 0` 不成立，最后 `.otherwise(0.0)`，深市会输出 0。

影响：

- 0 会被模型理解为“买卖耐心资本均衡”，而不是“该市场无此字段”。
- 这会混淆交易所差异和真实信号。

建议：

- 输入为空或任一 side duration 为空时输出 null。
- 或显式增加 `HasMaxDur` 指示变量。

### 12. 成交 VWAP 的跨市场重聚合权重不正确

位置：

- `feature_factory.py:1396-1418`

问题：

`trade_vwap` 合并时用 `trade_count` 加权，而 VWAP 应按成交量或成交金额加权。虽然同一 `SecurityID` 通常只来自一个市场，跨市场重聚合多数情况下不会触发，但代码逻辑本身不正确。

建议：

- 保留 `trade_vwap_num` 和 `trade_vwap_den` 到最终 group，再用 `sum(price*volume)/sum(volume)`。
- 对 `trade_price_dev` 这类均值可以用有效成交笔数加权，但要确认分母对应非空记录数。

### 13. OrderAggress 使用简单均价，不按订单量加权

位置：

- `feature_factory.py:1643-1644`
- `feature_factory.py:1764-1782`

问题：

新增买/卖委托价格用 `.mean()`，一笔 100 股订单和一笔 100 万股订单权重相同。

影响：

- 对大单激进程度不敏感。
- 容易被小额噪声单污染。

建议：

- 使用数量加权平均价：`sum(OrderPrice * Balance) / sum(Balance)`。
- 若最终采用金额口径，和 `OrderMoney` 统一。

### 14. 已实现偏度/峰度在未满 300 点时缩放错误

位置：

- `feature_factory.py:1921-1948`

问题：

rolling window `min_periods=30`，但 `RSkew` 和 `RKurt` 始终用常数 `N=300`。在开盘后 30 到 299 个快照期间，实际样本数不是 300。

影响：

- 开盘前 15 分钟内 skew/kurt 被系统性放大或缩放不一致。

建议：

- 同步计算 rolling count。
- 用有效样本数 `n_eff` 替代固定 N。

### 15. 涨跌停边界条件不完整

位置：

- `feature_factory.py:877-910`
- `feature_factory.py:1042-1067`

问题：

- 创业板不只有 `300xxx`，还包括 `301xxx`。
- 新股上市初期、北交所、ST、特殊处理股票的涨跌幅限制没有处理。
- `NearLimitUp/Down` 只判断距离 `< 0.02`，没有约束距离 `>= 0`；如果价格异常越过涨跌停，也会被判 near。

建议：

- 使用交易所原生 limit price 优先，这一点代码已经做了；但 fallback 规则应覆盖 301、ST、新股和北交所，或明确只支持已过滤股票池。
- near-limit 距离应满足 `0 <= dist < threshold`。

### 16. TradePenetration 使用 bucket 盘口判断每笔成交穿透，经济含义偏弱

位置：

- `feature_factory.py:1328-1343`

问题：

穿透成交应和成交发生前的盘口比较。当前做法是所有成交共用同一 3 秒快照价。如果这张盘口是 bucket 结束状态，成交已经改变了盘口；如果是 bucket 起点，bucket 后半段成交又看不到中间盘口变化。

建议：

- 用逐笔成交时间对盘口快照做 as-of join，取成交前最近盘口。
- 若没有更高频盘口，只能把该特征降级为粗略 proxy，并在文档中说明。

## 中优先级问题

### 17. BookSlope 归一化口径值得重新审视

位置：

- `FEATURE_SPEC.md:135-141`
- `feature_factory.py:508-567`

问题：

斜率单位是 `volume / price`。当前文档和代码都除以 `总深度 * MidPrice`，量纲变成近似 `1 / price^2`。更常见的无量纲处理是除以 `总深度 / MidPrice`，即 `slope * MidPrice / 总深度`。

影响：

- 高价股和低价股尺度可能被额外压缩一个价格平方因子。
- 横截面比较可能偏向低价股。

建议：

- 对比当前版本与 `slope * mid / total_depth` 的 IC、分组收益和行业/价格暴露。
- 若保留当前定义，文档应说明这是人为尺度压缩，而不是自然无量纲。

### 18. realized moments 没有校验 mid_price > 0

位置：

- `feature_factory.py:1907-1918`

问题：

`ln_mid = pl.col("mid_price").log()`。停牌、涨跌停单边无有效价或异常 0 价会导致 null/inf，后续 rolling 结果被污染。

建议：

- 仅在 `mid_price > 0` 时计算收益。
- 无效 mid 的收益应置 null，并确认 rolling 是否忽略 null 或中断窗口。

### 19. SZ tick rule 首笔默认 B 有系统偏差

位置：

- `feature_factory.py:1211-1238`

问题：

深市无原生 B/S 时，用 tick rule 是合理 fallback，但首笔成交默认 `B` 会在大量等价成交链里引入买方偏差。

建议：

- 首笔方向设为 null 或用 quote rule：成交价高于 mid 判 B，低于 mid 判 S。
- Lee-Ready 原始思想通常结合报价中点和滞后报价，不只是 tick test。

### 20. OrderDepthPos 对市价单和极端激进单处理不充分

位置：

- `feature_factory.py:1647-1675`

问题：

文档说 Price=0 市价单从价格敏感特征中剔除。剔除后，最激进的一类订单完全消失；同时负 depth 表示可成交或穿透型限价单，但后续 best/deep 统计用 `abs()`，会混淆“更激进”和“更深档”。

建议：

- 市价单单独输出 `MarketOrderFrac` 或纳入 OrderAggress 的最高激进档。
- 深度分布不要只用 `abs(depth)`，应区分负 depth、0-1 tick、1-5 tick、>5 tick。

### 21. 文档章节小计多处错误

位置：

- `FEATURE_SPEC.md:7`
- `FEATURE_SPEC.md:118`
- `FEATURE_SPEC.md:177`
- `FEATURE_SPEC.md:225`
- `FEATURE_SPEC.md:290`

问题：

文档总数 73 和代码注册特征数 73 一致，但多个章节标题的小计不一致。例如盘口静态结构标题写 12 个，逐项展开只有 10 个；逐笔成交标题写 11 个，逐项展开只有 10 个。

影响：

- 因子治理和上线清单容易混乱。
- 研究人员很难确认是否漏实现。

建议：

- 用代码注册表自动生成特征清单。
- 文档保留“概念分组”，数量由脚本产出，避免手工维护。

### 22. `_LIMIT_COLS` 把可选列写成必需列

位置：

- `feature_factory.py:872-910`

问题：

`HighLimitPrice` 和 `LowLimitPrice` 在函数中是可选列，有 fallback；但 `_LIMIT_COLS` 将它们写入 required_cols。当前 `compute_single` 没有校验所以暂时没暴露。一旦补上 required_cols 校验，会误伤沪市数据。

建议：

- 区分 `required_cols` 和 `optional_cols`。

### 23. PCA 使用 Python map_groups，生产性能可能成为瓶颈

位置：

- `feature_factory.py:2017-2106`

问题：

每个 timestamp 调一次 Python UDF，对全市场全天 3 秒截面会有明显开销。研究环境可接受，生产批处理可能慢。

建议：

- 如果 PCA 特征表现稳定，再考虑批量 numpy reshape 或按 timestamp 分块并行。
- 至少增加耗时日志和样本数统计。

## 低优先级问题和清理项

### 24. `_load_trade_sz` 把 `_mkt` 写成 `"sh"`

位置：

- `feature_factory.py:1199`

问题：

深市 trade loader 写 `pl.lit("sh").alias("_mkt")`。当前 `_mkt` 后续没有使用，所以不影响结果，但这是明显笔误。

建议：

- 改成 `"sz"`，或删除 `_mkt`。

### 25. `_BOOK_CURVE_COLS` 重复定义

位置：

- `feature_factory.py:707-720`

问题：

同一变量重复定义两遍，不影响运行，但降低维护清晰度。

建议：

- 删除重复定义。

### 26. `trade_price_dispersion` 计算了但没有注册使用

位置：

- `feature_factory.py:1317-1319`
- `feature_factory.py:1386-1413`

问题：

该字段计算和跨市场聚合都保留了，但没有对应注册特征。若不使用，应删除；若需要，应写入文档并注册。

## 规格与实现一致性概览

| 模块 | 状态 | 主要问题 |
|---|---|---|
| 静态盘口 | 基本可用 | BookSlope 归一化需复核；Depth_Imbalance 对无效价格缺少过滤 |
| OFI/订单流 | 公式方向基本正确 | 排序、跨日、OFI_Decay null、TS_Imbalance 未按 date 分组 |
| 多档 OBI/AmtOBI | 基本可用 | 盘口无效价/量需要统一过滤策略 |
| 盘口形状 | 部分可用 | BookSlope 量纲、BookConvexity 文档与实现尺度差异 |
| 涨跌停 | 部分可用 | 301/ST/新股/异常距离；深市 null 与 0 混淆 |
| 逐笔成交 | 风险较高 | 时间对齐、全天阈值、连续成交分桶、穿透口径 |
| 逐笔委托 | 风险较高 | 金额/数量口径不一致、深度分母错误、均价未按量加权 |
| realized moments/PCA | 研究可用，生产需修 | 跨日、排序、窗口有效样本数、性能 |
| 缓存/工厂 | 需重构 | required_cols 未校验；缓存 key 过粗 |

## 建议修复顺序

1. 先修时间排序和日期分组：这是所有时序因子的基础。
2. 明确 3 秒快照语义，重做逐笔成交/委托与盘口的对齐规则。
3. 去掉全天分位数未来函数，改为前日或截至当前的 lagged threshold。
4. 统一委托侧“金额 vs 数量”口径，并同步变量名和文档。
5. 修复 ConsecutiveBS、OrderDepthBestFrac/DeepFrac、OFI_Decay。
6. 重做缓存 key 和内存聚合缓存，避免同日不同输入污染。
7. 补 required_cols 校验和最小单元测试。
8. 最后再调 BookSlope、涨跌停 fallback 和 PCA 性能。

## 建议补充的测试

最少需要以下单元测试或小样本金标准：

- OFI：构造价格上移、下移、持平三种场景，逐档验证 e/f。
- 排序：同一数据打乱行顺序后，排序入口应产出一致结果。
- 跨日：两天同一股票拼接，开盘第一条 OFI/RVol/MaxDur 不应使用前一日。
- OFI_Decay：前 4 个 bucket 应给出部分历史结果，不应全 null。
- ConsecutiveBS：构造跨 3 秒边界的同向 streak。
- TradePenetration：验证成交前盘口 vs bucket 盘口两种口径差异。
- OrderImb：金额口径和数量口径分别给出预期值。
- OrderDepthBestFrac：包含买单、卖单、市价单、无效盘口时，分母只统计有效侧记录。
- LargeTrade/Order：同一上午样本不能使用下午数据决定大单阈值。
- Cache：同一日期不同 universe 或不同 snap_prices 不应复用 price-dependent 聚合。

## 上线判断

当前版本适合作为研究原型，不适合作为最终生产因子库。若用于回测，至少应先修复 P0/P1 问题，并在报告中明确以下约束：

- 单日输入。
- 输入已按 `SecurityID, timestamp` 排序。
- 逐笔聚合只用于 bucket close 后可见的特征。
- 全天 90% 分位特征不得用于日内实时预测。
- 委托侧当前是数量口径，不是金额口径。

在这些约束未落实前，任何 IC、分组收益、回测 Sharpe 都应视为可能被未来函数和口径错配污染。
