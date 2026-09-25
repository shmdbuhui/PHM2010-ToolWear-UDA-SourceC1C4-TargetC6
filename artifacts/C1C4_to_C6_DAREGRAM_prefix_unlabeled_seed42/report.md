# C6 前缀无标签 DARE-GRAM 对照

**名称：ResNet source-only + DARE-GRAM(prefix-unlabeled)。** 这是以现有 STFT＋ResNet18＋512→1 线性回归头为基础的对照，**不是**原仓库未经修改的 DAREGRAM 模型。

## 固定数据与训练协议

- 主实验 `E:\QLP\test-9.3\outputs\diagnosis\c6_full_lifecycle_diagnosis.csv` 的 `tool,cut_index,split` 列记录 C6 `calibration` 为真实 cut **1–94**、`test` 为 **95–315**。suffix 与 `E:\QLP\test-9.3\outputs\predictions_comparison.csv` 中 `target_test_suffix_70pct` 的 221 个 cut 逐项一致。两份划分文件只读取元数据列，不读取其中的标签列。完整清单和文件 SHA256 见 `config.json`。
- 源域仍为相同 C1+C4 各 315 个 cut，NPZ SHA256 与 source-only 记录一致。630 行均重新以原始信号和标签逐项核验；源域归一化六通道均值与标准差也与旧组逐元素相等。
- C6 前缀只从真实 cut 文件名 `c_6_001.csv` 至 `c_6_094.csv` 读取信号并作确定性 STFT。目标训练 Dataset 只有 `(stft_image, cut_index)`，没有标签字段。训练时每个目标 batch 都断言为 64 个合法前缀 cut，且与测试 cut 集合交集为空。
- 从 seed 42 重新初始化权重，源域 batch 64、目标域 batch 64，均 `drop_last=True`；目标域 94 条可产生 1 个完整 batch，耗尽后重新洗牌抽取。每轮 9 个源域 batch，对应 576 次目标 cut 抽取。50 轮共 28,800 次抽取，覆盖全部 94 个前缀 cut，测试交集为 0。逐轮使用记录在 `target_prefix_usage.csv`；每轮独立 cut 数为 93 或 94。
- 输入窗口、STFT、ResNet18、线性头、源域 MSE、Adam 两组参数学习率均 1e-3、固定学习率、50 epoch 不变。直接调用仓库 `models/DAREGRAM.py` 的 `DARE_GRAM_LOSS`，在两域 512 维特征上计算；总损失为 `source_mse + tradeoff(epoch) × dare_gram_loss`，其中逐轮 `tradeoff=2/(1+exp(-10*(epoch-1)/49))-1`，与仓库 `InitTrain._get_tradeoff` 的 `exp` 分支一致，比例系数固定为 1。
- 每轮日志记录源域 MSE、原始 DARE-GRAM 损失、实际加权对齐项、tradeoff、目标抽取总数与独立 cut 数，以及源域顺序哈希。训练期间不读取 C6 suffix 信号或标签；无 suffix 统计量拟合、选轮或调参。第 50 轮权重保存后只对 suffix 评估一次。

## 初始状态与批次核对

新组初始模型 SHA256 为 `791e81a865208ac373d2a6f5c599e792686145eddafbc4ef215d7b4890f983ae`，与按旧 source-only 入口、同 seed 42 重建的初始权重哈希相同。源域各轮 batch 顺序逐项通过按旧入口 DataLoader 设置重建的顺序断言；第 1 轮源域 MSE 为 **10486.574436**，与旧组日志完全相同。

**证据范围：**旧组当时未保存初始权重哈希或逐 batch cut 顺序，因此无法直接从历史运行文件验证这两项；上述一致性是对旧代码与随机数设置的重建核对。目标 batch 的额外前向会更新新组的 BatchNorm 状态，且训练目标不同，因此第 2 轮起的模型轨迹不应相同。

## 固定 221-cut suffix 结果

| 指标 | Source-only | + DARE-GRAM(prefix-unlabeled) | 新组－旧组 |
|---|---:|---:|---:|
| MAE | 23.021984 | 36.249985 | +13.228000 |
| RMSE | 25.397115 | 38.670888 | +13.273773 |
| R² | 0.434532 | -0.311016 | -0.745547 |
| MAPE | 15.477880% | 26.538013% | +11.060133 百分点 |

新组 `c6_suffix_predictions.csv` 共 221 行，`cut_index=95..315` 各一行，真实标签与旧组 CSV 逐项相同。单独读取新组 CSV 按公式重算，四项指标与日志一致。完整精度数值和差值在 `comparison.json`，最终模型在 `epoch50.pth`，预测图为 `c6_suffix_prediction.png`。

**结论：在固定 C6 suffix 不参与训练的条件下，这次使用 C6 前缀无标签信号做 DARE-GRAM 适应没有优于当前 source-only。**

## 运行命令与交付

在 `E:\QLP\PHM2010-ToolWear-UDA-Reproduction\upstream-reproduction` 目录执行：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='6'
.\.venv\Scripts\python.exe -u run_c1c4_to_c6_daregram_prefix.py --raw-root E:\QLP\source\source_mill --npz-dir .\dataset --split-record E:\QLP\test-9.3\outputs\predictions_comparison.csv --full-split-record E:\QLP\test-9.3\outputs\diagnosis\c6_full_lifecycle_diagnosis.csv --source-only-dir .\artifacts\C1C4_to_C6_source_only_seed42 --out-dir .\artifacts\C1C4_to_C6_DAREGRAM_prefix_unlabeled_seed42 --device cpu
```

`code.diff` 是新入口相对现有 source-only 入口的最小代码差异。新组的 `run.log`、`config.json`、`target_prefix_usage.csv`、权重、预测 CSV、图片、指标、比较 JSON 和源域 NPZ 行映射均在本目录。旧 source-only 的配置、权重、预测 CSV、图、指标及日志在本次运行后 SHA256 均未变化，见 `preserved_source_only_sha256.json`。
