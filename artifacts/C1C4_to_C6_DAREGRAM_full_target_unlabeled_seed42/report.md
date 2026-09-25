# C1+C4→C6 全目标域无标签 DARE-GRAM 实验

**协议：转导式 UDA。** C6 全部 315 个 cut 的输入信号参与过无标签对齐，其中包括最终评估的 suffix cut 95–315。C6 磨损标签不参与训练、对齐、归一化统计、模型选择或调参；它们仅在第 50 轮权重保存后用于评估。因此本组不是“未见目标样本”的严格独立测试。模型名称为 **ResNet source-only + DARE-GRAM(full-target-unlabeled)**，仍是 STFT＋ResNet18＋线性 512→1 回归头，不是原仓库未经修改的 DAREGRAM 模型。

## 数据使用清单

| 组别 | 训练有标签源域 | 训练可见的 C6 输入 | C6 训练标签 | 最终评估 |
|---|---|---|---|---|
| source-only | C1+C4 cut 1–315，各 315 条 | 无 | 无 | C6 suffix cut 95–315，共 221 条 |
| prefix-unlabeled | 相同 630 条 | cut 1–94，无标签 | 无 | 相同 221 条，评估输入未参与训练 |
| **full-target-unlabeled，转导式** | 相同 630 条 | **cut 1–315，无标签** | **无** | 相同 221 条，评估输入参与过无标签对齐 |

主实验 `E:\QLP\test-9.3\outputs\diagnosis\c6_full_lifecycle_diagnosis.csv` 的元数据列给出真实 calibration cut 1–94 和 test cut 95–315；后者与 `predictions_comparison.csv` 的固定 suffix 清单逐项一致。两份划分记录在训练前只按 `tool,cut_index,split` 列读取。目标信号逐 cut 从真实文件名 `c_6_001.csv` 至 `c_6_315.csv` 读取并用 `data_sampling.py` 变换；没有目标 NPZ，因此也没有从 NPZ 携带标签。目标 Dataset 只保存图像与真实 cut 编号，`__getitem__` 只返回 `(image, cut_index)`，代码在训练前断言它没有 `labels` 字段，训练循环也只解包这两个字段。C6 磨损 CSV 在训练和权重保存前未被程序打开。

源域 NPZ 的 630 行再次按原始信号文件和磨损标签逐行重算核验。C1+C4 合并图像的通道均值和标准差与 source-only 配置逐元素相等，固定用于两个域；不使用 C6 拟合归一化参数。输入为 `(630,6,128,128)` 源域图像和 `(315,6,128,128)` 无标签目标图像。标签均为 `mean(flute_1,flute_2,flute_3)`，目标标签仅在最终评估时读取。

## 训练与一致性核验

- 从 seed 42 重新初始化，初始模型 SHA256 为 `791e81a865208ac373d2a6f5c599e792686145eddafbc4ef215d7b4890f983ae`，与前缀组保存的初始哈希完全相同。
- 源域和目标域 batch 均为 64，`drop_last=True`；每轮 9 个源域更新步，共 450 步。源域 50 轮的顺序哈希逐轮与前缀组日志匹配。第 1 轮源域 MSE 为 10486.574436，与前两组一致。
- ResNet18、线性回归头、源域 MSE、Adam 两组学习率 1e-3、50 epoch、原 DARE-GRAM 特征损失和逐轮 `exp` tradeoff 均沿用前缀组；唯一计划中的适应数据变化为 C6 cut 1–94 扩展到 1–315。每轮原始 Gram loss、实际加权对齐项、source MSE 和目标使用数见 `run.log`。
- 目标域 DataLoader 每个洗牌周期有 4 个完整的 64-cut batch（256 条），余下 59 条因 `drop_last=True` 在该周期丢弃；下个周期重新洗牌。每轮循环取 9 个目标 batch，即 576 次抽取，覆盖 297–310 个不同 cut，重复抽取 266–279 次，单轮未抽中 5–18 个。50 轮合计 28,800 次抽取；`target_full_usage.csv` 证明全程使用集合恰为 cut 1–315，无漏项或额外 cut，且 221 个 suffix cut 全部至少参与一次无标签对齐。

## 固定 suffix 结果

三组预测 CSV 的 221 个真实 cut、顺序和标签逐项一致。新组预测 CSV 独立读取后重算的四项指标与日志一致。

| 组别 | 训练可见 C6 输入 | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| source-only | 无 | 23.021984 | 25.397115 | 0.434532 | 15.477880% |
| DARE-GRAM(prefix-unlabeled) | 1–94 | 36.249985 | 38.670888 | -0.311016 | 26.538013% |
| **DARE-GRAM(full-target-unlabeled)，转导式 UDA** | **1–315，含测试输入** | **16.335198** | **19.846246** | **0.654700** | **10.700444%** |

新组相对 source-only：MAE -6.686787、RMSE -5.550870、R² +0.220169、MAPE -4.777436 个百分点。新组相对前缀组：MAE -19.914787、RMSE -18.824643、R² +0.965716、MAPE -15.837569 个百分点。**这些差异是在相同 suffix 标签上、不同目标输入可见范围下得到的，不能把本转导式结果称为严格未见目标样本的性能。**

## 运行命令与文件

在 `E:\QLP\PHM2010-ToolWear-UDA-Reproduction\upstream-reproduction` 目录执行：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='6'
.\.venv\Scripts\python.exe -u run_c1c4_to_c6_daregram_full_target.py --raw-root E:\QLP\source\source_mill --npz-dir .\dataset --split-record E:\QLP\test-9.3\outputs\predictions_comparison.csv --full-split-record E:\QLP\test-9.3\outputs\diagnosis\c6_full_lifecycle_diagnosis.csv --source-only-dir .\artifacts\C1C4_to_C6_source_only_seed42 --prefix-dir .\artifacts\C1C4_to_C6_DAREGRAM_prefix_unlabeled_seed42 --out-dir .\artifacts\C1C4_to_C6_DAREGRAM_full_target_unlabeled_seed42 --device cpu
```

本目录包含 `run.log`、`config.json`、`epoch50.pth`、`metrics.json`、`comparison.json`、`c6_suffix_predictions.csv`、`c6_suffix_prediction.png`、`target_full_usage.csv`、两个源域 NPZ 行映射和本报告。`code.diff` 是新入口相对前缀组入口的代码差异。两组已有结果的配置、权重、预测 CSV、图片、指标和日志 SHA256 在本次运行后均未变化，核验记录在 `preserved_previous_sha256.json`。

首次预检因目标 batch 局部变量名覆盖完整 cut 清单而在**第一次参数更新前**触发白名单断言；失败日志独立留存在 `C1C4_to_C6_DAREGRAM_full_target_unlabeled_seed42_preflight_failed`。只修正该局部变量名后，使用相同预定参数完成了本次唯一的 50-epoch 正式训练；没有依据 suffix 指标修改损失权重或选择 epoch。
