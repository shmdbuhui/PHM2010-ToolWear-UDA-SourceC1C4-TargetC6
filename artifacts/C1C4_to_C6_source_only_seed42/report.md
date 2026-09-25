# C1+C4→C6 ResNet source-only 实验报告

**实验身份：开源仓库 STFT＋ResNet 的 C1+C4→C6 端到端 source-only baseline。** 输入是六通道 STFT 图像，与现有手工特征模型的输入表示不同；可比较固定 C6 suffix 同一测试任务上的最终性能，不能把分数差单独归因于回归网络。

## 协议与来源

- 独立入口：`run_c1c4_to_c6_source_only.py`；未改动既有单源入口或其输出。
- 固定划分来源：`E:\QLP\test-9.3\outputs\predictions_comparison.csv` 中 `tool=C6, split=target_test_suffix_70pct` 的 `cut_index` 列；准确清单为连续 cut **95–315**，共 **221** 条。入口只读取该记录的 `tool,cut_index,split` 三列来决定划分，不重新生成测试集。该主实验代码 `PHM2010_neural_PINN_main.py` 使用 `floor(0.30*315)=94` 个前缀 cut，并记录固定 suffix。完整清单保存在 `config.json`。
- 源域：C1 与 C4 各 315 个允许 cut，合计 630 个；训练不使用 C6 数据、C6 标签、C6 归一化统计或目标域 BN 更新。第 50 轮权重保存后，程序才加载 C6 suffix 信号、生成预测，再读取 C6 磨损标签评估。没有 C6 前缀微调或任何域对齐。
- 标签统一为 `mean(flute_1, flute_2, flute_3)`，与当前固定 suffix 主实验相同。原仓库预处理也是三刃均值。另一个 `E:\QLP\multidomain_test` 脚本使用三刃最大值，属于不同标签任务，其指标不与本次结果并列解释。用主实验所引用的 `EEMD-7.22/results/xgboost_mainline/target_{C1,C4,C6}/predictions_ensemble.csv` 核对：三个域均为 315 条、cut 顺序完全一致，原始三刃均值与其 `y_true` 的最大绝对差分别为 0、1.42e-14、3.55e-15。
- 操作记录说明：在训练进程运行期间，我另用只读命令核对了原始 C6 wear CSV 与主实验标签的一致性。这些标签没有传入训练进程，也未影响权重、统计量、参数或模型选择；但严格按“训练结束后才读取任何 C6 标签”的字面要求，这次人工核对发生得过早。训练入口自身直到保存第 50 轮权重后才读取 C6 标签。
- 与原 source-only 相同：`data_sampling.py` 中心 4096 点、六通道 STFT 256/224、log1p 幅值、128×128；`ResNet18`、512→1 线性头、MSE、Adam、固定学习率 1e-3、batch 64、50 epoch、seed 42。训练 DataLoader 使用 `drop_last=True`，每 epoch 9 个完整 batch；最终权重评估，无 C6 模型选择。

## 数据核验

| 项目 | 结果 |
|---|---|
| C1 源域 | 315 个信号文件与 315 条磨损记录，cut 1–315 各一次；标签范围 39.643520–165.172409 |
| C4 源域 | 315 个信号文件与 315 条磨损记录，cut 1–315 各一次；标签范围 24.216036–203.077911 |
| 合并训练集 | 630 个样本，张量 `(630,6,128,128)`；标签范围 24.216036–203.077911 |
| C6 固定测试集 | cut 95–315 共 221 个，预测输入张量 `(221,6,128,128)`；真实标签范围 110.948783–215.942238 |
| NPZ cut 映射 | 原 NPZ 只有 `samples,labels`。已将 C1/C4 每一行重新从对应真实文件名及 cut 重算 STFT，与 NPZ 图像逐元素完全相等，并核对该行标签与 wear CSV 的 float32 均值完全相等。映射保存在 `c1_npz_cut_index_mapping.csv` 和 `c4_npz_cut_index_mapping.csv`。NPZ 行号未用作 cut 编号。 |
| 归一化 | 仅合并 C1+C4 的 630 个训练图像计算每通道均值与标准差，并固定用于 C6；数值保存在 `config.json`。 |

## 最终 C6 suffix 指标

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| 23.021984 | 25.397115 | 0.434532 | 15.477880% |

`c6_suffix_predictions.csv` 恰好包含 221 行、按真实 `cut_index=95..315` 排序，每个 cut 一条预测，列为 `cut_index,true_vb,pred_vb`。独立读取 CSV 并按误差公式重算的四项指标与 `run.log` 完全一致；真实标签逐项与主实验固定 suffix 记录一致（容差 1e-6）。

## 运行命令

在 `E:\QLP\PHM2010-ToolWear-UDA-Reproduction\upstream-reproduction` 的 PowerShell 中：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='6'
.\.venv\Scripts\python.exe -u run_c1c4_to_c6_source_only.py --raw-root E:\QLP\source\source_mill --npz-dir .\dataset --split-record E:\QLP\test-9.3\outputs\predictions_comparison.csv --out-dir .\artifacts\C1C4_to_C6_source_only_seed42 --device cpu
```

输出目录须是尚不存在的新目录，以防意外覆盖。`code.diff` 是新增独立入口相对空文件的完整 diff。结果文件：`run.log`、`config.json`、`epoch50.pth`、`metrics.json`、`c6_suffix_predictions.csv`、`c6_suffix_prediction.png`、两个源域 NPZ 行映射和本报告。

原 C1→C4 的 source-only 日志、权重、预测 CSV、图片在本次运行后 SHA256 与本次留存值一致，见 `preserved_c1_to_c4_sha256.json`。
