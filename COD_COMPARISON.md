# COD 单源域适应对照协议

## 代码审计与移植

- 现有六方向五 seed 基线在 `run_five_seed_pairs.py`，源域及目标域各 315 个 CUT，源域均值和标准差按六通道全局统计后同时应用于两域。输入来自 `data_sampling.py` 的固定中心 4096 点 STFT；模型为 `networks/resnet.py` 的 ResNet18 与 `Linear(512,1)`。Adam、batch=63、50 epoch、固定学习率 1e-3、最终 epoch 模型及原有 DARE-GRAM 指数进度权重沿用。旧实验的 C6 指标曾只评估 95–315，新的三方法对照统一重算 1–315。
- 官方 [MPI3D_main.py](https://github.com/MathAI-LAB/COD/blob/main/code/MPI3D/MPI3D_main.py) 的 `mul_guassian_kernel` 与 `COD_Metric` 分别移植为 `models/COD.py` 的同名核函数及 `cod_metric`。它用源/目标 ResNet 特征构造 5 个高斯核，带宽取拼接样本两两平方距离的非对角均值并 detach；`kernel_mul=2`。条件核输入为源域真实回归标签与目标域当前预测值，分别拼接 4 列幅度 `1e-3` 的均匀随机数。目标条件整体 detach，特征核仍向源/目标特征提取器反传。指标为官方的 **CMMD + CKB**，包含 `epsilon=5e-2` 的核矩阵逆、中心化、特征分解和奇异值项；这里没有以单独的条件 MMD 代替 COD。
- 官方训练调用 `COD_Metric(feat_s, feat_t, y_source, y_target.detach(), device)` 位于上述文件训练循环；`run_cod_comparison.py:train` 对应实现。官方代码从目标 loader 读取 `labels_target` 仅用于随机附加列的形状，这里改用目标信号 batch 的长度，避免目标标签进入训练。目标磨损文件第一次读取位于 `run_one` 中 COD 检查点与预测固定之后；其他两个方法的检查点也已冻结，随后才进行重新推断和统一评估。
- 官方 MPI3D 默认 `tradeoff_cod=1e-3`、`epsilon=5e-2`、`warmup_num=3000`、`num_iter=10000`；本实验固定 COD 权重 1e-3、epsilon 5e-2，并按 30% 训练进度把 warmup 定为第 75/250 步开始。该换算在查看任何目标指标之前锁定。DARE-GRAM 使用现有项目权重和进度规则。所有方法使用相同 seed、初始权重、源批次顺序、训练预算及最终 epoch 选择；source-only 与 DARE-GRAM 复用已审计的旧检查点，并在新目录重新计算 1–315 的预测和指标。**不基于目标指标选参；本次没有搜索候选参数或使用源域验证来调参。**
- 每 epoch 保存源域 MSE、DARE-GRAM 原始损失及加权损失、COD 原始损失及加权损失、总损失。COD 为独立脚本运行，不接入原主训练入口。

## 运行

在 `upstream-reproduction` 目录执行：

```powershell
python run_cod_comparison.py smoke --device cuda
python run_cod_comparison.py run --device cuda
python run_cod_comparison.py summarize
python audit_cod_comparison.py
```

冒烟方向固定为 C1→C4，完全不读取预测指标；它检查 COD 前反向传播、特征梯度非零、`lambda_COD=0` 与 DARE-GRAM 的模型权重和损失一致、目标标签读取被禁用后的训练权重不变。正式训练仅在 `smoke.json` 全部通过后运行。产物写入新目录 `artifacts/cod_comparison_20260926`；完成后有 90 行 per-seed 指标、18 行六方向×三方法均值总表、六方向 COD 减 DARE-GRAM 的变化表、每方向/seed/方法 315 行真实值与预测值 CSV，以及统一坐标的 PNG 曲线。
