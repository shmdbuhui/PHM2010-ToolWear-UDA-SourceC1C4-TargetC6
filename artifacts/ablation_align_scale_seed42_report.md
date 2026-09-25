# DARE-GRAM 对齐损失公平消融：C1→C4，seed=42

## 结论

在相同 DAREGRAM 网络、优化器、两组学习率、预处理、归一化、batch 顺序和 50 epoch 预算下，`align_scale=1` **没有改善**本次 C1→C4 结果。相对于 `align_scale=0`，R² 下降 0.018136，MAE 增加 0.565845，RMSE 增加 0.735056，MAPE 增加 0.780573 个百分点。结论仅针对这个 seed 和原仓库的转导式协议。

## 代码和环境

- 基础仓库上游 commit：`5b18a3119a68be83a5b9c8294d80617288447a8d`。本次开始前的复现代码快照位于 `artifacts/ablation_baseline/`，本次单独的改动见 `artifacts/ablation_align_scale_diff.patch`。
- 新增 float 参数 `--align_scale`，默认 1.0；总损失为 `source_mse + align_scale * tradeoff[0] * dare_gram_loss`。两组都计算源域和目标域特征，也都计算未加权 DARE-GRAM loss。目标标签在训练批次读取后立即丢弃，回归 MSE 只用 C1 标签。
- 固定 `torch.Generator`：源域种子 42，目标域种子 43。每轮源域和目标域输入批次的完整字节 SHA256、初始网络状态 SHA256、BatchNorm 跟踪批次数都写入 epoch 日志。添加这些记录未改变网络、优化器、学习率、样本、预处理、归一化或评估协议。
- 使用原有 NPZ：`dataset/train_c1.npz`、`dataset/unlabeled_c4.npz`、`dataset/target_c4.npz`。文件 SHA256 见 `artifacts/ablation_npz_sha256.txt`。本机 Python 3.13.0、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128；两组均使用 CPU。
- 原始 DAREGRAM 的权重与曲线仍在原目录，未被覆盖。本次显式固定独立 DataLoader 生成器，因此 `align_scale=1` 的数值不要求等于先前全局 RNG 路径下的原始复现数值；两组消融在本次运行内具有相同批次顺序。

## 完整命令

在 `E:\QLP\PHM2010-ToolWear-UDA-Reproduction\upstream-reproduction` 下用 PowerShell 执行：

```powershell
$env:MPLBACKEND='Agg'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u train.py --model_name DAREGRAM --source c1 --target c4 --data_dir .\dataset --batch_size 128 --max_epoch 50 --lr 1e-3 --random_state 42 --cuda_device cpu --align_scale 0 --save --save_dir .\artifacts\checkpoints
.\.venv\Scripts\python.exe -u train.py --model_name DAREGRAM --source c1 --target c4 --data_dir .\dataset --batch_size 128 --max_epoch 50 --lr 1e-3 --random_state 42 --cuda_device cpu --align_scale 1 --save --save_dir .\artifacts\checkpoints
```

A 从 2026-09-24 21:21:07 至 21:29:25（北京时间，约 498 秒），B 从 21:22:07 至 21:30:22（约 495 秒）；两者均正常退出。命令、精确时间和退出码另存于 `artifacts/logs/ablation_align_scale{0,1}_seed42_run.json`。

## C4 指标

仅在训练完成后对 `target_c4.npz` 全部 315 个样本评估，没有按 C4 指标选 epoch 或调系数。下表由各自保存的 CSV 独立复算。

| 组别 | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| A：`align_scale=0` | 14.404730 | 17.377418 | 0.790066 | 15.142805% |
| B：`align_scale=1` | 14.970575 | 18.112474 | 0.771931 | 15.923377% |
| B−A | +0.565845 | +0.735056 | −0.018136 | +0.780573 个百分点 |

NPZ 没有保存真实 cut 编号，预测 CSV 第一列明确命名 `sample_index`，依 NPZ 顺序为 1–315，另两列为 `y_true_vb` 和 `y_pred_vb`。

## 训练过程核验

- 初始模型 SHA256 两组均为 `5e2d5d5ef56536f42ad349c7f0bbc486d866f84800e168f9b739e348c24fa84d`。
- 50 个 epoch 中，每轮源域和目标域批次 SHA256 在两组间逐一相同。第 1 轮 source MSE、未加权 Gram loss 也相同。
- 两组的首层 Backbone BatchNorm 跟踪数每轮增加 4，最终均为 200，证明两组都保留目标域前向与 BatchNorm 更新路径。
- 第 2 轮第 1 个 batch，对齐项对源域与目标域特征的梯度范数：A 为 `0, 0`；B 为 `0.506913245, 0.509140134`。A 全部 50 轮的实际加权项为 0；B 从第 2 轮起非零。逐轮核对了 `weighted_alignment = align_scale * tradeoff * unweighted_gram_loss` 及 `total_loss = source_mse + weighted_alignment`（日志舍入容差内）。
- 完整逐轮数值、全部批次哈希、指标及断言结果保存在 `artifacts/ablation_audit_seed42.json`；复核脚本为 `artifacts/audit_ablation.py`。两份权重均可用 `torch.load(..., weights_only=True)` 正常读取，两张曲线已目视检查。

## 文件路径

| 文件 | A：`align_scale=0` | B：`align_scale=1` |
|---|---|---|
| 逐 epoch 日志 | `artifacts/checkpoints/DAREGRAM_bs128_ep50_lr1e-3_align_scale0_seed42/[c1]To[c4]_[42]_align_scale0_seed42.log` | `artifacts/checkpoints/DAREGRAM_bs128_ep50_lr1e-3_align_scale1_seed42/[c1]To[c4]_[42]_align_scale1_seed42.log` |
| 权重 | 同目录、同名 `.pth` | 同目录、同名 `.pth` |
| 预测 CSV | `visualization/DAREGRAM_align_scale0_seed42/c1_tgt-c4_align_scale0_seed42.csv` | `visualization/DAREGRAM_align_scale1_seed42/c1_tgt-c4_align_scale1_seed42.csv` |
| 预测曲线 | 同目录、同名 `.png` | 同目录、同名 `.png` |
| 控制台日志 | `artifacts/logs/ablation_align_scale0_seed42_console.log` | `artifacts/logs/ablation_align_scale1_seed42_console.log` |

## 协议限制

`unlabeled_c4.npz` 和 `target_c4.npz` 的输入完全相同，且文件 SHA256 一致。B 在无标签对齐中见到的 C4 输入与最终评估输入相同，因此是**转导式 UDA 评估**，不是未见目标样本的独立测试。本次未使用前 30%／后 70% 划分，也不将这组数值直接与 C1+C4→C6 的严格测试指标比较。
