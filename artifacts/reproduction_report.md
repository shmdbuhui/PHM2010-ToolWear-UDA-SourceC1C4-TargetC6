# PHM2010 C1→C4 原仓库复现记录

## 状态与代码

- 已用真实 PHM2010 数据完成 README 中 source-only `resnet` 和 `DAREGRAM` 各 50 epoch，均保存日志、权重、预测 CSV 和曲线。
- 独立克隆目录：`E:\QLP\PHM2010-ToolWear-UDA-Reproduction\upstream-reproduction`；上游 commit：`5b18a3119a68be83a5b9c8294d80617288447a8d`。原有磨损预测项目未修改。
- 原仓库地址：<https://github.com/yongmini/Unsupervised-Domain-Adaptation-CNC-Tool-Wear-Monitoring>。
- 本地最小修复见 `artifacts/code.diff`：支持本机标签与信号文件的扁平目录；让训练加载真正使用 `--data_dir`；允许 `--cuda_device cpu` 并在没有 CUDA 时回退 CPU；将 `--save` 改为命令行开关；校正 README 文件名；让预测图及 CSV 以真实 cut 编号 1–315 为横轴。网络、损失、划分未改。
- `git diff --check`、`py_compile` 均通过。

## 环境

- Python 3.13.0；PyTorch 2.11.0+cu128；torchvision 0.26.0+cu128；RTX 3060 Ti。运行时 GPU 仅剩约 247 MiB 显存，故两个模型都在 CPU 上训练。
- 独立环境：`.venv`（以 `python -m venv --system-site-packages .venv` 创建，使用本机预装依赖）。完整环境快照 `artifacts/environment-freeze.txt`，核心依赖 `artifacts/requirements-reproduction.txt`，版本与设备信息 `artifacts/environment-summary.txt`。
- 绘图时设置 `MPLBACKEND=Agg`，避免本机 Python 的 Tcl/Tk 缺失；设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8` 以保证日志正常编码。

## 数据核验

原始目录：`E:\QLP\source\source_mill`。标签为 `c1_wear.csv` / `c4_wear.csv`，信号在 `c1` / `c4` 子目录，文件名为 `c_1_001.csv` / `c_4_001.csv` 等。

| 域 | 信号文件 | cut 编号 | 缺失/重复/未匹配 | 标签数 | 原始信号列 | 三刃平均磨损范围 | 预处理 NPZ |
|---|---:|---|---|---:|---:|---:|---|
| C1 | 315 | 1–315 | 0/0/0 | 315 | 7 | 39.6435–165.1724 | `(315,6,128,128)`；标签 `(315,)` |
| C4 | 315 | 1–315 | 0/0/0 | 315 | 7 | 24.2160–203.0779 | `(315,6,128,128)`；标签 `(315,)` |

预处理完整读取了 630 个 CSV，全部形成有效 cut；每个 NPZ 的输入和标签均无 NaN。训练使用六个输入通道（Fx/Fy/Fz/Vx/Vy/Vz），第七列 AE RMS 不进入模型。标签逐 cut 与三个刃磨损的算术平均值一致，未使用 VBmax。C1/C4 分别生成 `train_*.npz`、`target_*.npz`、`unlabeled_*.npz`，总计六个文件；每个域的三个文件内容相同。预处理耗时 102.96 秒。

## 精确命令

在克隆目录下（PowerShell）：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe data_sampling.py --root E:\QLP\source\source_mill --conds c1 c4 --out .\dataset
$env:MPLBACKEND='Agg'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u train.py --model_name resnet --source c1 --target c4 --data_dir .\dataset --batch_size 64 --max_epoch 50 --lr 1e-3 --random_state 42 --cuda_device cpu --save --save_dir .\artifacts\checkpoints
.\.venv\Scripts\python.exe -u train.py --model_name DAREGRAM --source c1 --target c4 --data_dir .\dataset --batch_size 128 --max_epoch 50 --lr 1e-3 --random_state 42 --cuda_device cpu --save --save_dir .\artifacts\checkpoints
```

source-only 首次完整训练后，绘图遇到 Tcl/Tk 缺失，进程在保存权重前退出；设置 `MPLBACKEND=Agg` 后按相同训练参数完整重跑。成功的 source-only 运行时间为 2026-09-24 20:48:14–20:51:49（215.29 秒），DAREGRAM 为 20:52:08–20:59:02（414.20 秒），均为北京时间。

## 指标

对 `target_c4.npz` 中全部 315 个 cut 评估，按 cut 1–315 顺序。以下数值由预测 CSV 再算，与训练日志的四舍五入数值一致。

| 模型 | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| resnet source-only | 12.011804 | 16.193158 | 0.817705 | 11.683562% |
| DAREGRAM | 14.194076 | 17.505486 | 0.786961 | 14.790320% |

## 产物

| 内容 | source-only | DAREGRAM |
|---|---|---|
| 训练日志 | `artifacts/checkpoints/resnet_bs64_ep50_lr1e-3/[c1]To[c4]_[42].log` | `artifacts/checkpoints/DAREGRAM_bs128_ep50_lr1e-3/[c1]To[c4]_[42].log` |
| 权重 | 同目录 `[c1]To[c4]_[42].pth` | 同目录 `[c1]To[c4]_[42].pth` |
| 曲线 | `visualization/resnet/c1_tgt-c4.png` | `visualization/DAREGRAM/c1_tgt-c4.png` |
| 预测 | `visualization/resnet/c1_tgt-c4.csv` | `visualization/DAREGRAM/c1_tgt-c4.csv` |

预处理完整日志：`artifacts/logs/preprocess.log`。两个训练的控制台记录：`artifacts/logs/source-only-rerun.log`、`artifacts/logs/daregram-console.log`。可读的逐 epoch 日志以上表中的模型日志为准。

## 评估与公平性审计

- `target_c4.npz` 与 `unlabeled_c4.npz` 的输入完全相同。DAREGRAM 以该批 C4 输入做无标签 Gram 对齐，又用同一批输入的标签计算最终指标，因此是**转导式 UDA 评估**，不是未见目标样本的独立测试。
- DAREGRAM 的目标标签虽被 NPZ 和 DataLoader 读入，训练中变量 `target_labels` 未用于预测或损失；回归 MSE 仅使用 C1 源域标签。归一化统计仅由 C1 训练数据计算。source-only 训练只使用 C1。
- 原仓库 source-only 回归头为 512→1，特征提取器与回归头学习率均为 1e-3；DAREGRAM 回归头为 512→256→1 且带 dropout，特征提取器学习率为 5e-4、回归头为 1e-3。batch 和每 epoch 更新次数也不同。这些原仓库结果不能单凭分数差用来识别域适应本身的因果效果。本次未增加同结构对照，以保持原始复现结果独立。

## 尚未解决的问题

- 原仓库预处理的 `fmax` 参数对应频率裁剪代码被注释，故传入的 `--fmax` 实际不改变频带；本次保留原行为。
- 该评估未提供独立的未见目标域 cut；如需归纳式评估，须另立实验协议，不能把本次转导式结果改名为独立测试。
