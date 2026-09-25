"""Build paired wide tables and report from completed read-only AdaBN outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/adabn_zscore_20260925")
RECON = Path("artifacts/c1c6_metric_reconciliation_20260925")
METHODS = ("source_only", "source_only_adabn", "daregram")
METRICS = ("R2", "MAE", "RMSE", "MAPE_percent")
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))


def label(source: str, target: str) -> str:
    return f"{source.upper()}→{target.upper()}"


def main() -> None:
    wide_path, report_path = ROOT / "paired_seed_comparison.csv", ROOT / "README.md"
    if wide_path.exists() or report_path.exists():
        raise FileExistsError("Summary output already exists")
    with (ROOT / "per_seed_metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = {(r["source"], r["target"], int(r["seed"]), r["method"]): r for r in rows}
    if len(rows) != len(index) or len(rows) != 90:
        raise ValueError("Expected 90 unique per-seed/method rows")
    wide = []
    for source, target in PAIRS:
        for seed in range(42, 47):
            members = [index[source, target, seed, method] for method in METHODS]
            if len({int(r["evaluation_count"]) for r in members}) != 1:
                raise ValueError("Mismatched evaluation count")
            record = {"source": source, "target": target, "seed": seed,
                      "evaluation_count": int(members[0]["evaluation_count"])}
            for method, member in zip(METHODS, members):
                for metric in METRICS:
                    record[f"{method}_{metric}"] = float(member[metric])
            record["delta_rmse_adabn_minus_source_only"] = (
                record["source_only_adabn_RMSE"] - record["source_only_RMSE"])
            wide.append(record)
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wide[0]))
        writer.writeheader()
        writer.writerows(wide)
    with (ROOT / "direction_summary.csv").open(newline="", encoding="utf-8") as f:
        summary = {(r["source"], r["target"], r["method"]): r for r in csv.DictReader(f)}
    recon = json.loads((RECON / "summary.json").read_text(encoding="utf-8"))
    lines = ["# PHM2010 Z-score 主实验：C1→C6 指标对账与 source-only + AdaBN", "",
             "## 第一项：C1→C6 指标差异", "",
             "逐 seed 核验表见 `../c1c6_metric_reconciliation_20260925/per_seed.csv`：同一 seed、同一方法的两份 checkpoint SHA-256 完全一致，源域输入归一化参数也相同。两份预测文件的目标评估 cut 不同：特征诊断读取 `five_seed_paired` 的 **cut 95–315（221 个）**，另一份 Z-score 汇总读取 `norm_comparison_20260925/zscore` 对相同 checkpoint 的 **cut 1–315（315 个）**。重算逐 seed 预测文件得到：", "",
             "| C1→C6 评估 cut | source-only RMSE | DARE-GRAM RMSE | 配对 ΔRMSE |",
             "|---|---:|---:|---:|",
             f"| 95–315 | {recon['suffix']['source_only_rmse_mean']:.4f} | {recon['suffix']['daregram_rmse_mean']:.4f} | {recon['suffix']['delta_rmse_mean']:+.4f} |",
             f"| 1–315 | {recon['full']['source_only_rmse_mean']:.4f} | {recon['full']['daregram_rmse_mean']:.4f} | {recon['full']['delta_rmse_mean']:+.4f} |",
             "", "每行先对五个 seed 各自计算 RMSE，再取算术均值；Δ 为同 seed 的 DARE-GRAM RMSE 减 source-only RMSE，随后对五个 Δ 取均值。标准差均为样本标准差（ddof=1）。当前磁盘汇总文件的精确全生命周期均值为 23.904592 与 20.072844，写成四位小数是 **23.9046 / 20.0728**；用户所引 23.9043 / 20.0726 并非当前文件中的数值，其来源无法从当前文件确认。按未舍入的逐 seed 值计算，Δ 为 **−3.831748**，四位小数为 **−3.8317**；直接拿已舍入的 23.9046 与 20.0728 相减会产生末位舍入差。特征诊断中的 **−5.2454** 对其标明的原主实验 cut 95–315 是正确的，属于不同评估范围，并非配对或均值公式错误。两次评估重叠 cut 的预测最大绝对差为 0.00321；范围差异是主要原因。", "",
             "以下 AdaBN 坚持原主实验的评估协议：目标 C6 只用 cut 95–315 计算指标，其他目标用 cut 1–315。", "",
             "## 第二项：source-only + AdaBN", "",
             "各方向、seed 42–46 均从 `five_seed_paired` 的 Z-score source-only 最终 checkpoint 建立独立模型副本。目标输入只使用原源刀拟合的六通道均值和标准差；没有重拟合输入归一化，也没有改动 STFT、回归头或 checkpoint 参数。所有模型参数被冻结，BN 的 gamma/beta 保持原值。", "",
             "固定的 BN 重估方案：模型先整体 `eval()`，仅 20 个 ResNet18 BatchNorm 层设为 `train()`；执行 `reset_running_stats()`，设置 `momentum=None`（跨 batch 累积平均）；按 cut 1–315 升序、batch size 63、单次遍历、`torch.no_grad()`。315 恰好分成五个满 batch，实际没有不足一批；切片遍历等价于 `drop_last=False`，若有尾批也会处理。结束后整个模型恢复 `eval()` 再预测 315 个 cut。Dropout 和其他层始终不进入训练态。目标 315 个 cut 全部用于无标签 BN 重估，符合当前 transductive 协议。目标评估标签直到 BN 重估和预测完成才从原评估文件读取。", "",
             "运行核验：30/30 个实验的非 BN 状态逐项不变，参数全部冻结；每个实验的 20 个 BN 层运行统计量均改变，`num_batches_tracked=5`；AdaBN 前后 conv1 首批输入张量逐位相同；全部 315 个预测及四项评估指标为有限值。原模型按相同全 cut 输入复算的 RMSE 与既有 RMSE 最大偏差低于 0.00036（C6 旧预测的分批起点不同）。未用目标评估指标选择 BN 参数或 checkpoint。", "",
             "### 五 seed RMSE 对照（均值 ± 样本标准差）", "",
             "| 方向 | source-only | source-only + AdaBN | DARE-GRAM | 配对 ΔRMSE：AdaBN − source-only | 改善 seed 数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        s, a, d = (summary[source, target, m] for m in METHODS)
        sample = [r for r in wide if r["source"] == source and r["target"] == target]
        count = sum(r["delta_rmse_adabn_minus_source_only"] < 0 for r in sample)
        fmt = lambda r: f"{float(r['RMSE_mean']):.3f}±{float(r['RMSE_sd']):.3f}"
        lines.append(f"| {label(source, target)} | {fmt(s)} | {fmt(a)} | {fmt(d)} | "
                     f"{float(a['delta_rmse_adabn_minus_source_only_mean']):+.3f}±"
                     f"{float(a['delta_rmse_adabn_minus_source_only_sd']):.3f} | {count}/5 |")
    lines += ["", "### 四项指标（五 seed 均值 ± 样本标准差）", "",
              "| 方向 | 方法 | R² | MAE | RMSE | MAPE (%) |",
              "|---|---|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        for method in METHODS:
            r = summary[source, target, method]
            fmt = lambda metric: f"{float(r[metric + '_mean']):.3f}±{float(r[metric + '_sd']):.3f}"
            lines.append(f"| {label(source, target)} | {method} | " +
                         " | ".join(fmt(metric) for metric in METRICS) + " |")
    lines += ["", "### 逐 seed 配对 RMSE", "",
              "| 方向 | seed | source-only | source-only + AdaBN | DARE-GRAM | ΔRMSE |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in wide:
        lines.append(f"| {label(r['source'], r['target'])} | {r['seed']} | "
                     f"{r['source_only_RMSE']:.3f} | {r['source_only_adabn_RMSE']:.3f} | "
                     f"{r['daregram_RMSE']:.3f} | {r['delta_rmse_adabn_minus_source_only']:+.3f} |")
    lines += ["", "AdaBN 仅在 C1→C4 的 3/5 个 seed 降低 RMSE，该方向五 seed 平均 ΔRMSE 为负；其余五方向的平均 RMSE 均上升，且各自 5/5 个 seed 未改善。由此不能将 AdaBN 视为统一优于原 source-only 或 DARE-GRAM 的方法，也没有按目标评估指标临时选取方向特定的最优组合。", "",
              "## 文件", "",
              "- `paired_seed_comparison.csv`：30 行，三种方法的四项指标并列及配对 ΔRMSE。",
              "- `per_seed_metrics.csv`：90 行逐方法原始指标。",
              "- `direction_summary.csv`：18 行方法 × 方向的五 seed 均值与样本标准差。",
              "- `<source>_to_<target>/seed_<seed>/predictions_all_cuts.csv`：AdaBN 的全部 315 个目标预测及评估标志。",
              "- `<source>_to_<target>/seed_<seed>/adabn_bn_buffers.pth`：仅保存重估后的 BN 运行缓冲区；与记录的原 source-only checkpoint 组合即可复原 AdaBN 模型。",
              "- `<source>_to_<target>/seed_<seed>/audit.json`：checkpoint 哈希、归一化参数、BN 状态与输入等同性核验。",
              "- `../c1c6_metric_reconciliation_20260925/per_seed.csv` 与 `summary.json`：两种 C6 评估范围的逐 seed 对账。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {wide_path} and {report_path}")


if __name__ == "__main__":
    main()
