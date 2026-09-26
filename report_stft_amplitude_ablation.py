"""Summarize completed, precommitted STFT amplitude ablation without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_stft_amplitude_ablation as experiment


def row(table, scheme, method, stage):
    out = table[(table.scheme == scheme) & (table.method == method) & (table.stage == stage)]
    if len(out) != 1 or int(out.iloc[0].n_seeds) != 5:
        raise ValueError((scheme, method, stage))
    return out.iloc[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("artifacts/stft_amplitude_ablation_20260926"))
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / "c4_selection_lock.json").is_file():
        raise ValueError("C4 selection must be frozen before reporting")
    lock = json.loads((root / "c4_selection_lock.json").read_text(encoding="utf-8"))
    c4 = pd.read_csv(root / "c4_summary.csv")
    c6 = pd.read_csv(root / "c6_summary.csv")
    d4 = pd.read_csv(root / "c4_distances.csv")
    d6 = pd.read_csv(root / "c6_distances.csv")
    for target in ("c4", "c6"):
        for scheme in ("A", "B", "C"):
            for seed in range(42, 47):
                for method in ("source_only", "daregram"):
                    path = root / f"c1_to_{target}/{scheme}/seed_{seed}/{method}/predictions.csv"
                    x = pd.read_csv(path)
                    if list(x.columns) != ["cut_index", "true_vb", "pred_vb"] or x.cut_index.tolist() != list(range(1, 316)):
                        raise ValueError(f"Incomplete prediction: {path}")
                    if scheme == "B":
                        a = pd.read_csv(root / f"c1_to_{target}/A/seed_{seed}/{method}/predictions.csv")
                        if np.max(np.abs(a.pred_vb.to_numpy()-x.pred_vb.to_numpy())) > 1e-4:
                            raise ValueError(f"B identity mismatch: {path}")
    # B's input is exactly A's. Publish an explicit duplicate of input metrics.
    for target, distances in (("c4", d4), ("c6", d6)):
        input_b = distances[(distances.scheme == "A") & (distances.layer == "stft_48d")].copy()
        input_b["scheme"] = "B"
        if not ((distances.scheme == "B") & (distances.layer == "stft_48d")).any():
            distances = pd.concat([distances, input_b], ignore_index=True)
            distances.to_csv(root / f"{target}_distances.csv", index=False, float_format="%.17g")
        if target == "c4":
            d4 = distances
        else:
            d6 = distances
    input_stats = []
    for target in ("c4", "c6"):
        for scheme in ("A", "B", "C"):
            source, mean, std = experiment.normalized_inputs(root, "c1", scheme)
            tgt, _, _ = experiment.normalized_inputs(root, target, scheme, mean, std)
            for role, cube in (("c1", source), (target, tgt)):
                for lo, hi in experiment.STAGES:
                    for channel_index, channel in enumerate(("Fx", "Fy", "Fz", "Vx", "Vy", "Vz")):
                        z = cube[lo-1:hi, channel_index].astype(np.float64)
                        input_stats.append({"target": target, "scheme": scheme, "domain": role,
                                            "stage": f"{lo}-{hi}", "channel": channel,
                                            "mean": z.mean(), "std": z.std(),
                                            "p01": np.quantile(z, .01), "p50": np.quantile(z, .5),
                                            "p99": np.quantile(z, .99),
                                            "cut_rms_mean": np.sqrt(np.mean(z*z, axis=(1, 2))).mean()})
            del source, tgt
    pd.DataFrame(input_stats).to_csv(root / "input_channel_stats.csv", index=False, float_format="%.17g")
    lines = [
        "# STFT 幅值处理消融：C1→C4 后锁定方案，再评估 C1→C6", "",
        "## 配置与标签边界", "",
        "- A：原处理 `abs(STFT) → log1p → resize → C1 六通道 Z-score`，复用并逐 Cut 重放已经完成的五 seed 训练；全部原始预测重放误差须 ≤1e-4 VB。",
        "- B：按要求在非负 STFT 幅值上做 `log1p`；由于 A 本来就在这个位置做 `log1p`，B 与 A 数学上相同。B 独立训练以检查恒等对照，不能解释为第二种压缩强度。",
        "- C：`M=abs(STFT)`；每 Cut 的六通道 129×121 个幅值共同计算 `r=sqrt(mean(M²))`；`M'=M/max(r,1e-8)`；然后 `log1p(M')`、原 resize、C1 六通道 Z-score。常数为 `1e-8`。归一化可能移除磨损幅值信息。",
        "- 三组均使用原始信号中心 4096 点、50 kHz、Hann STFT 256/224、不执行频率裁剪、六通道 128×128、ResNet18 + Linear(512,1)、50 epoch、batch 63、Adam 0.001、同 seed 初始化与源域批次顺序、最终 epoch 权重。仅输入幅值处理不同。",
        "- C4 全 315 Cut 仅作为训练结束后的验证；C6 Cut 1–315 在 DARE-GRAM 训练时仅使用无标签 STFT，C6 标签在 `c4_selection_lock.json` 固定后、C6 全部训练完成后才用于评价。source-only 不使用目标输入训练。",
        f"- C4 固定选择：**{lock['selected_for_recommendation']}**。预定规则：{lock['rule']}。C6 的 A/B/C 均按预声明计划运行，C6 指标未用于选择方案。",
        "- 原训练权重与配置：`artifacts/five_seed_paired/c1_to_{c4,c6}/seed_{42..46}/{source_only,daregram}/`；A 的 C6 全程预测参照：`artifacts/norm_comparison_20260925/zscore/c1_to_c6/.../predictions.csv`。本实验权重、配置、缓存、逐 Cut 预测及审计均在本目录。", "",
        "## 五 seed 平均指标", "",
        "表中 R²、MAE、RMSE 是每个 seed 全程指标的平均值；后段误差与增量也先按 seed 计算再平均。", "",
        "| 目标 | 方案 | 方法 | R² | MAE | RMSE | 后段 MAE | 后段 pred−true | 后段预测增量 | 后段真实增量 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, table in (("C4", c4), ("C6", c6)):
        for scheme in ("A", "B", "C"):
            for method in ("source_only", "daregram"):
                full, late = row(table, scheme, method, "all"), row(table, scheme, method, "211-315")
                lines.append(f"| {target} | {scheme} | {method} | {full.R2_mean:.3f} | {full.MAE_mean:.2f} | "
                             f"{full.RMSE_mean:.2f} | {late.MAE_mean:.2f} | {late.signed_error_mean:+.2f} | "
                             f"{late.pred_increment_mean:.2f} | {late.true_increment_mean:.2f} |")
    lines += ["", "## 域差异与解释", "",
              "距离先用 C1 训练折逐维标准化，再取到 C1 训练特征的最近邻；表中为目标距离 / C1 留出距离的中位数比。STFT 使用每通道四个分布量和四个频带均值，共 48 维；ResNet 使用 512 维。跨模型特征坐标不同，512 维距离**不能直接用于 A/B/C 性能排序**。", "",
              "| 目标 | 方案 | STFT 前/中/后距离比 | source-only 512D 前/中/后距离比 | DARE-GRAM 512D 前/中/后距离比 |",
              "|---|---|---|---|---|"]
    for target, table in (("C4", d4), ("C6", d6)):
        for scheme in ("A", "B", "C"):
            values = []
            for layer, method in (("stft_48d", "shared"), ("resnet_512d", "source_only"),
                                  ("resnet_512d", "daregram")):
                group = table[(table.scheme == scheme) & (table.layer == layer) & (table.method == method)]
                ratios = [group[group.stage == stage].target_over_c1_holdout_median.mean()
                          for stage in ("1-105", "106-210", "211-315")]
                if not np.isfinite(ratios).all():
                    raise ValueError(f"Missing distances: {target} {scheme} {layer} {method}")
                values.append(" / ".join(f"{v:.2f}" for v in ratios))
            lines.append(f"| {target} | {scheme} | " + " | ".join(values) + " |")
    c1_wear = pd.read_csv(r"E:\QLP\source\source_mill\c1_wear.csv")[["flute_1", "flute_2", "flute_3"]].mean(axis=1)
    y_max = float(c1_wear.max())
    lines += ["", f"C1 训练标签上界为 **{y_max:.2f} VB**。C6 后段五 seed 均值预测是否集中于该上界，用每个 Cut 的均值曲线在 `[{y_max-10:.2f},{y_max+10:.2f}]` 内的比例表示：", ""]
    late_summary = {}
    for scheme in ("A", "B", "C"):
        for method in ("source_only", "daregram"):
            vectors = np.stack([pd.read_csv(root / f"c1_to_c6/{scheme}/seed_{seed}/{method}/predictions.csv").pred_vb.to_numpy()
                                for seed in range(42, 47)])
            late = vectors.mean(0)[210:]
            late_summary[scheme, method] = (float(np.mean(np.abs(late-y_max)<=10)), float(late.max()))
            lines.append(f"- {scheme} {method}：{late_summary[scheme,method][0]:.1%}；后段均值曲线最大值 {late_summary[scheme,method][1]:.2f} VB。")
    a_input = d6[(d6.scheme == "A") & (d6.layer == "stft_48d")]
    c_input = d6[(d6.scheme == "C") & (d6.layer == "stft_48d")]
    a_late = float(a_input[a_input.stage == "211-315"].target_over_c1_holdout_median.mean())
    c_late = float(c_input[c_input.stage == "211-315"].target_over_c1_holdout_median.mean())
    stats = pd.DataFrame(input_stats)
    def p99_ratio(scheme, channel):
        sub = stats[(stats.target == "c6") & (stats.scheme == scheme) &
                    (stats.stage == "211-315") & (stats.channel == channel)]
        return float(sub[sub.domain == "c6"].p99.iloc[0] / sub[sub.domain == "c1"].p99.iloc[0])
    c4_delta = [float(row(c4, "C", m, "all").MAE_mean-row(c4, "A", m, "all").MAE_mean) for m in ("source_only", "daregram")]
    c6_inc = [float(row(c6, "C", m, "211-315").pred_increment_mean-row(c6, "A", m, "211-315").pred_increment_mean)
              for m in ("source_only", "daregram")]
    lines += ["", "## 结论", "",
              f"- **输入域差异：混合结果。** C6 后段 STFT 48 维输入距离比由 A 的 {a_late:.2f} 降至 C 的 {c_late:.2f}，但振动通道的高分位数比反而增加。B=A。距离减小只说明此描述空间的输入分布更近，不证明保留了磨损信息。",
              "- C6 后段输入 Z-score 后的 p99 比值（C6/C1）：" + "; ".join(
                  f"{channel} A={p99_ratio('A',channel):.2f}, C={p99_ratio('C',channel):.2f}"
                  for channel in ("Vx", "Vy", "Vz")) + "。逐域、通道和阶段见 `input_channel_stats.csv`。",
              f"- **C4 预测受损。** C 相对 A 的全程 MAE：source-only {c4_delta[0]:+.2f}、DARE-GRAM {c4_delta[1]:+.2f} VB；后段 MAE 分别增加 {row(c4,'C','source_only','211-315').MAE_mean-row(c4,'A','source_only','211-315').MAE_mean:.2f} 和 {row(c4,'C','daregram','211-315').MAE_mean-row(c4,'A','daregram','211-315').MAE_mean:.2f} VB。因此 C4 预定规则保留 A。",
              f"- **C6 后段低估未改善。** C 相对 A 的预测增量变化：source-only {c6_inc[0]:+.2f}、DARE-GRAM {c6_inc[1]:+.2f} VB，但真实增量为 {row(c6,'A','source_only','211-315').true_increment_mean:.2f} VB，两个方法的后段 MAE 与负偏差均变大。",
              f"- **没有贴住 C1 标签上界。** A 的 DARE-GRAM 有 {late_summary['A','daregram'][0]:.1%} 的后段均值预测处于上界 ±10 VB；C 的 source-only、DARE-GRAM 分别为 {late_summary['C','source_only'][0]:.1%}、{late_summary['C','daregram'][0]:.1%}，最大值仅 {late_summary['C','source_only'][1]:.2f} 与 {late_summary['C','daregram'][1]:.2f} VB。C 使预测停在更低的位置，不能解释为改善上界饱和。",
              "- 这项实验表明当前逐 Cut 能量归一化不能修复低估；它同时可能移除了与磨损相关的幅值信息，因此不能据此断言原始输入域差异与低估无因果关系。",
              "- B 不构成独立幅值消融；若需要检验比原处理更强的压缩，应先明确不同变换，并开展新实验，不能把第二次 `log1p` 冒充当前 B。", "",
              "## 输出与命令", "",
              "- `c4_metrics.csv`、`c6_metrics.csv`：每 seed 全程及三个阶段；`c4_summary.csv`、`c6_summary.csv`：五 seed 汇总；`c4_distances.csv`、`c6_distances.csv`：STFT 和 512 维距离；`input_channel_stats.csv`：STFT 输入统计；`c*_checkpoint_audit.csv`：权重与批次顺序；`c1_to_*/{A,B,C}/seed_*/.../predictions.csv`：逐 Cut 预测。",
              "- `c1_to_*/{c4,c6}_{A,B,C}_curves.png`：同一目标的三组图共用纵坐标。`cache_C/*_energy_divisors.csv`：逐 Cut 能量与除零常数。", "",
              "```powershell", "cd upstream-reproduction",
              "python run_stft_amplitude_ablation.py --phase c4 --out-root artifacts/stft_amplitude_ablation_20260926",
              "python run_stft_amplitude_ablation.py --phase c6 --out-root artifacts/stft_amplitude_ablation_20260926",
              "python report_stft_amplitude_ablation.py --root artifacts/stft_amplitude_ablation_20260926", "```", "",
              "未加入 OOR-PGA、门控、物理约束、目标域监督微调或预测后处理。"]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {root / 'README.md'}")


if __name__ == "__main__":
    main()
