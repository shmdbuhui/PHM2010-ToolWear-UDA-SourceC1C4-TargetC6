"""Compare frozen center-window A with M center and three-window mean, by Cut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_multi_window_ablation as multi
import run_single_source_pairs as base


STAGES = {"full": (1, 315), "early": (1, 105),
          "middle": (106, 210), "late": (211, 315)}
VARIANTS = ("A", "M_center", "M_mean")
METRICS = ("R2", "MAE", "RMSE", "signed_error", "late_pred_increment")


def measure(cuts: np.ndarray, truth: np.ndarray, pred: np.ndarray, stage: str) -> dict:
    lo, hi = STAGES[stage]
    mask = (cuts >= lo) & (cuts <= hi)
    y, p = truth[mask], pred[mask]
    result = {"n_cuts": len(y), "MAE": float(np.mean(np.abs(p - y))),
              "RMSE": float(np.sqrt(np.mean(np.square(p - y)))),
              "signed_error": float(np.mean(p - y)),
              "late_pred_increment": float(pred[-1] - pred[210])}
    result["R2"] = float(1 - np.sum(np.square(p - y)) / np.sum(np.square(y - np.mean(y))))
    if stage != "late":
        result["late_pred_increment"] = np.nan
    return result


def summarize(frame: pd.DataFrame, keys: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = len(group)
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(a_root: Path, m_root: Path, require_complete: bool) -> None:
    if a_root.resolve() == m_root.resolve():
        raise ValueError("A and M roots must be different")
    positions = []
    for tool in multi.TOOLS:
        path = m_root / "feature_cache" / f"{tool}_windows.csv"
        frame = pd.read_csv(path)
        if (len(frame) != 945 or frame.cut_index.tolist() != list(np.repeat(base.ALL_CUTS, 3)) or
            frame.window.tolist() != [0, 1, 2] * 315 or
            (frame.end_exclusive_0based - frame.start_0based).ne(4096).any()):
            raise ValueError(f"Window position alignment failed: {path}")
        positions.append(frame)
    pd.concat(positions, ignore_index=True).to_csv(m_root / "all_window_positions.csv", index=False)

    a_index = pd.read_csv(a_root / "per_seed_metrics_full_1_315.csv")
    rows, deltas, provenance = [], [], []
    for source, target in multi.PAIRS:
        for seed in multi.SEEDS:
            folder = m_root / f"{source}_to_{target}" / f"seed_{seed}"
            if not (folder / "complete.json").is_file():
                if require_complete:
                    raise FileNotFoundError(f"M experiment incomplete: {folder}")
                continue
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if (config["source"], config["target"], config["seed"], config["target_labels_training_reads"]) != (
                    source, target, seed, 0):
                raise ValueError(f"M config mismatch: {folder}")
            epoch_audits = {method: pd.read_csv(folder / method / "epoch_audit.csv")
                            for method in multi.METHODS}
            for method, audit in epoch_audits.items():
                expected_target = 315 if method == "daregram" else 0
                if (audit.epoch.tolist() != list(range(1, 51)) or
                    not audit.source_cuts.eq(315).all() or
                    not audit.target_cuts.eq(expected_target).all() or
                    not audit.steps.eq(5).all()):
                    raise ValueError(f"M epoch/Cut audit mismatch: {folder / method}")
            if (not epoch_audits["source_only"].source_order_sha256.equals(
                    epoch_audits["daregram"].source_order_sha256) or
                not epoch_audits["source_only"].source_windows_sha256.equals(
                    epoch_audits["daregram"].source_windows_sha256)):
                raise ValueError(f"M methods did not share source Cut/window choices: {folder}")
            for method in multi.METHODS:
                m_path = folder / method / "predictions.csv"
                a_path = a_root / "per_seed" / f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
                m, a = pd.read_csv(m_path), pd.read_csv(a_path)
                a_record = a_index[(a_index.source == source) & (a_index.target == target) &
                                   (a_index.seed == seed) & (a_index.method == method)]
                if len(a_record) != 1 or base.file_hash(a_path) != a_record.iloc[0].per_cut_sha256:
                    raise ValueError(f"Frozen A per-Cut provenance mismatch: {a_path}")
                if (m.cut_index.tolist() != base.ALL_CUTS or a.cut_index.tolist() != base.ALL_CUTS or
                    not np.allclose(m.true_vb, a.true_vb, atol=1e-6, rtol=0)):
                    raise ValueError(f"A/M Cut or label mismatch: {source}->{target} {seed} {method}")
                if not np.isfinite(m[["pred_w25", "pred_center", "pred_w75", "pred_mean"]]).all().all():
                    raise ValueError(f"Invalid M prediction: {m_path}")
                if not np.allclose(m.pred_mean, m[["pred_w25", "pred_center", "pred_w75"]].mean(axis=1),
                                   atol=1e-10, rtol=0):
                    raise ValueError(f"M aggregation mismatch: {m_path}")
                cuts = m.cut_index.to_numpy(int)
                y = m.true_vb.to_numpy(float)
                predictions = {"A": a.pred_vb.to_numpy(float),
                               "M_center": m.pred_center.to_numpy(float),
                               "M_mean": m.pred_mean.to_numpy(float)}
                for stage in STAGES:
                    by_variant = {}
                    for variant, pred in predictions.items():
                        item = {"source": source, "target": target, "seed": seed,
                                "method": method, "variant": variant, "stage": stage,
                                **measure(cuts, y, pred, stage)}
                        rows.append(item)
                        by_variant[variant] = item
                    for contrast, v0, v1 in (("M_center_minus_A", "A", "M_center"),
                                              ("M_mean_minus_A", "A", "M_mean"),
                                              ("M_mean_minus_M_center", "M_center", "M_mean")):
                        deltas.append({"source": source, "target": target, "seed": seed,
                                       "method": method, "stage": stage, "contrast": contrast,
                                       **{key: by_variant[v1][key] - by_variant[v0][key]
                                          for key in METRICS}})
                provenance.append({"source": source, "target": target, "seed": seed, "method": method,
                                   "A_prediction": str(a_path.resolve()), "A_sha256": base.file_hash(a_path),
                                   "M_prediction": str(m_path.resolve()), "M_sha256": base.file_hash(m_path),
                                   "M_checkpoint": str((folder / method / "final.pth").resolve()),
                                   "M_checkpoint_sha256": base.file_hash(folder / method / "final.pth")})
    if require_complete and len(provenance) != 60:
        raise ValueError(f"Expected 60 paired results; found {len(provenance)}")
    if not rows:
        raise ValueError("No completed M experiments")
    metrics = pd.DataFrame(rows)
    difference = pd.DataFrame(deltas)
    metrics.to_csv(m_root / "per_seed_metrics.csv", index=False)
    difference.to_csv(m_root / "per_seed_deltas.csv", index=False)
    summarize(metrics, ["source", "target", "method", "variant", "stage"], METRICS).to_csv(
        m_root / "six_direction_summary.csv", index=False)
    summarize(difference, ["source", "target", "method", "stage", "contrast"], METRICS).to_csv(
        m_root / "paired_delta_summary.csv", index=False)
    pd.DataFrame(provenance).to_csv(m_root / "provenance.csv", index=False)
    make_report(m_root, pd.DataFrame(rows), pd.DataFrame(deltas), len(provenance))


def make_report(root: Path, metrics: pd.DataFrame, deltas: pd.DataFrame, complete: int) -> None:
    summary = summarize(metrics, ["source", "target", "method", "variant", "stage"], METRICS)
    paired = summarize(deltas, ["source", "target", "method", "stage", "contrast"], METRICS)
    lines = ["# 多窗口信号截取消融", "",
             f"完成 {complete}/60 个方向 × seed × 方法实验。表中先逐 seed 计算，再报告均值 ± 样本标准差。",
             "", "## 协议", "",
             "A 直接读取冻结的 `artifacts/full_1_315_baseline_zscore_20260925/per_seed/`。M 每个 Cut 在原始信号 25%、50%、75% 位置取 4096 点，保留原 STFT/log1p/128×128、ResNet18、回归头和 DARE-GRAM 损失。", "",
             "M 逐通道 z-score 仅由源刀具 315 个训练 Cut 的三个窗口拟合；目标域沿用同一统计量。每 epoch 每 Cut 抽取一个窗口，315 个源 Cut、DARE-GRAM 的 315 个无标签目标 Cut 均各出现一次，每 epoch 5 步。两方法使用同样的源窗口抽样。50 epoch、batch 63、Adam 0.001、最后一轮 checkpoint。", "",
             "窗口越界时夹到合法范围；窗口含非有限值或六通道全零行时，取距离要求位置最近的完整有效窗口，平局选较早位置。每个窗口实际位置和原因见 `all_window_positions.csv`。", "",
             "三窗口预测在原始 VB 数值上直接算术平均。阶段为 1–105、106–210、211–315；有符号误差定义为预测减真实值；后段预测增量定义为预测 Cut 315 减预测 Cut 211。", "",
             "## 全程 1–315", "",
             "| 方向 | 方法 | A R² / MAE / RMSE | M 中心 R² / MAE / RMSE | M 均值 R² / MAE / RMSE |",
             "|---|---|---:|---:|---:|"]
    def fmt(row):
        return " / ".join(f"{row[f'{name}_mean']:.3f}±{row[f'{name}_sd']:.3f}"
                          if row["n_seeds"] > 1 else f"{row[f'{name}_mean']:.3f}"
                          for name in ("R2", "MAE", "RMSE"))
    for source, target in multi.PAIRS:
        for method in multi.METHODS:
            vals = [summary[(summary.source == source) & (summary.target == target) &
                            (summary.method == method) & (summary.variant == variant) &
                            (summary.stage == "full")].iloc[0] for variant in VARIANTS
                    if len(summary[(summary.source == source) & (summary.target == target) &
                                    (summary.method == method) & (summary.variant == variant) &
                                    (summary.stage == "full")])]
            if len(vals) == 3:
                lines.append(f"| {source.upper()}→{target.upper()} | {method} | " +
                             " | ".join(fmt(v) for v in vals) + " |")
    lines += ["", "## 配对变化：M 均值减 A", "",
              "负 ΔMAE / ΔRMSE 表示改善。后段 signed error 趋近 0 表示低估减轻。", "",
              "| 方向 | 方法 | ΔMAE 全程 | ΔRMSE 全程 | 后段 ΔMAE | 后段 Δ有符号误差 | 后段 Δ预测增量 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for source, target in multi.PAIRS:
        for method in multi.METHODS:
            f = paired[(paired.source == source) & (paired.target == target) &
                       (paired.method == method) & (paired.contrast == "M_mean_minus_A")]
            if len(f) != 4:
                continue
            full, late = f[f.stage == "full"].iloc[0], f[f.stage == "late"].iloc[0]
            def val(row, key):
                return f"{row[f'{key}_mean']:+.2f}±{row[f'{key}_sd']:.2f}" if row.n_seeds > 1 else f"{row[f'{key}_mean']:+.2f}"
            lines.append(f"| {source.upper()}→{target.upper()} | {method} | {val(full, 'MAE')} | "
                         f"{val(full, 'RMSE')} | {val(late, 'MAE')} | "
                         f"{val(late, 'signed_error')} | {val(late, 'late_pred_increment')} |")
    if complete == 60:
        full_delta = deltas[deltas.stage == "full"]
        avg = full_delta[full_delta.contrast == "M_mean_minus_A"]
        center = full_delta[full_delta.contrast == "M_center_minus_A"]
        aggregation = full_delta[full_delta.contrast == "M_mean_minus_M_center"]
        avg_groups = avg.groupby(["source", "target", "method"])
        center_groups = center.groupby(["source", "target", "method"])
        agg_groups = aggregation.groupby(["source", "target", "method"])
        avg_better = sum(g.MAE.mean() < 0 for _, g in avg_groups)
        avg_all_seeds_better = sum((g.MAE < 0).all() for _, g in avg_groups)
        all_seed_names = [f"{s.upper()}→{t.upper()} {method}" for (s, t, method), g in avg_groups
                          if (g.MAE < 0).all()]
        center_better = sum(g.MAE.mean() < 0 for _, g in center_groups)
        aggregation_better = sum(g.MAE.mean() < 0 for _, g in agg_groups)
        lines += ["", "## 五 seed 判断", "",
                  f"M 三窗口平均的全程 MAE 在 12 个方向×方法组合中有 {avg_better} 个均值优于 A，"
                  f"其中 {avg_all_seeds_better} 个组合的五个 seed 全部改善。因此六方向结果"
                  + ("稳定改善。" if avg_all_seeds_better == 12 else "没有稳定改善；保留 A 为基准，M 作为消融。"),
                  "五个 seed 全部改善的组合：" + ("、".join(all_seed_names) if all_seed_names else "无") + "。",
                  "",
                  f"M 中心对 A 的全程 MAE 均值有 {center_better}/12 个组合改善；"
                  f"M 三窗口平均对 M 中心有 {aggregation_better}/12 个组合改善。"
                  "前一比较反映多位置训练和源域三窗口 z-score 统计量共同作用；后一比较只反映推理聚合。"
                  "整体收益更常来自推理聚合，训练侧变化并未普遍改善；C1→C6 source_only 则是训练侧改善而聚合略退化。", ""]
        lines += ["C6 后段（Cut 211–315）的低估和误差：", "",
                  "| 方向 | 方法 | A 后段有符号误差 | M 均值后段有符号误差 | A 后段 MAE | M 均值后段 MAE |",
                  "|---|---|---:|---:|---:|"]
        for source in ("c1", "c4"):
            for method in multi.METHODS:
                sub = summary[(summary.source == source) & (summary.target == "c6") &
                              (summary.method == method) & (summary.stage == "late")]
                a = sub[sub.variant == "A"].iloc[0]
                m = sub[sub.variant == "M_mean"].iloc[0]
                lines.append(f"| {source.upper()}→C6 | {method} | {a.signed_error_mean:+.2f} | "
                             f"{m.signed_error_mean:+.2f} | {a.MAE_mean:.2f} | {m.MAE_mean:.2f} |")
        lines += ["", "C1→C6 的 source_only 后段低估明显减轻，DARE-GRAM 仅小幅减轻；"
                  "C4→C6 的两种方法后段低估均加重，后段 MAE 也升高。", ""]
        worst = avg_groups.MAE.mean().sort_values(ascending=False).head(3)
        lines += ["", "全程 MAE 均值退化最大的组合：" +
                  "；".join(f"{s.upper()}→{t.upper()} {method} +{value:.2f}"
                           for (s, t, method), value in worst.items()) + "。", ""]
        source_path = root / "source_domain_audit" / "summary.csv"
        if source_path.is_file():
            source_summary = pd.read_csv(source_path)
            wide = source_summary.pivot(index=["source", "target", "method"],
                                        columns="variant", values="MAE_mean")
            degradation = wide["M_mean"] - wide["A"]
            lines += [f"源域训练 Cut 的样本内 MAE：M 均值有 {(degradation > 0).sum()}/12 个组合高于 A；"
                      f"最大增加 {degradation.max():+.2f}。这是训练数据拟合诊断，不代表独立源域泛化。", ""]
    lines += ["", "## 解释边界", "",
              "M 中心与 A 的差异包含多位置训练及源域三窗口统计量变化；M 均值与 M 中心的配对差异直接量化三窗口推理聚合。完整早/中/晚分段和三种配对差值见 CSV。", "",
              "## 复现", "", "在 `upstream-reproduction` 目录执行：", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe run_multi_window_ablation.py --prepare-only",
              ".\\.venv\\Scripts\\python.exe run_multi_window_ablation.py --seeds 42",
              ".\\.venv\\Scripts\\python.exe run_multi_window_ablation.py --seeds 43 44 45 46",
              ".\\.venv\\Scripts\\python.exe audit_multi_window_source.py",
              ".\\.venv\\Scripts\\python.exe report_multi_window_ablation.py", "```", "",
              "A 来源：`artifacts/full_1_315_baseline_zscore_20260925/`；M 输出：`artifacts/multi_window_25_50_75_20260926/`。`provenance.csv` 保存逐组 A 预测文件及哈希、M 预测和 checkpoint 哈希。", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    parser.add_argument("--m-root", type=Path, default=multi.DEFAULT_OUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run(args.a_root, args.m_root, require_complete=not args.allow_partial)
    print(f"Compared A and M; wrote {args.m_root / 'README.md'}")


if __name__ == "__main__":
    main()
