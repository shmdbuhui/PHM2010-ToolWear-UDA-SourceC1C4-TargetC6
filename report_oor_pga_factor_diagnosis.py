"""Report the frozen A-E OOR-PGA factor mechanism audit without method selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("artifacts/oor_pga_factor_diagnosis_full_1_315_20260925")
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))
METHODS = ("source_only", "daregram")
ARMS = ("A_original", "B_original_capped_1p5", "C_delayed_uncapped",
        "D_delayed_capped_1p5", "E_early_reference_persistent")


def row(summary: pd.DataFrame, source: str, target: str, method: str,
        arm: str, segment: str) -> pd.Series:
    result = summary[(summary.source == source) & (summary.target == target) &
                     (summary.base_method == method) & (summary.arm == arm) &
                     (summary.segment == segment)]
    if len(result) != 1:
        raise ValueError("Missing arm summary")
    return result.iloc[0]


def main() -> None:
    report = ROOT / "README.md"
    if report.exists():
        raise FileExistsError(report)
    factor_audit = pd.read_csv(ROOT / "factor_trigger_m_audit_full_1_315.csv", keep_default_na=False)
    summary = pd.read_csv(ROOT / "five_seed_summary_full_1_315.csv")
    paired = pd.read_csv(ROOT / "paired_five_seed_delta_vs_A_E_full_1_315.csv")
    errors = pd.read_csv(ROOT / "per_cut_error_decomposition_full_1_315.csv")
    if len(factor_audit) != 6 or len(summary) != 240 or len(paired) != 288 or len(errors) != 3780:
        raise ValueError("Incomplete factor audit")
    focus_rows = []
    for source, target, method in (("c1", "c6", "source_only"),
                                    ("c1", "c6", "daregram"),
                                    ("c4", "c1", "source_only"),
                                    ("c4", "c1", "daregram"),
                                    ("c6", "c1", "daregram")):
        files = [pd.read_csv(ROOT / "evaluated_per_seed" /
                             f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv")
                 for seed in range(42, 47)]
        baseline = np.stack([f.baseline_pred_vb.to_numpy(float)[210:] -
                             f.true_vb.to_numpy(float)[210:] for f in files])
        baseline_sse = (baseline ** 2).sum(axis=0)
        for arm in ARMS:
            e = np.stack([f[f"pred_{arm}_vb"].to_numpy(float)[210:] -
                          f.true_vb.to_numpy(float)[210:] for f in files])
            below = e[e < 0]
            sse = (e ** 2).sum(axis=0)
            focus_rows.append({"source": source, "target": target, "base_method": method,
                               "arm": arm, "scope": "full_1_315", "segment": "late_211_315",
                               "n_seed_cut_predictions": e.size,
                               "underprediction_seed_cut_count": len(below),
                               "mean_underprediction_magnitude_conditional": float(-below.mean()) if len(below) else "",
                               "mean_signed_error_all_seed_cuts": float(e.mean()),
                               "mean_positive_overprediction_all_seed_cuts": float(np.maximum(e, 0).mean()),
                               "pooled_cut_count_worse_than_baseline_SSE": int((sse > baseline_sse + 1e-10).sum()),
                               "pooled_cut_count_better_than_baseline_SSE": int((sse < baseline_sse - 1e-10).sum())})
    pd.DataFrame(focus_rows).to_csv(ROOT / "focus_late_underprediction_full_1_315.csv",
                                    index=False, float_format="%.17g")
    lines = ["# PHM2010 full_1_315：乘法式 OOR-PGA 因子机制诊断", "",
             "## 范围、顺序与复现门槛", "",
             "只读取已冻结的六方向×seed 42–46×source-only/DARE-GRAM 原预测及两套 OOR 因子；没有重训、改 checkpoint、改门控参数或覆盖先前结果。`python diagnose_oor_pga_factors.py prepare` 首先核对基线、原 OOR、新门控的输入锁及 checkpoint/预测哈希，在**不读取目标磨损标签**的阶段保存 60 份含 A–E 因子与预测的 `unlabeled_predictions/`，并写 `prediction_lock_before_target_labels.json`。`python diagnose_oor_pga_factors.py evaluate` 核验所有冻结文件后才读取目标真实磨损，计算指标和误差图。全部结果只采用 cut 1–315。", "",
             "A=已保存的原触发点/原乘法因子；B=min(A,1.5)；C=新门控确认点 t0 后 `(τ/τ_t0)^m`，不加上限或软权重；D=min(C,1.5)；E=已保存的完整新门控。无有效新触发则 C/D 保持原预测，原因写入审计 CSV。A/E 的逐 cut 预测已与既有无标签文件核对，全部方向、seed、方法和阶段的 R²/MAE/RMSE/偏差亦逐项复现，之后才评价 B–D。B–D **仅用于拆解机制**，不作为按目标指标选出的新方法。", "",
             "源域幂指数 m 的逐 cut 原 OOR 文件与新门控触发记录，六方向以 Python 十进制文本转 float 后**完全相同**；差值均为 0。用 pandas 快速解析汇总 CSV 时个别值可显出约 10⁻¹⁶ 的末位差，这是读写/解析精度，不是重新拟合。B–D 用原逐 cut 文件的共同源域 m；A/E 保留原始因子和预测。", "",
             "## 触发、晚期因子与 1.5 上限", "",
             "| 方向 | m | 原触发 | 新确认 t0 | A 晚期平均/最大因子 | E 晚期平均/最大因子 | B 因 1.5 截断 cut | E 达到 1.5 cut |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        r = factor_audit[(factor_audit.source == source) & (factor_audit.target == target)].iloc[0]
        lines.append(f"| {source.upper()}→{target.upper()} | {float(r.old_m):.6f} | {r.old_trigger_cut} | "
                     f"{r.new_confirmed_trigger_cut} | {float(r.A_original_late_mean_factor):.3f} / "
                     f"{float(r.A_original_late_max_factor):.3f} | "
                     f"{float(r.E_early_reference_persistent_late_mean_factor):.3f} / "
                     f"{float(r.E_early_reference_persistent_late_max_factor):.3f} | "
                     f"{int(r.B_original_capped_1p5_full_at_1p5_count)} | "
                     f"{int(r.E_early_reference_persistent_full_at_1p5_count)} |")
    lines.extend(["", "`factors/` 含六方向 A–E 因子随 cut 变化的图及逐 cut r_t、d_t、新门控软权重和五组因子。`factor_trigger_m_audit_full_1_315.csv` 也列出每组晚期平均因子、最大因子、首次达到 1.5 的 cut 和数量。E 的六方向从未达到 1.5；C/D/E 的因子逐 cut 相同到 ≤4.5×10⁻¹⁶，因为触发后增长实际发生的 cut 上软权重均为 1，延迟后的纯乘法因子也均低于 1.5。", "",
                  "## 五组因子的 full_1_315 RMSE：五 seed 均值 ± 样本标准差", "",
                  "| 方向 | 基模型 | A 原 OOR | B 原因子截断 | C 仅延迟 | D 延迟+截断 | E 完整新门控 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for source, target in PAIRS:
        for method in METHODS:
            values = []
            for arm in ARMS:
                r = row(summary, source, target, method, arm, "full")
                values.append(f"{r.RMSE_mean:.2f} ± {r.RMSE_sd:.2f}")
            lines.append(f"| {source.upper()}→{target.upper()} | {method} | " + " | ".join(values) + " |")
    lines.extend(["", "所有 A–E 的全程与早期 1–105、中期 106–210、晚期 211–315 的逐 seed R²、MAE、RMSE、偏差和五 seed 均值±样本标准差见 `per_seed_metrics_full_1_315.csv`、`five_seed_summary_full_1_315.csv`。B–D 相对 A 与 E 的同 seed 配对 ΔR²/MAE/RMSE/偏差、改善 seed 数分别见 `paired_seed_delta_vs_A_E_full_1_315.csv`、`paired_five_seed_delta_vs_A_E_full_1_315.csv`。", "",
                  "## C1→C6：为何新门控收益较少", "",
                  "C1→C6 的 A 在 cut 161 触发，E/C/D 在 cut 210 确认。A 最大因子 1.242，C/E 最大 1.140，均未达到 1.5；因此 B=A、D=C。C=E 的实际预测也逐点一致：软权重没有进一步压低此方向的修正。**收益损失来自延迟触发及由较晚 t0 产生的较小晚期比例因子。**", "",
                  "| 基模型 | 中期 RMSE A→E | 晚期 RMSE A→E | 晚期平均有符号误差 A→E | 晚期低估的 seed-cut 数 A→E | 晚期低估条件均值 A→E |",
                  "|---|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        mid_a, mid_e = (row(summary, "c1", "c6", method, arm, "middle")
                        for arm in ("A_original", "E_early_reference_persistent"))
        late_a, late_e = (row(summary, "c1", "c6", method, arm, "late")
                          for arm in ("A_original", "E_early_reference_persistent"))
        fa = next(r for r in focus_rows if (r["source"], r["target"], r["base_method"], r["arm"]) ==
                  ("c1", "c6", method, "A_original"))
        fe = next(r for r in focus_rows if (r["source"], r["target"], r["base_method"], r["arm"]) ==
                  ("c1", "c6", method, "E_early_reference_persistent"))
        lines.append(f"| {method} | {mid_a.RMSE_mean:.2f}→{mid_e.RMSE_mean:.2f} | "
                     f"{late_a.RMSE_mean:.2f}→{late_e.RMSE_mean:.2f} | "
                     f"{late_a.signed_bias_mean:+.2f}→{late_e.signed_bias_mean:+.2f} | "
                     f"{fa['underprediction_seed_cut_count']}→{fe['underprediction_seed_cut_count']} /525 | "
                     f"{fa['mean_underprediction_magnitude_conditional']:.2f}→"
                     f"{fe['mean_underprediction_magnitude_conditional']:.2f} |")
    lines.extend(["", "A 在 cut 161–209 的修正略使中期 RMSE 增加；E 延迟触发使这一段回到基线。可是晚期 cut 211–315 中，A 的五 seed 合计平方误差在 source-only 的 104/105 cut、DARE-GRAM 的 84/105 cut 小于 E。晚期损失超过中期获益。C1→C6 的两张 `focus_figures/` 图并列原预测、A、B、C、D、E、真实磨损与 A/E 有符号误差；`per_cut_error_decomposition_full_1_315.csv` 保存每 cut 五 seed SSE。", "",
                  "## 其他两处退步", "",
                  "- C4→C1：新门控 t0=259，F 在后段缓慢增大（最大约 1.072），源预测晚期本就偏高：source-only/DARE-GRAM 原基线晚期平均偏差约 +17.75/+18.50，新门控变为 +21.41/+22.23。实际被修正的 56 个晚期 cut，五 seed 合计 SSE 在两种基模型下均逐 cut 增大；所以即便消除了原方案的极早触发灾难，E 相对未修正基线仍退步。", "- C6→C1 的 DARE-GRAM：原基线晚期平均偏差 +8.06，新门控使其变为 +10.18；新因子在 cut 270–315 大于 1，46 个被修正 cut 的五 seed 合计 SSE 均增加。对应全程 RMSE 9.53→10.46。原 OOR 在 cut 258 更早放大，RMSE 11.00，更差。", "",
                  "## 文件与解释边界", "",
                  "`unlabeled_predictions/` 为标签读取前冻结的 60 份 A–E 预测和因子；`evaluated_per_seed/` 加入真实标签和逐 cut 有符号误差。`focus_late_underprediction_full_1_315.csv` 给出晚期低估数量与条件平均低估量；`focus_figures/` 给出 C1→C6、C4→C1、C6→C1 两种基模型的逐 cut 预测与误差图。`prediction_lock_before_target_labels.json`、`evaluation_manifest_full_1_315.json`、`factor_trigger_m_audit_full_1_315.csv` 记录源数据、旧缓存/预测、checkpoint、因子和命令的哈希与来源。", "",
                  "这是对已看过目标域结果的诊断性反事实。B–D 没有成为新的预选方法；这些对照只能说明本次固定预测与因子下哪一步改变了误差，不能证明在独立数据上的最佳触发点或上限。", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
