"""Summarize the frozen early-reference persistent OOR-PGA exploratory control."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("artifacts/early_reference_persistent_oor_pga_full_1_315_20260925")
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))
METHODS = ("source_only", "daregram")
VARIANTS = ("baseline", "original_multiplicative_oor_pga",
            "early_reference_persistent_oor_pga")


def pm(row: pd.Series, name: str, digits: int = 2) -> str:
    return f"{row[name + '_mean']:.{digits}f} ± {row[name + '_sd']:.{digits}f}"


def name(source: str, target: str) -> str:
    return f"{source.upper()}→{target.upper()}"


def main() -> None:
    path = ROOT / "README.md"
    if path.exists():
        raise FileExistsError(path)
    audit = pd.read_csv(ROOT / "input_feature_audit_full_1_315.csv")
    origin = pd.read_csv(ROOT / "input_checkpoint_prediction_audit_full_1_315.csv")
    triggers = pd.read_csv(ROOT / "trigger_summary_full_1_315.csv")
    metrics = pd.read_csv(ROOT / "five_seed_summary_full_1_315.csv")
    paired = pd.read_csv(ROOT / "paired_five_seed_delta_full_1_315.csv")
    if len(origin) != 60 or len(triggers) != 6 or len(metrics) != 144 or len(paired) != 96:
        raise ValueError("Incomplete audit or evaluation")
    lines = ["# PHM2010 full_1_315：early-reference persistent OOR-PGA 探索性对照", "",
             "## 顺序、数据与固定方法", "",
             "本轮只对既有 Z-score source-only、DARE-GRAM 最终 checkpoint 的六方向×seed 42–46 全程预测做后处理，不重训、不修改 checkpoint、不覆盖原乘法式 OOR-PGA。依次运行：", "",
             "1. `python run_early_reference_persistent_oor_pga.py audit`：读取目标刀 cut 1–315 的**无标签** STFT、既有六通道×八频带 STFT-48、源刀逐维 min/max、原 OOR rate 和 60 组原预测；复算越界 flag/rate、核对 cut、缓存、预测和 checkpoint SHA-256，输出 `r_profiles/`。此阶段不读目标磨损。",
             "2. `python run_early_reference_persistent_oor_pga.py prepare`：仅读源刀 315 个训练标签拟合单调阶段，复用先前的源刀 PGA 指数 m；用同方向共同的无标签门控生成并冻结 60 份不含目标标签的逐 cut 预测和 `prediction_lock_before_target_labels.json`。",
             "3. `python run_early_reference_persistent_oor_pga.py evaluate`：核验全部 60 份冻结预测及哈希后，才读取目标真实磨损并计算指标、作图。", "",
             "所有文件名和图标题标注 full_1_315；原 C6 95–315 旧口径没有进入本次表格。CSV 序列化造成的原预测最大绝对差低于 10⁻¹²，逐项按 cut 顺序一致；逐文件差值和来源见 `input_checkpoint_prediction_audit_full_1_315.csv`。48 维定义、源刀支持范围与原 OOR 文件逐维核对，见 `input_feature_audit_full_1_315.csv` 和原实验 `source_48_feature_support.csv`。STFT 原始频轴 0–25 kHz，129 原始频率 bin 调整成 128 行后按每带 16 行分八带，每带取频率/时间均值。", "",
             "固定规则：`b=median(r_1..r_32)`，`s_t=median(r_max(1,t−4)..r_t)`（只看当前与过去），`d_t=max(0,s_t−b)`；δL=2/48、δH=5/48、K=5、Fmax=1.5。源刀 315 个磨损标签用 `IsotonicRegression(increasing=True)` 拟合单调趋势，以拟合起点到终点磨损增量达到 60% 的首个 cut 定义 τ_source。确认计数仅在 cut>32、τ_t≥τ_source 且 d_t≥δH 时累加，其余 cut 清零；第 5 个合格 cut 才触发 t0，不回填。触发前及触发 cut 因子为 1；之后 `w=clip((d−δL)/(δH−δL),0,1)`，`growth=clip((τ/τ_t0)^m−1,0,Fmax−1)`，`F=1+w·growth`，预测乘 F。m≤0、源趋势异常或无触发时保持原预测并记录原因。本轮三个源刀趋势有效、m>0、六方向均触发。", "",
             "同方向五 seed 与两种基模型复用完全相同的 b、d、τ_source、t0、m 和 F。所有阈值、连续长度、因子上限与 60% 阶段守卫均为本次固定的探索性设定，没有按目标指标搜索或按方向开关。**这些参数是看过目标域现象后提出的，本轮结果不构成独立测试验证。** 本方法是乘法后处理，区别于原单次 OOR-PGA 和累积加法式 OOR-PGA-like。", "",
             "## 标签读取前的 r_t 曲线审计", "",
             "| 方向 | 前 32 cut 中位数 b | 晚期 211–315 r 中位数 | 晚期−早期 | 原 OOR 首触发 cut |",
             "|---|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        r = audit[(audit.source == source) & (audit.target == target)].iloc[0]
        lines.append(f"| {name(source,target)} | {r.r_early_1_32_median:.3f} | "
                     f"{r.r_late_211_315_median:.3f} | {r.r_late_minus_early_median:+.3f} | "
                     f"{int(r.original_oor_trigger_cut)} |")
    lines.extend(["", "C1→C6 的晚期 r 中位数比前 32 cut 高 0.292，曲线后期持续抬升。C1→C4、C4→C1、C6→C4 前 32 cut 的 r 中位数分别约 0.021、0、0.010；原方案的 cut 2–3 触发主要是短暂高值/尖峰，而非贯穿早期的高静态基线。六张原始 r 曲线在 `r_profiles/`，没有按这些观察修改固定参数。", "",
                  "## 源阶段与新门控触发", "",
                  "| 方向 | 源刀 60% cut | τ_source | 新 t0 cut | 最大 F | 实际修正 cut 数 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for source, target in PAIRS:
        r = triggers[(triggers.source == source) & (triggers.target == target)].iloc[0]
        lines.append(f"| {name(source,target)} | {int(r.source_stage_cut)} | {r.tau_source:.3f} | "
                     f"{int(r.trigger_cut)} | {r.max_factor:.3f} | {int(r.n_changed_cuts)} |")
    lines.extend(["", "六方向 t0 均在 cut 210 或之后。触发 cut 的 F=1，cut 1–32 和所有触发前 cut 均保持原预测，所有因子有限且处于 [1,1.5]。源刀拟合趋势在 `source_trends/`，逐 cut b/s/d/确认计数/τ_source/t0/w/F 在 `gate_profiles/` 与 `unlabeled_predictions/`。", "",
                  "## 六方向 full_1_315 RMSE：五 seed 均值 ± 样本标准差", "",
                  "| 方向 | 基模型 | 原基线 | 原乘法 OOR-PGA | early-reference persistent OOR-PGA |",
                  "|---|---|---:|---:|---:|"])
    for source, target in PAIRS:
        for method in METHODS:
            vals = []
            for variant in VARIANTS:
                r = metrics[(metrics.source == source) & (metrics.target == target) &
                            (metrics.base_method == method) & (metrics.variant == variant) &
                            (metrics.segment == "full")].iloc[0]
                vals.append(pm(r, "RMSE"))
            lines.append(f"| {name(source,target)} | {method} | " + " | ".join(vals) + " |")
    lines.extend(["", "完整 R²、MAE、RMSE 和有符号偏差的逐 seed 及五 seed 汇总见 `per_seed_metrics_full_1_315.csv`、`five_seed_summary_full_1_315.csv`。", "",
                  "## 新门控相对原基线的同 seed 配对 ΔRMSE（负值为改善）", "",
                  "| 方向 | 基模型 | 全程 ΔRMSE | 改善 seed | 早期 ΔRMSE | 中期 ΔRMSE | 晚期 ΔRMSE |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for source, target in PAIRS:
        for method in METHODS:
            chosen = {}
            for segment in ("full", "early", "middle", "late"):
                chosen[segment] = paired[(paired.source == source) & (paired.target == target) &
                                         (paired.base_method == method) &
                                         (paired.variant == "early_reference_persistent_oor_pga") &
                                         (paired.segment == segment)].iloc[0]
            lines.append(f"| {name(source,target)} | {method} | {pm(chosen['full'],'delta_RMSE')} | "
                         f"{int(chosen['full'].improved_seed_count_RMSE)}/5 | "
                         f"{pm(chosen['early'],'delta_RMSE')} ({int(chosen['early'].improved_seed_count_RMSE)}/5) | "
                         f"{pm(chosen['middle'],'delta_RMSE')} ({int(chosen['middle'].improved_seed_count_RMSE)}/5) | "
                         f"{pm(chosen['late'],'delta_RMSE')} ({int(chosen['late'].improved_seed_count_RMSE)}/5) |")
    lines.extend(["", "同 seed 的原乘法式 OOR-PGA 配对差值也在 `paired_seed_delta_full_1_315.csv` 和 `paired_five_seed_delta_full_1_315.csv`；配对 ΔMAE、Δ偏差及各阶段改善 seed 数均保存于其中。早中期所有新门控 Δ 均为精确 0：不是误差抵消，而是因子仍为 1。", "",
                  "## 关键方向与局限", ""])
    for method in METHODS:
        values = []
        for variant in VARIANTS:
            r = metrics[(metrics.source == "c1") & (metrics.target == "c6") &
                        (metrics.base_method == method) & (metrics.variant == variant) &
                        (metrics.segment == "late")].iloc[0]
            values.append((r.RMSE_mean, r.signed_bias_mean))
        lines.append(f"- C1→C6 {method} 晚期 RMSE：基线 {values[0][0]:.2f}、原 OOR {values[1][0]:.2f}、新门控 {values[2][0]:.2f}；"
                     f"对应偏差 {values[0][1]:+.2f}、{values[1][1]:+.2f}、{values[2][1]:+.2f}。新门控保留部分晚期改善，但弱于原 OOR。")
    lines.extend(["- 原方案灾难性退步的 C1→C4、C4→C1、C6→C4 新 t0 分别延至 286、259、270，避免了 cut 2–3 乘法放大；然而 C4→C1 新门控仍使两种基模型全程 RMSE 上升，不能宣称全面改善。C6→C4 两种基模型则有小幅改善。", "",
                  "## 文件、哈希与可重算性", "",
                  "- `input_audit_lock.json`、`input_feature_audit_full_1_315.csv`、`input_checkpoint_prediction_audit_full_1_315.csv`：无标签输入、原预测与全部 60 个 checkpoint/预测来源和 SHA-256。",
                  "- `prediction_lock_before_target_labels.json`、`unlabeled_predictions/`：目标标签读取前冻结的 60 份 315-cut 原预测、原乘法 OOR、新门控预测及逐 cut 门控参数。",
                  "- `trigger_summary_full_1_315.csv`、`source_stage_fit_audit.json`、`source_trends/`、`gate_profiles/`：固定参数、拟合方法与原因、源刀单调趋势及六方向 r/d/阈值/t0 图。",
                  "- `evaluated_per_seed/`、`per_seed_metrics_full_1_315.csv`、`five_seed_summary_full_1_315.csv`、`paired_seed_delta_full_1_315.csv`、`paired_five_seed_delta_full_1_315.csv`：标签读取后的 60 份逐 cut 数据、指标及配对差值。六张三方案预测图在 `prediction_figures/`。",
                  "- `evaluation_manifest_full_1_315.json`：冻结文件哈希、实际运行命令、标签来源及其 SHA-256。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
