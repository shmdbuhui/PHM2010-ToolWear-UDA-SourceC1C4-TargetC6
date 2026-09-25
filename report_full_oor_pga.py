"""Write reviewable reports for frozen full-cut baseline and STFT-48 OOR-PGA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("artifacts/full_1_315_baseline_zscore_20260925")
OOR = Path("artifacts/oor_pga_full_1_315_zscore_20260925")
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))


def fmt(v: float) -> str:
    return f"{v:.2f}"


def pm(row: pd.Series, metric: str) -> str:
    return f"{fmt(row[metric + '_mean'])} ± {fmt(row[metric + '_sd'])}"


def label(s: str, t: str) -> str:
    return f"{s.upper()}→{t.upper()}"


def main() -> None:
    base_summary = pd.read_csv(BASE / "five_seed_summary_full_1_315.csv")
    results = pd.read_csv(OOR / "five_seed_summary_full_1_315.csv")
    paired = pd.read_csv(OOR / "paired_five_seed_delta_full_1_315.csv")
    triggers = pd.read_csv(OOR / "triggers_and_factors_full_1_315.csv")
    audit = pd.read_csv(BASE / "checkpoint_audit_full_1_315.csv")
    logs = pd.read_csv(BASE / "training_log_epoch_audit_full_1_315.csv")
    if len(audit) != 60 or len(logs) != 60 or len(results) != 96 or len(paired) != 48:
        raise ValueError("Missing completed experiment rows")
    if not ((logs[logs.method == "daregram"].actual_min_unique_target_cuts == 315).all() and
            (logs[logs.method == "source_only"].actual_max_unique_target_cuts == 0).all()):
        raise ValueError("Training log protocol check failed")

    b = ["# PHM2010 Z-score 单源迁移：统一 full_1_315 基线", "",
         "## 协议与来源", "",
         "六方向、seed 42–46、source-only / DARE-GRAM 共 60 个现有最终 checkpoint。逐组核对 checkpoint 与训练配置、源刀 STFT 缓存和源刀拟合的六通道输入 Z-score；DARE-GRAM 使用目标刀 1–315 的全部无标签 STFT，source-only 训练不接收目标输入，训练期间不读取目标磨损。每组 50 epoch、batch 63、Adam 0.001、ResNet18 + Linear(512,1)、末轮 checkpoint。训练日志中每个 DARE-GRAM epoch 均记录 `target_unique=315`，source-only 均为 0。无需重训。", "",
         "目标刀 C1/C4/C6 均按原始 cut 1–315 精确匹配磨损标签，主要指标统一为 **full_1_315**。C6 的 95–315 旧评估口径保留在原 `artifacts/five_seed_paired` 目录，没有进入下表。C6 全程预测来自同一 checkpoint 的既有 Z-score 全程复评，checkpoint 哈希已核对；所有 60 组预测均有 315 个无重复且有限的值。", "",
         "## 六方向基线：full_1_315，五 seed 均值 ± 样本标准差", "",
         "| 方向 | 方法 | R² | MAE | RMSE | MAPE (%) | 有符号偏差 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for s, t in PAIRS:
        for method in ("source_only", "daregram"):
            row = base_summary[(base_summary.source == s) & (base_summary.target == t) &
                               (base_summary.method == method)].iloc[0]
            b.append(f"| {label(s,t)} | {method} | {pm(row,'R2')} | {pm(row,'MAE')} | "
                     f"{pm(row,'RMSE')} | {pm(row,'MAPE_percent')} | {pm(row,'signed_bias')} |")
    b.extend(["", "逐 seed 指标见 `per_seed_metrics_full_1_315.csv`，可从 `per_seed/` 中逐 cut CSV 独立重算。60 组全程 RMSE 与先前分段分析的对应值最大差 7.11×10⁻¹⁵。`checkpoint_audit_full_1_315.csv` 提供配置、checkpoint、预测、原始标签与 STFT 缓存的路径和 SHA-256；`training_log_epoch_audit_full_1_315.csv` 保存逐组 epoch 使用量核查。", "",
              "运行命令：`python run_full_1_315_baseline_protocol.py`。基线在 OOR-PGA 运行前以 `protocol_lock_full_1_315.json` 锁定。旧版 C6 95–315 只作历史结果，不作为当前主指标。", ""])
    baseline_report = BASE / "README.md"
    if baseline_report.exists():
        raise FileExistsError(baseline_report)
    baseline_report.write_text("\n".join(b), encoding="utf-8")

    o = ["# PHM2010 full_1_315：乘法式 OOR-PGA（STFT-48 移植对照）", "",
         "## 运行顺序与实现锁定", "",
         "先完成并锁定 [统一全程基线](../full_1_315_baseline_zscore_20260925/README.md)，再运行 `python run_oor_pga_full_1_315.py prepare`，把 30 组修正前后逐 cut 预测、OOR 分数、gate、指数和因子保存到 `unlabeled_predictions/` 与 `oor_scores/`。此阶段仅读取源刀训练磨损标签；对原预测文件只读取 cut 和预测两列，没有读取目标标签、误差或指标。`prediction_lock_before_target_labels.json` 保存此时的文件哈希。随后单独运行 `python run_oor_pga_full_1_315.py evaluate`，才打开目标刀磨损文件并计算指标、出图。基模型、原预测和旧评估目录未改动。", "",
         "本次是 **48 维 STFT 越界特征的移植实现**：6 个传感通道各取 8 个连续频带，每带为调整到 128 行后的 16 个频率行，沿频率行和时间轴取均值。原 STFT 为 fs=50 kHz、nperseg=256，`data_sampling.py` 中 fmax 裁剪被注释，因此频轴覆盖约 0–25 kHz；原始 129 个频率 bin 被线性调整至 128 行。源刀 315 个样本逐维拟合 Z-score，再以源刀逐维最小/最大值定义支持范围；目标刀每个 cut 的越界特征数除以 48 得 OOR rate。`gate=clip((rate−3/48)/(6/48−3/48),0,1)`。TL=3/48、TH=6/48 是把 EEMD 的 3/43、6/43 按越界特征个数移植，**没有在 48 维上得到验证**；阈值不是由目标标签或结果选择。", "",
         "tau=(cut−1)/314。仅用本方向源刀 315 个训练标签，以 Theil–Sen 拟合 log(VB)=log(A)+m·log(tau)，tau=0 排除。第一个 tau>0 且 gate 完全打开的 cut 为 t_OOR（K=1）；触发当 cut 因子仍为 1，之后因子为 `(tau/t_OOR)^m`，直接乘原预测。未触发则全程保持原预测。本次六方向均触发。乘法因子在触发后不再由 gate 开关控制。这不是 OOR-PGA-like 累积加法，也没有 late-gated 探索列。", "",
         "本地参考实现为 `E:\\QLP\\EEMD-7.22\\run_oor_triggered_pga.py` 及其 OOR/PGA 依赖；其 43 个表格特征与本实验 STFT 不相同。草稿 PR #1 的 `run_resnet_oor_pga_postprocess.py` 在本地不存在，GitHub 连接失败，因此没有把无法核查的 PR 代码或参数当作已验证来源。", "",
         "## 触发与乘法因子（目标刀 cut 1–315）", "",
         "| 方向 | t_OOR cut | tau | 源刀 m | cut 315 因子 |",
         "|---|---:|---:|---:|---:|"]
    for s, t in PAIRS:
        r = triggers[(triggers.source == s) & (triggers.target == t)].iloc[0]
        o.append(f"| {label(s,t)} | {int(r.trigger_cut)} | {r.t_oor:.4f} | {r.m:.4f} | {r.factor_at_cut_315:.2f} |")
    o.extend(["", "cut 2–3 的早触发发生在 C1→C4、C4→C1、C6→C4；按预定 K=1 首次全开规则，之后持续乘法放大，结果可能严重失真。未因目标评估结果重新挑选阈值、持久长度或方向。", "",
              "## 四组主要结果：full_1_315，RMSE 五 seed 均值 ± 标准差", "",
              "| 方向 | source-only | source-only+OOR-PGA | DARE-GRAM | DARE-GRAM+OOR-PGA |",
              "|---|---:|---:|---:|---:|"])
    for s, t in PAIRS:
        vals = []
        for method in ("source_only", "source_only_oor_pga", "daregram", "daregram_oor_pga"):
            r = results[(results.source == s) & (results.target == t) &
                        (results.method == method) & (results.segment == "full")].iloc[0]
            vals.append(pm(r, "RMSE"))
        o.append(f"| {label(s,t)} | " + " | ".join(vals) + " |")
    o.extend(["", "完整 R²、MAE、RMSE、MAPE 和有符号偏差的逐 seed 与五 seed 均值±标准差，含早中晚三段，分别见 `per_seed_metrics_full_1_315.csv` 和 `five_seed_summary_full_1_315.csv`。以下 Δ 均是同 seed OOR-PGA 减对应未修正基线，负值为改善。", "",
              "## 全程配对 ΔRMSE：full_1_315", "",
              "| 方向 | 基线 | ΔRMSE 五 seed 均值 ± 标准差 | 改善 seed 数 |",
              "|---|---|---:|---:|"])
    for s, t in PAIRS:
        for method in ("source_only", "daregram"):
            r = paired[(paired.source == s) & (paired.target == t) &
                       (paired.baseline_method == method) & (paired.segment == "full")].iloc[0]
            o.append(f"| {label(s,t)} | {method} | {pm(r,'delta_RMSE')} | {int(r.improved_seed_count_RMSE)}/5 |")
    o.extend(["", "## 早中晚代价：配对 ΔRMSE（full_1_315 固定阶段）", "",
              "| 方向 | 基线 | 早期 1–105 | 中期 106–210 | 晚期 211–315 |",
              "|---|---|---:|---:|---:|"])
    for s, t in PAIRS:
        for method in ("source_only", "daregram"):
            values = []
            for segment in ("early", "middle", "late"):
                r = paired[(paired.source == s) & (paired.target == t) &
                           (paired.baseline_method == method) & (paired.segment == segment)].iloc[0]
                values.append(f"{pm(r,'delta_RMSE')} ({int(r.improved_seed_count_RMSE)}/5)")
            o.append(f"| {label(s,t)} | {method} | " + " | ".join(values) + " |")
    o.extend(["", "`paired_seed_delta_full_1_315.csv` 和 `paired_five_seed_delta_full_1_315.csv` 也给出各段配对 ΔMAE、Δ有符号偏差。", "",
              "## C1→C6 晚期与早中期", ""])
    for method in ("source_only", "daregram"):
        parts = []
        for segment in ("early", "middle", "late"):
            a = results[(results.source == "c1") & (results.target == "c6") &
                        (results.method == method) & (results.segment == segment)].iloc[0]
            b = results[(results.source == "c1") & (results.target == "c6") &
                        (results.method == method + "_oor_pga") & (results.segment == segment)].iloc[0]
            parts.append(f"{segment} RMSE {a.RMSE_mean:.2f}→{b.RMSE_mean:.2f}，偏差 {a.signed_bias_mean:.2f}→{b.signed_bias_mean:.2f}")
        o.append(f"- {method}：" + "；".join(parts) + "。")
    o.extend(["", "C1→C6 的触发 cut 为 161；早期预测完全不变，中期有小幅代价，晚期低估明显减轻。但其他方向并未得到一致收益：C1→C4、C4→C1、C6→C4 的极早触发产生远大于基线的误差；C4→C6 全程也退步。C6→C1 的 source-only 略改善，DARE-GRAM 略退步。没有按目标指标选择不同方向的参数或方法，这些数值仅评价固定的移植方案。", "",
              "## 文件与复现", "",
              "- `unlabeled_predictions/`：标签读取前保存的 30 组逐 cut 原预测与修正预测，均为 315 行；`prediction_lock_before_target_labels.json` 记录全部哈希。",
              "- `oor_scores/`：六方向逐 cut OOR 得分、gate、48 维越界 flag、触发与乘法因子。`source_48_feature_support.csv` 给出源域逐维范围；`source_pga_exponents.csv` 给出源刀拟合指数。",
              "- `evaluated_per_seed/`：读取真实标签后保存的 30 组四方法逐 cut 预测和有符号误差；`per_seed_metrics_full_1_315.csv`、`five_seed_summary_full_1_315.csv`、`paired_seed_delta_full_1_315.csv`、`paired_five_seed_delta_full_1_315.csv` 为指标。",
              "- `figures/`：六方向四方法五 seed 平均预测、逐 cut 平均误差以及 OOR 分数/gate 曲线，标题均标 full_1_315。",
              "- `evaluation_manifest_full_1_315.json`、基线审计 CSV 和两个 lock JSON 记录源数据、缓存、checkpoint、预测及标签的哈希与运行命令。所有指标可从逐 cut CSV 重算。", ""])
    report = OOR / "README.md"
    if report.exists():
        raise FileExistsError(report)
    report.write_text("\n".join(o), encoding="utf-8")
    print(f"Wrote {baseline_report} and {report}")


if __name__ == "__main__":
    main()
