"""Export scope-specific tables and narrative from the completed stage-error audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/stage_error_analysis_zscore_20260925")
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))
METHODS = ("source_only", "daregram")
SEGMENTS = ("early", "middle", "late")


def read(name: str) -> list[dict]:
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name: str, rows: list[dict]) -> None:
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_label(source: str, target: str) -> str:
    return f"{source.upper()}→{target.upper()}"


def number(row: dict, key: str, digits: int = 2) -> str:
    return f"{float(row[key]):.{digits}f}" if row[key] != "" else "不可计算"


def change(row: dict, key: str, digits: int = 2) -> str:
    return f"{float(row[key]):+.{digits}f}" if row[key] != "" else "不可计算"


def export_bias_counts() -> None:
    rows = []
    for source, target in PAIRS:
        for method in METHODS:
            path = ROOT / "per_cut" / f"{source}_to_{target}_{method}_cuts_1_315.csv"
            with path.open(newline="", encoding="utf-8") as f:
                cuts = list(csv.DictReader(f))
            if len(cuts) != 315:
                raise ValueError(path)
            early = cuts[:105]
            late = cuts[210:]
            rows.append({"source": source, "target": target, "method": method,
                         "scope": "full_cuts_1_315", "early_cuts_1_105": 105,
                         "early_positive_five_seed_mean_error_cuts": sum(
                             float(r["five_seed_mean_signed_error"]) > 0 for r in early),
                         "late_cuts_211_315": 105,
                         "late_negative_five_seed_mean_error_cuts": sum(
                             float(r["five_seed_mean_signed_error"]) < 0 for r in late)})
    write("cut_bias_sign_counts_full_cuts_1_315.csv", rows)


def main() -> None:
    report_path = ROOT / "README.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    export_bias_counts()
    names = tuple(
        f"{stem}_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv"
        for stem in ("per_seed_stage_metrics", "paired_seed_stage_metrics",
                     "stage_summary_by_method", "paired_stage_summary",
                     "top_squared_error_cuts_per_seed"))
    all_rows = {name: read(name) for name in names}
    for name in names:
        stem = name.removesuffix("_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv")
        for scope, suffix in (("full_1_315", "full_cuts_1_315"),
                              ("suffix_95_315", "c6_suffix_cuts_95_315")):
            selected = [r for r in all_rows[name] if r["scope"] == scope]
            if selected:
                write(f"{stem}_{suffix}.csv", selected)
    extrap = read("per_seed_extrapolation_metrics_full_cuts_1_315.csv")
    extrap_summary = []
    for source, target in PAIRS:
        for method in METHODS:
            for group in ("at_or_below_source_max", "above_source_max"):
                selected = [r for r in extrap if (r["source"], r["target"], r["method"], r["target_group"]) ==
                            (source, target, method, group)]
                values = {metric: np.asarray([float(r[metric]) for r in selected if r[metric] != ""])
                          for metric in ("MAE", "RMSE", "signed_bias")}
                row = {"source": source, "target": target, "method": method,
                       "scope": "full_cuts_1_315", "group": group,
                       "source_training_label_max": selected[0]["source_training_label_max"],
                       "n_target_cuts": selected[0]["n"], "n_seeds": 5}
                for metric, vector in values.items():
                    row[f"{metric}_mean"] = float(vector.mean()) if len(vector) else ""
                    row[f"{metric}_sd"] = float(vector.std(ddof=1)) if len(vector) > 1 else ""
                extrap_summary.append(row)
    write("extrapolation_summary_full_cuts_1_315.csv", extrap_summary)
    top_rows = all_rows["top_squared_error_cuts_per_seed_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv"]
    top_summary = []
    for source, target in PAIRS:
        for method in METHODS:
            selected = [r for r in top_rows if (r["source"], r["target"], r["method"],
                         r["scope"], r["rank"]) == (source, target, method, "full_1_315", "1")]
            v5 = np.asarray([float(r["top_5_share"]) for r in selected])
            v10 = np.asarray([float(r["top_10_share"]) for r in selected])
            top_summary.append({"source": source, "target": target, "method": method,
                                "scope": "full_cuts_1_315", "n_seeds": 5,
                                "top_5_squared_error_share_mean": float(v5.mean()),
                                "top_5_squared_error_share_sd": float(v5.std(ddof=1)),
                                "top_10_squared_error_share_mean": float(v10.mean()),
                                "top_10_squared_error_share_sd": float(v10.std(ddof=1))})
    write("top_error_share_summary_full_cuts_1_315.csv", top_summary)

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    labels = manifest["label_sources"]
    full_pair = {(r["source"], r["target"], r["segment"]): r
                 for r in all_rows["paired_stage_summary_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv"] if r["scope"] == "full_1_315"}
    suffix_pair = {(r["source"], r["target"], r["segment"]): r
                   for r in all_rows["paired_stage_summary_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv"] if r["scope"] == "suffix_95_315"}
    full_methods = {(r["source"], r["target"], r["method"], r["segment"]): r
                    for r in all_rows["stage_summary_by_method_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv"] if r["scope"] == "full_1_315"}
    suffix_methods = {(r["source"], r["target"], r["method"], r["segment"]): r
                      for r in all_rows["stage_summary_by_method_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv"] if r["scope"] == "suffix_95_315"}
    ext_index = {(r["source"], r["target"], r["method"], r["group"]): r for r in extrap_summary}
    top_index = {(r["source"], r["target"], r["method"]): r for r in top_summary}
    lines = ["# PHM2010 Z-score 主实验：逐 cut 与分段误差分析", "",
             "## 标签、checkpoint 与预测来源", "",
             "原始标签来自 `E:/QLP/source/source_mill/{c1,c4,c6}_wear.csv` 的 `cut,flute_1,flute_2,flute_3`，按训练代码定义取三刃均值并转为 float32。逐刀检查 cut 编号精确为 1–315，无重复、无缺失、无非有限三刃值。逐 cut 可用性和真实值见 `label_availability_cuts_1_315.csv`。", "",
             "| 刀具 | cut 1–315 有真实标签 | 缺失 cut | 原始磨损 CSV SHA-256 |",
             "|---|---:|---|---|"]
    for item in labels:
        lines.append(f"| {item['tool'].upper()} | {item['cuts_1_315_with_label']}/315 | "
                     f"{item['missing_cuts'] or '无'} | `{item['sha256']}` |")
    lines += ["", "60 组预测均从已有文件取得，无需补推理：C1/C4 目标的 40 组使用 `five_seed_paired` 全程预测；C6 目标的 20 组使用 `norm_comparison_20260925/zscore` 对相同最终 checkpoint 的全程重新评估预测。逐组校验 checkpoint SHA-256、源域 Z-score 参数、cut 编号及预测文件中的真实值与原始磨损 CSV 一致。具体路径和哈希见 `checkpoint_prediction_provenance.csv`。目标无标签适应方案和 checkpoint 均未改动。", "",
             "统一保存六方向目标 cut 1–315 的逐 cut 预测。C1/C4 使用 1–315 全程指标；目标 C6 同时计算 1–315 全程和原主实验 95–315 后段指标。两套指标均从同一份 C6 全程预测按 cut 精确筛选，因此后段重算值可与旧 95–315 文件因推理分批数值差异略有不同；最大同 cut 预测绝对差为 0.00355，未发生 checkpoint 替换。所有分段先算每个 seed，再求五 seed 均值与样本标准差（ddof=1）；配对 Δ=DARE-GRAM−source-only。R² 只在至少两个有效标签且该段标签离差平方和大于 1e−12 时计算，本次 320 个逐 seed 段全部可定义。", "",
             "## 全程 cut 1–315：固定分段", "",
             "早期 1–105、中期 106–210、晚期 211–315；每段 n=105。下表是**配对 Δ**的五 seed 均值±样本标准差；负 ΔMAE/ΔRMSE 表示 DARE-GRAM 误差较小。两方法各自的 MAE、RMSE、有符号偏差和 R² 均值±标准差见 `stage_summary_by_method_full_cuts_1_315.csv`，逐 seed 原值见 `per_seed_stage_metrics_full_cuts_1_315.csv`。", "",
             "| 方向 | 阶段（cut） | ΔMAE | ΔRMSE | Δ有符号偏差 |",
             "|---|---|---:|---:|---:|"]
    for source, target in PAIRS:
        for segment in SEGMENTS:
            r = full_pair[source, target, segment]
            lines.append(f"| {pair_label(source, target)} | {segment} {r['cut_first']}–{r['cut_last']} | "
                         f"{change(r, 'delta_MAE_mean')}±{number(r, 'delta_MAE_sd')} | "
                         f"{change(r, 'delta_RMSE_mean')}±{number(r, 'delta_RMSE_sd')} | "
                         f"{change(r, 'delta_signed_bias_mean')}±{number(r, 'delta_signed_bias_sd')} |")
    lines += ["", "## C6 后段 cut 95–315：独立固定分段", "",
              "早期 95–168（n=74）、中期 169–241（n=73）、晚期 242–315（n=74）。这是原主实验目标 C6 的评估范围，不能替代或混入上述全程分段。文件名含 `c6_suffix_cuts_95_315`。", "",
              "| 方向 | 阶段（cut） | ΔMAE | ΔRMSE | Δ有符号偏差 |",
              "|---|---|---:|---:|---:|"]
    for source, target in PAIRS:
        if target != "c6":
            continue
        for segment in SEGMENTS:
            r = suffix_pair[source, target, segment]
            lines.append(f"| {pair_label(source, target)} | {segment} {r['cut_first']}–{r['cut_last']} | "
                         f"{change(r, 'delta_MAE_mean')}±{number(r, 'delta_MAE_sd')} | "
                         f"{change(r, 'delta_RMSE_mean')}±{number(r, 'delta_RMSE_sd')} | "
                         f"{change(r, 'delta_signed_bias_mean')}±{number(r, 'delta_signed_bias_sd')} |")
    lines += ["", "## C6 全程与后段整体指标对照", "",
              "| 方向 | 方法 | 全程 1–315 RMSE | 后段 95–315 RMSE |",
              "|---|---|---:|---:|"]
    for source, target in PAIRS:
        if target != "c6":
            continue
        for method in METHODS:
            full = full_methods[source, target, method, "all"]
            suffix = suffix_methods[source, target, method, "all"]
            lines.append(f"| {pair_label(source, target)} | {method} | "
                         f"{number(full, 'RMSE_mean', 4)}±{number(full, 'RMSE_sd', 4)} | "
                         f"{number(suffix, 'RMSE_mean', 4)}±{number(suffix, 'RMSE_sd', 4)} |")
    lines += ["", "C1→C6 的配对 ΔRMSE 为全程 −3.8317、后段 −5.2453；DARE-GRAM 的主要收益是晚期欠预测减轻。全程早、中期反而退步，使全程收益小于后段。C4→C6 的配对 ΔRMSE 为全程 +1.5440、后段 +1.5700，两范围均为平均退步；在后段早、中、晚三段的平均 ΔRMSE 也均为正。", "",
              "## 高磨损外推：目标真实值超过源刀训练标签最大值", "",
              "这是固定 checkpoint 的事后诊断，未用目标标签调模型。所有比较使用目标 cut 1–315，并按同一源刀最大训练标签分组。", "",
              "| 方向 | 源刀最大标签 | 高于上限的目标 cut 数 | 方法 | 上限内 MAE | 上限外 MAE |",
              "|---|---:|---:|---|---:|---:|"]
    for source, target in PAIRS:
        for method in METHODS:
            below = ext_index[source, target, method, "at_or_below_source_max"]
            above = ext_index[source, target, method, "above_source_max"]
            lines.append(f"| {pair_label(source, target)} | {float(below['source_training_label_max']):.2f} | "
                         f"{above['n_target_cuts']} | {method} | "
                         f"{number(below, 'MAE_mean')} | {number(above, 'MAE_mean')} |")
    lines += ["", "C1→C6 的 71 个上限外 cut 上，source-only/DARE-GRAM MAE 分别为 45.98/35.25，远高于上限内的 5.70/9.31；C1→C4 的 33 个上限外 cut 也明显更难。C4→C6 有 21 个上限外 cut，source-only MAE 变化较小（15.18→15.91），DARE-GRAM 则增大（16.84→26.22）。以 C6 为源或 C4→C1 时，目标没有超过源刀上限的 cut，不能估计该条件下误差。上限外 cut 与磨损后期高度相关，此比较是相关性诊断，不代表单独的因果效应。", "",
              "## 误差形态与异常 cut", "",
              "全程 C1→C6 的晚期平均有符号偏差为 source-only −35.7、DARE-GRAM −27.2；两方法五 seed 平均预测在晚期 105/105 个 cut 均低估，DARE-GRAM 减轻幅度。C6→C4 的晚期偏差为 −33.0 对 +4.2，source-only 在 105/105 个晚期 cut 低估，改善集中在晚期。C6→C1 的 source-only 早、中、晚均低估，DARE-GRAM 对三段均有改善，但晚期平均偏差转为正值。C1→C4 的早期两方法平均高估约 +8.2/+9.7，五 seed 平均误差为正的 cut 分别为 96/105、97/105；C6→C4 的早期 DARE-GRAM 平均高估约 +11.1，105/105 个 cut 为正。计数见 `cut_bias_sign_counts_full_cuts_1_315.csv`。", "",
              "少数 cut 对平方误差有一定影响，但总体并非孤立异常点决定 RMSE：例如全程 C1→C6 的前五个最大平方误差 cut 平均贡献 13.3%（source-only）与 12.5%（DARE-GRAM）；C6→C4 为 18.6% 与 13.6%。C1→C4 的前五 cut 贡献较高，为 22.0% 与 19.1%，主要落在相邻的末段 cut 311–315；C1→C6、C6→C4 的最大误差也多出现在连续的末期 cut，而非单个离群 cut。逐 seed 前十 cut、误差和平方误差占比见 `top_squared_error_cuts_per_seed_full_cuts_1_315.csv`。", "",
              "## 交付文件与图", "",
              "- `per_cut/<source>_to_<target>_<method>_cuts_1_315.csv`：12 文件；每 cut 的真实值、五 seed 预测、五 seed 预测均值、逐 seed 有符号/绝对误差及均值误差。",
              "- `per_seed_stage_metrics_full_cuts_1_315.csv`、`paired_seed_stage_metrics_full_cuts_1_315.csv`、`stage_summary_by_method_full_cuts_1_315.csv`、`paired_stage_summary_full_cuts_1_315.csv`：六方向全程逐 seed、配对与汇总。",
              "- 同名 `..._c6_suffix_cuts_95_315.csv`：两个目标 C6 方向的原评估后段结果。",
              "- `per_seed_extrapolation_metrics_full_cuts_1_315.csv`、`extrapolation_summary_full_cuts_1_315.csv`：训练标签范围内外诊断。",
              "- `figures/<source>_to_<target>_full_1_315_curves_and_errors.png`：六方向全程真实值、五 seed 预测曲线、有符号和绝对误差；目标 C6 另有两个 `suffix_95_315` 图。各图使用相同的纵轴范围、域方法颜色和图例；灰色虚线标固定段界，C6 全程图的紫色点线标原评估起点 95。无平滑或插值。",
              "- `checkpoint_prediction_provenance.csv`、`manifest.json`：checkpoint/预测来源、哈希、原评估 cut 和本次筛选协议。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote scope-specific tables and {report_path}")


if __name__ == "__main__":
    main()
