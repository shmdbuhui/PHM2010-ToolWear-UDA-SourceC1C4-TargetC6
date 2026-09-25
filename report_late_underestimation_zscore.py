"""Create paired late-cut coverage and a reviewable post-hoc range audit report."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/late_underestimation_extrapolation_zscore_20260925")
STAGE = Path("artifacts/stage_error_analysis_zscore_20260925")
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))
METHODS = ("source_only", "daregram")


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def direction(source: str, target: str) -> str:
    return f"{source.upper()}→{target.upper()}"


def fmt(value: str | float, digits: int = 2) -> str:
    return "不适用" if value == "" else f"{float(value):.{digits}f}"


def main() -> None:
    report = ROOT / "README.md"
    if report.exists():
        raise FileExistsError(report)
    per_seed_late = []
    for source, target in PAIRS:
        a = read(STAGE / "per_cut" / f"{source}_to_{target}_source_only_cuts_1_315.csv")
        b = read(STAGE / "per_cut" / f"{source}_to_{target}_daregram_cuts_1_315.csv")
        for scope, lo, hi in (("full_cuts_1_315", 211, 315),
                              ("c6_suffix_cuts_95_315", 242, 315)) if target == "c6" else (("full_cuts_1_315", 211, 315),):
            for seed in range(42, 47):
                key = f"seed_{seed}_signed_error"
                source_error = np.asarray([float(row[key]) for row in a[lo - 1:hi]])
                dare_error = np.asarray([float(row[key]) for row in b[lo - 1:hi]])
                per_seed_late.append({"source": source, "target": target, "scope": scope,
                                      "seed": seed, "late_cut_first": lo, "late_cut_last": hi,
                                      "n_late_cuts": hi - lo + 1,
                                      "cuts_daregram_lower_absolute_error": int((np.abs(dare_error) < np.abs(source_error)).sum()),
                                      "cuts_daregram_lower_squared_error": int((dare_error ** 2 < source_error ** 2).sum()),
                                      "source_only_late_RMSE": float(np.sqrt(np.mean(source_error ** 2))),
                                      "daregram_late_RMSE": float(np.sqrt(np.mean(dare_error ** 2))),
                                      "paired_delta_late_RMSE": float(np.sqrt(np.mean(dare_error ** 2)) -
                                                                      np.sqrt(np.mean(source_error ** 2)))})
    for scope in ("full_cuts_1_315", "c6_suffix_cuts_95_315"):
        subset = [r for r in per_seed_late if r["scope"] == scope]
        write(ROOT / f"per_seed_late_improvement_coverage_{scope}.csv", subset)

    ranges = read(ROOT / "direction_label_ranges_full_cuts_1_315.csv")
    grouped_full = read(ROOT / "range_underestimation_summary_full_cuts_1_315.csv")
    grouped_suffix = read(ROOT / "range_underestimation_summary_c6_suffix_cuts_95_315.csv")
    paired_full = read(ROOT / "paired_range_underestimation_summary_full_cuts_1_315.csv")
    maxima = read(ROOT / "per_seed_maxima_full_cuts_1_315.csv")
    top = read(ROOT / "top_10_cuts_by_pooled_squared_error_full_cuts_1_315.csv")
    shares = read(ROOT / "pooled_top_error_shares_full_cuts_1_315.csv")
    coverage = read(ROOT / "late_improvement_coverage_full_cuts_1_315.csv")
    reconciliation = read(ROOT / "rmse_reconciliation_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv")
    gr = {(r["source"], r["target"], r["method"], r["segment"], r["range_group"]): r
          for r in grouped_full}
    gr_suffix = {(r["source"], r["target"], r["method"], r["segment"], r["range_group"]): r
                 for r in grouped_suffix}
    cov = {(r["source"], r["target"]): r for r in coverage}
    share = {(r["source"], r["target"], r["method"]): r for r in shares}
    lines = ["# PHM2010 Z-score：晚期低估与源域标签范围的事后核查", "",
             "## 数据对应关系与核验", "",
             "仅使用现有 source-only / DARE-GRAM 最终 checkpoint、逐 cut 预测及原始磨损标签。六方向、seed 42–46、两方法共 60 组，逐组核对方向、seed、方法、checkpoint SHA-256、预测文件 SHA-256、配置、cut 1–315 与目标原始标签；逐 cut 预测和原预测文件逐值一致。源刀训练标签来自配置中 `source_wear_sha256` 指向的 `c1_wear.csv`、`c4_wear.csv` 或 `c6_wear.csv`，按原训练代码取三刃均值并转 float32。三把刀 cut 1–315 全有真实标签，未插值、补标签或改预测。路径与哈希见 `verified_provenance_full_cuts_1_315.csv`。目标真实磨损是否超过源域最大标签仅用于本次事后分组，未用于模型、checkpoint 或无标签适应。", "",
             "目标为 C1/C4 时只用全程 cut 1–315；目标为 C6 时分别保留全程 cut 1–315 和原协议后段 cut 95–315，两套结果和文件名分开。全程阶段为 1–105、106–210、211–315；后段阶段为 95–168、169–241、242–315。每 seed 先算指标，再取五 seed 均值±样本标准差（ddof=1）；Δ 一律为同 seed DARE-GRAM 减 source-only。若范围×阶段没有样本，CSV 写 `status=not_applicable_no_cuts`、数值留空，表内标“不适用”，不填零。连续低估区间要求原始 cut 编号连续，遇到不属于分组的 cut 即中断。", "",
             "## 训练标签上限与首次越界", "",
             "| 方向 | 源域训练标签最大值 | 目标真实最大值 | 越界 cut 数 | 首次越界 cut |",
             "|---|---:|---:|---:|---:|"]
    for r in ranges:
        lines.append(f"| {direction(r['source'], r['target'])} | {float(r['source_training_label_max']):.2f} | "
                     f"{float(r['target_true_max']):.2f} | {r['target_cuts_above_source_max']} | "
                     f"{r['first_target_cut_above_source_max'] if r['first_target_cut_above_source_max'] != 'not_applicable' else '不适用'} |")
    lines += ["", "`per_cut_range_flags_full_cuts_1_315.csv` 为六方向每个真实 cut 给出精确越界标记。", "",
              "## 全程 cut 1–315：范围内、范围外与低估", "",
              "| 方向 | 方法 | 分组 | n/cut | MAE | RMSE | 平均有符号误差 | 低估 cut 数（五 seed 均值） |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        for method in METHODS:
            for group in ("within_source_max", "above_source_max"):
                r = gr[source, target, method, "all", group]
                lines.append(f"| {direction(source, target)} | {method} | {group} | {r['n_per_seed']} | "
                             f"{fmt(r['MAE_mean'])} | {fmt(r['RMSE_mean'])} | "
                             f"{fmt(r['signed_bias_mean'])} | {fmt(r['under_count_mean'], 1)} |")
    lines += ["", "完整五 seed 标准差、逐 seed MAE/RMSE/偏差、低估数和最长连续区间，及范围×阶段细分均在 `per_seed_range_underestimation_full_cuts_1_315.csv`、`range_underestimation_summary_full_cuts_1_315.csv` 和相应 `paired_...` 文件；C6 原协议后段使用同名 `c6_suffix_cuts_95_315.csv`，绝不与全程合并。", "",
              "## 晚期改善覆盖范围", "",
              "下表按五个 seed 的原始平方误差逐 cut 相加；“改善 cut”指该 cut 的 DARE-GRAM 五 seed 合计平方误差小于 source-only。前十 cut 占比是**正向平方误差减少量**中的份额，不是拿五 seed 平均预测计算的 RMSE。逐 seed 改善 cut 数见 `per_seed_late_improvement_coverage_full_cuts_1_315.csv`。", "",
              "| 方向 | 晚期改善 cut 数 | 晚期净 SSE 减少 | 最大 10 个正向改进 cut 占正向减少量 | 逐 seed 改善 cut 数 |",
              "|---|---:|---:|---:|---|"]
    for source, target in (("c1", "c6"), ("c6", "c4"), ("c6", "c1")):
        r = cov[source, target]
        counts = [x["cuts_daregram_lower_squared_error"] for x in per_seed_late
                  if (x["source"], x["target"], x["scope"]) == (source, target, "full_cuts_1_315")]
        lines.append(f"| {direction(source, target)} | {r['cuts_daregram_lower_pooled_squared_error']}/{r['n_late_cuts']} | "
                     f"{float(r['net_late_SSE_reduction']):,.0f} | "
                     f"{100 * float(r['top_10_cuts_share_of_positive_SSE_reduction']):.1f}% | "
                     f"{', '.join(map(str, counts))} /105 |")
    lines += ["", "C1→C6 的改善覆盖全程晚期 104/105 cut，原协议后段晚期 242–315 为 74/74 cut；各 seed 的全程晚期改善 cut 数为 105、93、83、103、104。前十 cut 仅占正向 SSE 减少量的 24.5%，收益广泛分布。C6→C4 的改善覆盖 79/105 cut，各 seed 为 78、68、84、75、84；前十 cut 占 37.2%，末期大误差 cut 有较明显贡献，但不是少数 cut 独占收益。C6→C1 也覆盖 102/105 cut。后段 C6 的独立覆盖结果见 `late_improvement_coverage_c6_suffix_cuts_95_315.csv`。", "",
              "## C1→C6：各 seed 晚期低估", "",
              "| seed | source-only 低估 cut / 最长连续 | DARE-GRAM 低估 cut / 最长连续 | source-only 偏差 | DARE-GRAM 偏差 |",
              "|---:|---:|---:|---:|---:|"]
    for seed in range(42, 47):
        s = next(r for r in read(ROOT / "per_seed_range_underestimation_full_cuts_1_315.csv")
                 if (r["source"], r["target"], int(r["seed"]), r["method"], r["segment"], r["range_group"]) ==
                 ("c1", "c6", seed, "source_only", "late", "all_labels"))
        d = next(r for r in read(ROOT / "per_seed_range_underestimation_full_cuts_1_315.csv")
                 if (r["source"], r["target"], int(r["seed"]), r["method"], r["segment"], r["range_group"]) ==
                 ("c1", "c6", seed, "daregram", "late", "all_labels"))
        lines.append(f"| {seed} | {s['under_count']}/105；{s['longest_under_run']} "
                     f"({s['longest_under_first_cut']}–{s['longest_under_last_cut']}) | "
                     f"{d['under_count']}/105；{d['longest_under_run']} "
                     f"({d['longest_under_first_cut']}–{d['longest_under_last_cut']}) | "
                     f"{float(s['signed_bias']):+.2f} | {float(d['signed_bias']):+.2f} |")
    late_s = gr["c1", "c6", "source_only", "late", "all_labels"]
    late_d = gr["c1", "c6", "daregram", "late", "all_labels"]
    lines += ["", f"source-only 五个 seed 在 211–315 均连续低估 105/105 cut；DARE-GRAM 的 seed 42 低估 103/105，最长连续 214–315，其余四个 seed 均连续低估 105/105。平均晚期偏差从 {float(late_s['signed_bias_mean']):.2f} 变为 {float(late_d['signed_bias_mean']):.2f}，减少低估约 {float(late_d['signed_bias_mean']) - float(late_s['signed_bias_mean']):+.2f}（预测−真实的变化）。", "",
              "## 平方误差最高的 cut（五 seed 原始 SSE 合计）", "",
              "每方法先对每个 cut 将五个 seed 的平方误差相加，再除以该方法在所标范围内的五 seed 总 SSE；未先平均预测。逐 seed 的前 1/5/10 cut 占比另见 `per_seed_top_error_shares_full_cuts_1_315.csv`。", "",
              "| 方向 | 方法 | 前 10 cut（从高到低） | 前 1 / 5 / 10 cut 占全程 SSE |",
              "|---|---|---|---:|"]
    for source, target in (("c1", "c6"), ("c6", "c4"), ("c6", "c1")):
        for method in METHODS:
            ordered = sorted((r for r in top if (r["source"], r["target"], r["method"]) ==
                              (source, target, method)), key=lambda r: int(r["rank"]))
            x = next(r for r in shares if (r["source"], r["target"], r["method"]) ==
                     (source, target, method))
            lines.append(f"| {direction(source, target)} | {method} | "
                         f"{', '.join(r['cut_index'] for r in ordered)} | "
                         f"{100 * float(x['top_1_share']):.1f}% / "
                         f"{100 * float(x['top_5_share']):.1f}% / "
                         f"{100 * float(x['top_10_share']):.1f}% |")
    lines += ["", "C1→C6 和 C6→C4 的大误差 cut 多是相邻末期 cut 306–315，属于持续尾端偏差。C6→C1 的 DARE-GRAM 前十 cut 分布较散。`top_10_cuts_by_pooled_squared_error_full_cuts_1_315.csv` 保留每个 cut 的 SSE 和占比；C6 后段有独立的 `..._c6_suffix_cuts_95_315.csv`。", "",
              "## 输出是否受源域标签上限约束", "",
              "`per_seed_maxima_full_cuts_1_315.csv` 对全部 60 组列出源域标签最大值、目标真实最大值、预测最大值及其 cut、预测超过源域上限的 cut 数。C1→C6 的 source-only 五 seed 预测最大值约 148.6–157.0，均未超过 C1 上限 165.17；DARE-GRAM seed 42/43 分别有 24/32 个预测 cut 超过上限，其余三 seed 没有。C6→C4 的目标真实最大值 203.08，本来就低于 C6 源域上限 215.94，虽然两方法全部 seed 预测均未超过源域上限，source-only 仍严重晚期低估。C6→C1 也无目标越界，DARE-GRAM 仍明显改善。故“预测未越过源域上限”与“目标末期低估”必须分开陈述；部分模型确实能预测出源域上限外数值，现有数据不支持通用的硬输出截断。", "",
              "## 越界关联与结论边界", "",
              "C1→C6 首次越界为 cut 245：越界 71 cut 的 source-only / DARE-GRAM RMSE 为 48.20 / 37.29，范围内 244 cut 为 7.80 / 10.73，越界后误差明显增加。但晚期 211–244 仍在源域范围内，source-only 34/34 cut 低估，RMSE 15.34；DARE-GRAM 平均低估约 33.6/34 cut，RMSE 11.41。C6→C4 和 C6→C1 根本没有目标越界，仍有晚期低估或显著的 DARE-GRAM 收益。C4→C6 的 source-only 越界内外 MAE 仅 15.18→15.91，DARE-GRAM 则 16.84→26.22。越界和大误差在 C1→C6 有强关联，但不能据此断言“标签范围外推是六方向的主要瓶颈”；磨损阶段、刀具域差异与预测形状同时变化，事后分组不是因果证明。", "",
              "## 全程与 C6 原协议 RMSE 核对", "",
              "`rmse_reconciliation_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv` 共 80 行：本次逐 cut 重算与上一份分段分析的全程和后段 RMSE 逐项一致。60 组全程 RMSE 与对应既有全程指标的最大差为机器精度；20 组 C6 后段与更早 `five_seed_paired` 原预测指标的最大差约 0.000379。原因是本次从已有全程预测按 cut 95–315 筛选，旧后段预测是从 cut 95 开始分批推理；checkpoint 哈希相同。同一文件同时保留两来源的原值、重算值和差值。C6 两范围从未混合汇总。", "",
              "## 文件与图", "",
              "- `per_seed_range_underestimation_full_cuts_1_315.csv` 与 `..._c6_suffix_cuts_95_315.csv`：逐 seed、阶段、范围组的 n/MAE/RMSE/有符号偏差/低估数量/最长连续低估起止；无样本为不适用。",
              "- `range_underestimation_summary_...csv` 与 `paired_range_underestimation_summary_...csv`：五 seed 均值±样本标准差及同 seed DARE-GRAM−source-only 差值，按评估 cut 分文件。",
              "- `per_seed_maxima_full_cuts_1_315.csv`、`direction_label_ranges_full_cuts_1_315.csv`、`per_cut_range_flags_full_cuts_1_315.csv`：预测输出范围、源域标签上限、首次目标越界和逐 cut 标记。",
              "- `top_10_cuts_by_pooled_squared_error_...csv`、`pooled_top_error_shares_...csv`、`per_seed_top_error_shares_...csv`、`late_improvement_coverage_...csv`：原始逐 seed SSE 贡献和改善覆盖。",
              "- `figures/`：C1→C6 全程与 C6 后段、C6→C4 全程、C6→C1 全程共四张图。每图按 cut 原序绘制真实磨损、两方法的五 seed 预测均值与各 seed 预测点、源域标签上限、首次越界线（若存在）、固定段界、平均有符号误差及每 cut 占总 SSE 比例；不平滑或插值。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
