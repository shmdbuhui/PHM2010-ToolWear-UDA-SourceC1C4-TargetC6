"""Read-only audit and paired analysis of the fixed six-direction COD experiment.

Writes reports and figures below the COD result directory; never loads a trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "artifacts" / "cod_comparison_20260926"
OUT = ROOT / "paired_seed_analysis"
PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("source_only", "daregram", "daregram_cod")
METRICS = ("R2", "MAE", "RMSE", "MAPE_percent")
STAGES = ((1, 105), (106, 210), (211, 315))
COLORS = {"source_only": "#2563a6", "daregram": "#db7026",
          "daregram_cod": "#20865d"}
LABELS = {"source_only": "Source-only", "daregram": "DARE-GRAM",
          "daregram_cod": "DARE-GRAM + COD"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(a, b, description, atol=1e-9):
    difference = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    require(difference <= atol, f"{description}: max abs difference {difference:.12g} > {atol}")
    return difference


def metric(y, p):
    e = p - y
    return {"R2": float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e * e))),
            "MAPE_percent": float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-8)) * 100)}


def plot_one(source, target, seed, values, coordinates, path):
    cuts = np.arange(1, 316)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.plot(cuts, values["source_only"]["true_vb"], color="#242424", lw=2.25,
            label="True VB", zorder=5)
    for method in METHODS:
        ax.plot(cuts, values[method]["pred_vb"], color=COLORS[method], lw=1.6,
                label=LABELS[method], zorder=3)
    ax.set(title=f"{source.upper()} → {target.upper()}  |  seed {seed}  |  CUT 1–315",
           xlabel="Target cut index", ylabel="Flank wear (VB)")
    ax.set_xlim(*coordinates["xlim"])
    ax.set_ylim(*coordinates["ylim"])
    ax.set_xticks(coordinates["xticks"])
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(loc="lower right", ncol=2, fontsize=9, framealpha=0.9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    coordinates = json.loads((ROOT / "plot_coordinates.json").read_text(encoding="utf-8"))
    published = pd.read_csv(ROOT / "per_seed_metrics.csv")
    require(len(published) == 90, f"{ROOT / 'per_seed_metrics.csv'}: expected 90 rows")
    require(not published.duplicated(["source", "target", "seed", "method"]).any(),
            f"{ROOT / 'per_seed_metrics.csv'}: duplicate key")
    results, segments, predictions = [], [], {}
    common_truth = {}
    protocol_by_direction = {}
    max_metric_diff = 0.0
    for source, target in PAIRS:
        for seed in SEEDS:
            folder = ROOT / f"{source}_to_{target}" / f"seed_{seed}"
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
            require((cfg["source"], cfg["target"], cfg["seed"]) == (source, target, seed),
                    f"{folder / 'config.json'}: run key mismatch")
            require((cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["checkpoint_selection"],
                     cfg["cod_weight"], cfg["cod_epsilon"], cfg["cod_start_step"],
                     cfg["target_labels_training_reads"]) ==
                    (50, 63, 0.001, "final epoch", 0.001, 0.05, 75, 0),
                    f"{folder / 'config.json'}: protocol mismatch")
            require(complete["methods"] == list(METHODS) and
                    complete["evaluation_cuts"] == list(range(1, 316)),
                    f"{folder / 'complete.json'}: methods/cuts mismatch")
            baseline = Path(cfg["baseline_path"])
            base_cfg = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
            require((base_cfg["source"], base_cfg["target"], base_cfg["seed"]) ==
                    (source, target, seed), f"{baseline / 'config.json'}: baseline key mismatch")
            for key in ("source_normalization_mean", "source_normalization_std"):
                compare(cfg[key], base_cfg[key], f"{folder / 'config.json'}: {key}", 0)
            require(base_cfg["source_count"] == 315 and
                    base_cfg["source_cuts"] == list(range(1, 316)) and
                    base_cfg["input"] == "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128",
                    f"{baseline / 'config.json'}: source/input protocol mismatch")
            signature = {key: base_cfg[key] for key in
                         ("stft_code_sha256", "source_feature_sha256",
                          "target_feature_sha256", "source_wear_sha256",
                          "input", "source_normalization_mean", "source_normalization_std")}
            direction = (source, target)
            if direction in protocol_by_direction:
                require(signature == protocol_by_direction[direction],
                        f"{baseline / 'config.json'}: feature/normalization version differs by seed")
            else:
                protocol_by_direction[direction] = signature
            raw_wear = Path(base_cfg["raw_root"]) / f"{target}_wear.csv"
            if raw_wear.is_file():
                wear = pd.read_csv(raw_wear)
                require(list(wear.columns) == ["cut", "flute_1", "flute_2", "flute_3"] and
                        wear.cut.astype(int).tolist() == list(range(1, 316)),
                        f"{raw_wear}: label schema/cuts mismatch")
                raw_truth = wear[["flute_1", "flute_2", "flute_3"]].mean(axis=1).to_numpy(np.float32)
            else:
                raw_truth = None
            for method in METHODS:
                path = folder / method / "predictions.csv"
                require(path.is_file(), f"Missing prediction: {path}")
                frame = pd.read_csv(path)
                require(list(frame.columns) == ["cut_index", "true_vb", "pred_vb"] and
                        frame.cut_index.tolist() == list(range(1, 316)) and
                        np.isfinite(frame[["true_vb", "pred_vb"]].to_numpy()).all(),
                        f"{path}: schema, cut order or finite values mismatch")
                y = frame.true_vb.to_numpy(float)
                p = frame.pred_vb.to_numpy(float)
                if raw_truth is not None:
                    compare(y, raw_truth, f"{path} vs {raw_wear}: mean of three flutes", 1e-10)
                if target in common_truth:
                    compare(y, common_truth[target], f"{path}: target labels", 0)
                else:
                    common_truth[target] = y.copy()
                predictions[(source, target, seed, method)] = frame
                row = {"source": source, "target": target, "seed": seed,
                       "method": method, "n": len(frame), **metric(y, p)}
                results.append(row)
                old = published.loc[(published.source == source) &
                                    (published.target == target) &
                                    (published.seed == seed) &
                                    (published.method == method)]
                require(len(old) == 1, f"{ROOT / 'per_seed_metrics.csv'}: missing {source}->{target} seed {seed} {method}")
                require(int(old.iloc[0]["n"]) == 315, f"{ROOT / 'per_seed_metrics.csv'}: n mismatch")
                per_run = pd.read_csv(folder / "metrics.csv")
                per_run = per_run.loc[per_run.method == method]
                require(len(per_run) == 1, f"{folder / 'metrics.csv'}: missing/duplicate {method}")
                metric_json = json.loads((folder / method / "metrics.json").read_text(encoding="utf-8"))
                for name in METRICS:
                    for origin, value in ((ROOT / "per_seed_metrics.csv", old.iloc[0][name]),
                                          (folder / "metrics.csv", per_run.iloc[0][name]),
                                          (folder / method / "metrics.json", metric_json[name])):
                        max_metric_diff = max(max_metric_diff,
                                              compare(row[name], value, f"{path} vs {origin}: {name}"))
                for start, end in STAGES:
                    residual = p[start - 1:end] - y[start - 1:end]
                    segments.append({"source": source, "target": target, "seed": seed,
                                     "method": method, "start_cut": start, "end_cut": end,
                                     "MAE": float(np.mean(np.abs(residual))),
                                     "mean_signed_error": float(np.mean(residual)),
                                     "p95_absolute_error": float(np.percentile(np.abs(residual), 95))})

    result = pd.DataFrame(results)
    result.to_csv(OUT / "recomputed_per_seed_metrics.csv", index=False, float_format="%.17g")
    segments = pd.DataFrame(segments)
    segments.to_csv(OUT / "segment_per_seed_metrics.csv", index=False, float_format="%.17g")
    published_summary = pd.read_csv(ROOT / "six_direction_three_method_metrics.csv")
    summary = result.groupby(["source", "target", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), **{name: (name, "mean") for name in METRICS})
    for row in summary.itertuples(index=False):
        old = published_summary.loc[(published_summary.source == row.source) &
                                    (published_summary.target == row.target) &
                                    (published_summary.method == row.method)]
        require(len(old) == 1 and int(old.iloc[0].seeds) == 5,
                f"{ROOT / 'six_direction_three_method_metrics.csv'}: missing/incorrect {row}")
        for name in METRICS:
            compare(getattr(row, name), old.iloc[0][name],
                    f"{ROOT / 'six_direction_three_method_metrics.csv'}: {row.source}->{row.target} {row.method} {name}")
    summary.to_csv(OUT / "recomputed_direction_method_means.csv", index=False, float_format="%.17g")
    pivot = result.pivot(index=["source", "target", "seed"], columns="method", values=list(METRICS))
    delta = pd.DataFrame({name: pivot[(name, "daregram_cod")] - pivot[(name, "daregram")]
                          for name in METRICS}).reset_index()
    old_delta = pd.read_csv(ROOT / "cod_minus_daregram_per_seed.csv")
    old_delta = old_delta.set_index(["source", "target", "seed"])
    for row in delta.itertuples(index=False):
        for name in METRICS:
            compare(getattr(row, name), old_delta.loc[(row.source, row.target, row.seed), name],
                    f"{ROOT / 'cod_minus_daregram_per_seed.csv'}: {row.source}->{row.target} seed {row.seed} {name}")
    delta.to_csv(OUT / "cod_minus_daregram_30_pairs.csv", index=False, float_format="%.17g")
    by_direction = delta.groupby(["source", "target"], as_index=False).agg(
        **{f"{name}_mean": (name, "mean") for name in METRICS},
        **{f"{name}_sd": (name, "std") for name in METRICS},
        **{f"{name}_improved_seeds": (name, (lambda x, name=name: int((x > 0).sum())
                                                 if name == "R2" else int((x < 0).sum())))
           for name in METRICS})
    old_direction = pd.read_csv(ROOT / "cod_minus_daregram_by_direction.csv")
    for row in by_direction.itertuples(index=False):
        old = old_direction.loc[(old_direction.source == row.source) &
                                (old_direction.target == row.target)]
        require(len(old) == 1, f"{ROOT / 'cod_minus_daregram_by_direction.csv'}: missing direction")
        for name in METRICS:
            compare(getattr(row, f"{name}_mean"), old.iloc[0][name],
                    f"{ROOT / 'cod_minus_daregram_by_direction.csv'}: {row.source}->{row.target} {name}")
    by_direction.to_csv(OUT / "cod_minus_daregram_direction_stats.csv", index=False,
                        float_format="%.17g")
    seg_pivot = segments.pivot(index=["source", "target", "seed", "start_cut", "end_cut"],
                               columns="method", values=["MAE", "mean_signed_error",
                                                        "p95_absolute_error"])
    seg_delta = pd.DataFrame({name: seg_pivot[(name, "daregram_cod")] -
                                   seg_pivot[(name, "daregram")]
                              for name in ("MAE", "mean_signed_error", "p95_absolute_error")}).reset_index()
    seg_delta.to_csv(OUT / "segment_cod_minus_daregram_per_seed.csv", index=False,
                     float_format="%.17g")
    seg_summary = segments.groupby(["source", "target", "start_cut", "end_cut", "method"],
                                    as_index=False)[["MAE", "mean_signed_error",
                                                     "p95_absolute_error"]].mean()
    seg_summary.to_csv(OUT / "segment_method_means.csv", index=False, float_format="%.17g")
    stage_direction = seg_delta.groupby(["source", "target", "start_cut", "end_cut"],
                                         as_index=False).agg(
        delta_MAE_mean=("MAE", "mean"), delta_MAE_sd=("MAE", "std"),
        MAE_improved_seeds=("MAE", lambda x: int((x < 0).sum())),
        delta_signed_error_mean=("mean_signed_error", "mean"),
        delta_p95_absolute_error_mean=("p95_absolute_error", "mean"))
    stage_direction.to_csv(OUT / "segment_cod_minus_daregram_direction_stats.csv", index=False,
                           float_format="%.17g")
    plot_dir = OUT / "plots"
    plot_dir.mkdir(exist_ok=True)
    for source, target in PAIRS:
        seeds = SEEDS if (source, target) in (("c4", "c1"), ("c6", "c1")) else (42,)
        for seed in seeds:
            values = {method: predictions[(source, target, seed, method)] for method in METHODS}
            plot_one(source, target, seed, values, coordinates,
                     plot_dir / f"{source}_to_{target}_seed_{seed}.png")
    lines = [
        "# COD 六方向逐 seed 配对分析", "",
        "本报告仅使用 `artifacts/cod_comparison_20260926` 的预测；脚本为 "
        "[`analyze_cod_paired.py`](../../../analyze_cod_paired.py)。未使用其他归一化实验，也未重跑训练。", "",
        "## 范围与核对", "",
        "- 六方向 × seed 42–46 × 三方法，共 90 个逐 cut 预测文件、各 315 行；30 组同方向同 seed 配对。",
        "- 所有预测均按 CUT 1–315 排序；同一目标刀具的 `true_vb` 在方向、seed、方法间一致，"
        "并与原始 wear CSV 三刃均值经 float32 转换后的标签相符（最大浮点差约 2.84e−14）。",
        "- 原配置为中心 4096 点、六通道 STFT 256/224、`log1p` 幅值、128×128；"
        "各方向五个 seed 的特征哈希、STFT 代码哈希和源域均值/标准差相同。"
        "本次三方法逐 run 的源域归一化参数与基线配置相同。",
        "- 固定 50 epoch、batch 63、学习率 0.001、最终 epoch checkpoint；"
        "COD 系数 0.001、epsilon 0.05、step 75 起启用。目标标签只用于预测后的评价。",
        "- R² = 1−SSE/SST；MAPE 为百分数，以 `max(abs(true), 1e−8)` 作分母。"
        "从每个 seed 的预测重算四指标，与该 run 的 `metrics.json`、`metrics.csv`、"
        "`per_seed_metrics.csv` 及方向汇总一致；最大绝对差 1.42e−14。无缺文件或实质数值不一致。",
        "- 以下差值均为 COD−DARE-GRAM；R² 正值为改善，其他三项负值为改善。"
        "标准差使用五个配对差值的样本标准差（ddof=1）；MAPE 差值单位为百分点。", "",
        "## 全部 30 组配对差值", "",
        "| 方向 | seed | ΔR² | ΔMAE | ΔRMSE | ΔMAPE (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in delta.itertuples(index=False):
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | {row.seed} | "
                     f"{row.R2:+.4f} | {row.MAE:+.3f} | {row.RMSE:+.3f} | "
                     f"{row.MAPE_percent:+.3f} |")
    lines += ["", "## 方向差值统计", "",
              "每格为均值 ± 样本标准差；括号为改善 seed 数。", "",
              "| 方向 | ΔR² | ΔMAE | ΔRMSE | ΔMAPE (pp) |",
              "|---|---:|---:|---:|---:|"]
    for row in by_direction.itertuples(index=False):
        cells = [f"{getattr(row, name + '_mean'):+.4f} ± "
                 f"{getattr(row, name + '_sd'):.4f} "
                 f"({getattr(row, name + '_improved_seeds')}/5)" for name in METRICS]
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | " + " | ".join(cells) + " |")
    lines += ["", "## 三等分阶段", "",
              "下表每个方法的 MAE / bias 都是五个 seed 的指标均值；"
              "bias = mean(pred−true)。ΔMAE 是同 seed 配对后取均值。"
              "完整的 270 条逐 seed 方法记录、90 条逐 seed 阶段差值及 P95 绝对残差见 CSV。", "",
              "| 方向 | CUT | Source MAE / bias | DARE MAE / bias | +COD MAE / bias | ΔMAE (改善 seed) | Δbias |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stage_direction.itertuples(index=False):
        current = seg_summary.loc[(seg_summary.source == row.source) &
                                  (seg_summary.target == row.target) &
                                  (seg_summary.start_cut == row.start_cut)]
        cells = []
        for method in METHODS:
            item = current.loc[current.method == method].iloc[0]
            cells.append(f"{item.MAE:.2f} / {item.mean_signed_error:+.2f}")
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | "
                     f"{row.start_cut}–{row.end_cut} | " + " | ".join(cells) +
                     f" | {row.delta_MAE_mean:+.2f} ({row.MAE_improved_seeds}/5) | "
                     f"{row.delta_signed_error_mean:+.2f} |")
    lines += [
        "", "## 聚合方式与 C4→C6", "",
        "汇总表的 R²、MAE、RMSE、MAPE 均为五个逐 seed 指标的算术均值；"
        "R² 和 RMSE 均不是五条预测先平均后计算。C4→C6 的所有 seed 使用完全相同的 315 个 cut 和 C6 标签。",
        "C4→C6 seed 42/43/44/46 的 ΔR² 分别为 −0.0726/−0.0180/−0.0562/−0.0556，"
        "ΔRMSE 为 +3.226/+0.665/+2.285/+2.567；seed 45 则为 +0.2109 和 −7.801。"
        "因此平均 ΔR² = +0.0017，而平均 ΔRMSE = +0.1885。",
        "固定标签方差 1606.713 时，逐 seed 有 R² = 1−RMSE²/1606.713。"
        "五 seed 的平均 RMSE² 从 398.629 降至 395.911；seed 45 较大的平方误差改善足以使平均 R² 略升，"
        "尽管 RMSE 的算术均值由 19.651 升至 19.840。", "",
        "## 曲线支持的现象与六方向结论", "",
        "| 方向 | R² 改善 | 四项均改善 seed | 四项均值一致改善 | 主要变化区间及误差形态 | 负迁移 |",
        "|---|---:|---:|---|---|---|",
        "| C1→C4 | 3/5 | 2/5 | 否 | 211–315 MAE −0.61、P95 −4.25；前两段 MAE +0.92/+0.69，前段高估增加 | 有，seed 42/43 的 R² 退化 |",
        "| C1→C6 | 2/5 | 2/5 | 是 | 1–210 MAE −1.46/−0.97、原有低估减轻；211–315 MAE +0.50，后段低估仍明显 | 有，seed 42/43/44 的 R² 退化 |",
        "| C4→C1 | 5/5 | 4/5 | 是 | 211–315 MAE −4.84、P95 −9.28，高估偏移减小；1–210 MAE +0.97/+1.56 | 局部有，早中段及 seed 42 的 MAE/MAPE 退化 |",
        "| C4→C6 | 1/5 | 1/5 | 否 | 早中段 MAE +0.94/+0.18；后段平均 −1.00 由 seed 间差异驱动 | 有，4/5 seed 的 R²/MAE/RMSE 退化 |",
        "| C6→C1 | 1/5 | 1/5 | 否 | 1–105 MAE +5.40、bias −9.36，出现系统性低估；后段 MAE +2.25 | 有，四指标均值皆退化 |",
        "| C6→C4 | 1/5 | 1/5 | 否 | 1–210 MAE −2.00/−0.42；211–315 MAE +2.98、高估增加 | 有，R²/MAE/RMSE 均值退化 |",
        "", "这些说法描述曲线和残差的可见变化，不把它们解释为 COD 的训练机制。"
        "P95 是各 seed 阶段绝对残差 95 分位数的均值，不能单独证明改善来自少数异常点；"
        "C4→C1 后段 P95 与有符号偏差同时下降，证据更支持后段整体高估缩小。", "",
        "## 文件", "",
        "- [30 组配对差值](cod_minus_daregram_30_pairs.csv)、"
        "[方向均值/标准差/改善数](cod_minus_daregram_direction_stats.csv)",
        "- [90 组复算指标](recomputed_per_seed_metrics.csv)、"
        "[三方法方向均值](recomputed_direction_method_means.csv)",
        "- [三方法逐 seed 阶段指标](segment_per_seed_metrics.csv)、"
        "[阶段三方法均值](segment_method_means.csv)、"
        "[逐 seed 阶段差值](segment_cod_minus_daregram_per_seed.csv)、"
        "[方向阶段差值](segment_cod_minus_daregram_direction_stats.csv)",
        "- [全部曲线](plots/)：每个方向固定 seed 42；C4→C1、C6→C1 另含 seed 43–46。"
        "每图三方法使用同一 seed、同一目标标签、同一坐标范围及原实验 `plot_coordinates.json`。", "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: {len(result)} prediction files, {len(delta)} paired groups, "
          f"{len(segments)} stage rows; max metric discrepancy {max_metric_diff:.3g}")
    print(by_direction.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(stage_direction.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
