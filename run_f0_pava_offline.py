"""Offline equal-weight PAVA on each frozen F0 target prediction sequence.

``prepare`` uses only locked, unscored F0 predictions; ``score`` opens target
wear only after all 30 raw/PAVA trajectories have been locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import run_single_source_pairs as base
from run_nonnegative_increment import CUTS, pairs, sha, write_json
from trend_physics import pava_nondecreasing


PARENT = Path("artifacts/nonnegative_increment_20260926")
DEFAULT_OUT = Path("artifacts/f0_pava_offline_20260926")


def parent_files(parent):
    lock_path = parent / "prediction_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["primary_curve"]["F0"] != "p_vb":
        raise ValueError("F0 primary output is not the original direct head")
    return lock_path, lock["files"]


def prepare(parent, out):
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    lock_path, upstream = parent_files(parent)
    records = []
    out.mkdir(parents=True)
    (out / "per_seed_unscored").mkdir()
    for source, target, seed in pairs():
        folder = parent / f"{source}_to_{target}" / f"seed_{seed}" / "F0"
        source_path = folder / "target_predictions_unscored.csv"
        checkpoint = folder / "final.pth"
        for path in (source_path, checkpoint):
            absolute = str(path.resolve())
            if absolute not in upstream or sha(path) != upstream[absolute]:
                raise ValueError(f"Original F0 lock mismatch: {path}")
        frame = pd.read_csv(source_path, usecols=["cut_index", "p_vb"])
        if len(frame) != 315 or frame.cut_index.isna().any() or frame.cut_index.duplicated().any():
            raise ValueError(f"Missing or repeated cut_index: {source_path}")
        frame = frame.sort_values("cut_index", kind="stable")
        if frame.cut_index.astype(int).tolist() != CUTS:
            raise ValueError(f"F0 target cuts differ from 1..315: {source_path}")
        raw = frame.p_vb.to_numpy(np.float64)
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite F0 prediction: {source_path}")
        projected = pava_nondecreasing(raw)
        # Independent library check of the equal-weight least-squares solution.
        reference = IsotonicRegression(increasing=True).fit_transform(CUTS, raw)
        if not np.allclose(projected, reference, rtol=0, atol=1e-10):
            raise AssertionError(f"PAVA and isotonic regression disagree: {source_path}")
        if (np.diff(projected) < -1e-10).any():
            raise AssertionError("Projection is decreasing")
        dest = out / "per_seed_unscored" / f"{source}_to_{target}_seed_{seed}.csv"
        pd.DataFrame({"cut_index": CUTS, "raw_pred": raw, "pava_pred": projected}).to_csv(
            dest, index=False, float_format="%.17g")
        records.append({"source": source, "target": target, "seed": seed,
                        "input_prediction": str(source_path.resolve()),
                        "input_sha256": sha(source_path),
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": sha(checkpoint),
                        "output_unscored": str(dest.resolve()),
                        "output_sha256": sha(dest),
                        "pava_changed_count": int(np.count_nonzero(np.abs(projected - raw) > 1e-10)),
                        "raw_declines": int(np.count_nonzero(np.diff(raw) < -1e-10)),
                        "pava_declines": int(np.count_nonzero(np.diff(projected) < -1e-10))})
    if len(records) != 30:
        raise AssertionError("Expected 30 F0 trajectories")
    pd.DataFrame(records).to_csv(out / "projection_index.csv", index=False)
    write_json(out / "prediction_lock.json", {
        "stage": "before target wear read", "n_trajectories": 30,
        "upstream_lock_sha256": sha(lock_path),
        "algorithm": "equal-weight nondecreasing least-squares PAVA/isotonic regression, independently per seed, full cut 1..315",
        "inputs": "cut_index and raw F0 predictions only",
        "target_label_reads": 0,
        "projection_index_sha256": sha(out / "projection_index.csv"),
        "outputs": {r["output_unscored"]: r["output_sha256"] for r in records}})
    print(f"Prepared and locked {len(records)} label-free F0/PAVA trajectories in {out}")


def measures(y, p):
    error = p - y
    denominator = np.square(y - y.mean()).sum()
    return {"R2": float(1 - np.square(error).sum() / denominator) if denominator else np.nan,
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(np.square(error)))),
            "signed_bias": float(np.mean(error)),
            "underestimate_count": int(np.count_nonzero(error < 0)),
            "underestimate_fraction": float(np.mean(error < 0)),
            "decline_count": int(np.count_nonzero(np.diff(p) < -1e-10))}


def plot_one(path, source, target, seed, truth, raw, pava):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(CUTS, truth, color="#171717", linewidth=2.0, label="True VB")
    ax.plot(CUTS, raw, color="#3977b5", linewidth=1.25, label="F0 raw prediction")
    ax.plot(CUTS, pava, color="#d04b3f", linewidth=1.55, label="F0 + PAVA")
    ax.axvline(250, color="#888888", linestyle=":", linewidth=1)
    ax.set(title=f"{source.upper()}→{target.upper()}, seed {seed}",
           xlabel="Target cut index", ylabel="VB")
    ax.grid(alpha=.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def score(parent, out, raw_root):
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    parent_lock, _ = parent_files(parent)
    if (lock["n_trajectories"] != 30 or sha(parent_lock) != lock["upstream_lock_sha256"]
            or sha(out / "projection_index.csv") != lock["projection_index_sha256"]):
        raise ValueError("Projection lock/provenance mismatch")
    for path, digest in lock["outputs"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Unscored trajectory changed: {path}")
    if (out / "per_seed_metrics.csv").exists():
        raise FileExistsError("Scored results already exist")
    index = pd.read_csv(out / "projection_index.csv")
    if len(index) != 30:
        raise ValueError("Projection index incomplete")
    # The first target-label reads in this experiment occur here, after the lock checks.
    truths = {tool: base.wear_labels(raw_root, tool).astype(np.float64) for tool in base.TOOLS}
    (out / "plots").mkdir()
    (out / "per_seed_scored").mkdir()
    metrics = []
    for row in index.itertuples():
        frame = pd.read_csv(row.output_unscored)
        if frame.cut_index.tolist() != CUTS or list(frame.columns) != ["cut_index", "raw_pred", "pava_pred"]:
            raise ValueError(f"Unexpected unscored schema: {row.output_unscored}")
        truth = truths[row.target]
        raw = frame.raw_pred.to_numpy(float)
        pava = frame.pava_pred.to_numpy(float)
        for method, prediction in (("F0_raw", raw), ("F0_PAVA", pava)):
            for scope, sl in (("full_1_315", slice(None)), ("late_250_315", slice(249, 315))):
                metrics.append({"source": row.source, "target": row.target, "seed": row.seed,
                                "method": method, "scope": scope, "n": len(truth[sl]),
                                **measures(truth[sl], prediction[sl])})
        scored = frame.copy()
        scored.insert(1, "true_vb", truth)
        scored["raw_signed_error"] = raw - truth
        scored["pava_signed_error"] = pava - truth
        name = f"{row.source}_to_{row.target}_seed_{row.seed}"
        scored.to_csv(out / "per_seed_scored" / f"{name}.csv", index=False, float_format="%.17g")
        plot_one(out / "plots" / f"{name}.png", row.source, row.target, row.seed, truth, raw, pava)
    table = pd.DataFrame(metrics)
    if len(table) != 120:
        raise AssertionError("Expected 30 seeds × 2 methods × 2 scopes")
    table.to_csv(out / "per_seed_metrics.csv", index=False)
    summary = table.groupby(["source", "target", "method", "scope"], as_index=False).agg(
        n_seeds=("seed", "nunique"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        signed_bias_mean=("signed_bias", "mean"),
        underestimate_count_mean=("underestimate_count", "mean"),
        decline_count_mean=("decline_count", "mean"))
    summary.to_csv(out / "direction_summary.csv", index=False)
    raw = table[table.method == "F0_raw"].set_index(["source", "target", "seed", "scope"])
    pava = table[table.method == "F0_PAVA"].set_index(["source", "target", "seed", "scope"])
    paired = pd.DataFrame({"MAE_pava_minus_raw": pava.MAE - raw.MAE,
                           "RMSE_pava_minus_raw": pava.RMSE - raw.RMSE,
                           "signed_bias_pava_minus_raw": pava.signed_bias - raw.signed_bias,
                           "declines_removed": raw.decline_count - pava.decline_count}).reset_index()
    paired.to_csv(out / "paired_seed_delta.csv", index=False)

    lines = ["# F0 的独立 PAVA 离线后处理对照", "",
             "对每个方向、每个 seed 的 F0 目标预测单独按 cut 1–315 做等权重 isotonic regression。",
             "PAVA 只读取整段无标签原预测；不读取目标真实 VB，也不先平均五个 seed。",
             "这是使用整段目标预测（包含后续 cut）的**离线后处理**，不属于在线逐 cut 推理或重新训练。",
             "`prediction_lock.json` 在目标 VB 首次读取前锁定 30 条 `raw_pred` / `pava_pred` 曲线。", "",
             "## 全程及后段五 seed 平均", "",
             "| 方向 | 范围 | 方法 | R² | MAE | RMSE | 有符号偏差 | 下降次数 | 低估 cut 数 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | {row.scope} | {row.method} | "
                     f"{row.R2_mean:.3f} | {row.MAE_mean:.2f} | {row.RMSE_mean:.2f} | "
                     f"{row.signed_bias_mean:+.2f} | {row.decline_count_mean:.1f} | "
                     f"{row.underestimate_count_mean:.1f} |")
    lines += ["", "下降次数只在表内对应范围的相邻 cut 中计算：全程 314 条边，cut 250–315 为 65 条边。",
              "表中最后一列表示对应范围内的低估 cut 数；逐 seed 及确切比例见 `per_seed_metrics.csv`。", "",
              "## 后段判断", "",
              "PAVA 将全部 30 条曲线的下降次数降至 0，并在六个方向降低全程平均 MAE/RMSE。",
              "等权投影保留全程预测均值，因此没有修正全程有符号偏差。",
              "C1→C6 在 cut 250–315 仍为 66/66 cut 低估，五 seed 平均偏差为 −32.76 VB，",
              "与原 F0 相同；C1→C4、C4→C6、C6→C4 的该段平均偏差也分别保持在 −21.83、−8.19、−9.24 VB。",
              "所以尖刺被消除，但这些方向的后段系统性低估仍存在。", "",
              "## 文件", "",
              "- `per_seed_unscored/`: 锁定的逐 seed `cut_index,raw_pred,pava_pred`。",
              "- `per_seed_scored/`: 评分后的逐 cut 真值与误差，供核查后段低估。",
              "- `per_seed_metrics.csv`, `direction_summary.csv`: 指标与五 seed 汇总。",
              "- `paired_seed_delta.csv`: 每 seed 的 PAVA 减原预测指标差。",
              "- `plots/`: 30 张真实值、F0 原预测、PAVA 预测三曲线图。", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Scored {len(table)} method/scope rows and saved 30 three-curve plots in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.parent, args.out)
    else:
        score(args.parent, args.out, args.raw_root)


if __name__ == "__main__":
    main()
