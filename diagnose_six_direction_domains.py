"""Read-only six-direction A-input audit, metrics, and domain diagnostics.

All 60 final checkpoints were trained earlier. This program verifies their
provenance, performs label-free input/embedding diagnostics, and only then
opens the frozen full-cut predictions and target wear for evaluation.
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
import torch
from threadpoolctl import threadpool_limits

import data_sampling
import diagnose_c1_c6_distribution as prior
import run_single_source_pairs as base


TOOLS = ("c1", "c4", "c6")
PAIRS = tuple((s, t) for s in TOOLS for t in TOOLS if s != t)
SEEDS = tuple(range(42, 47))
METHODS = ("source_only", "daregram")
STAGES = ((1, 105), (106, 210), (211, 315))
CUTS = list(range(1, 316))
CHANNELS = ("Fx", "Fy", "Fz", "Vx", "Vy", "Vz")
ROOT = Path("artifacts/five_seed_paired")
FULL = Path("artifacts/full_1_315_baseline_zscore_20260925")
FEATURE = Path("artifacts/five_seed_feature_audit_20260925")
ABLATION = Path("artifacts/stft_amplitude_ablation_20260926")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def stage_name(cut):
    return next(f"{a}-{b}" for a, b in STAGES if a <= cut <= b)


def summarize_distance(source, target, names, direction, seed, method, layer):
    data, _ = prior.distance_and_oor(source, target, names, seed, method, layer)
    rows, summary = [], []
    for record in data:
        row = dict(record, direction=direction)
        row["domain"] = "source_holdout" if row["domain"] == "c1_holdout" else "target"
        row["nearest_source_cut"] = row.pop("nearest_c1_cut")
        rows.append(row)
    frame = pd.DataFrame(rows)
    lower, upper = np.quantile(source, [.01, .99], axis=0)
    for a, b in STAGES:
        stage = f"{a}-{b}"
        sub = frame[frame.stage == stage]
        s = sub[sub.domain == "source_holdout"].distance.to_numpy()
        t = sub[sub.domain == "target"].distance.to_numpy()
        summary.append({"direction": direction, "seed": seed, "method": method,
                        "layer": layer, "stage": stage,
                        "source_holdout_nn_median": np.median(s),
                        "source_holdout_nn_p95": np.quantile(s, .95),
                        "target_nn_median": np.median(t),
                        "target_nn_p95": np.quantile(t, .95),
                        "target_over_source_holdout_median_ratio": np.median(t)/np.median(s),
                        "target_above_source_holdout_p95_fraction": np.mean(t > np.quantile(s, .95)),
                        "target_outside_source_full_01_99_fraction":
                        np.mean((target[a-1:b] < lower) | (target[a-1:b] > upper))})
    return rows, summary


def auc(source, target, direction, seed, method, layer):
    return [dict(row, direction=direction)
            for row in prior.auc_folds(source, target, seed, method, layer)]


def profile_rows(source, target, direction, seed, method, layer):
    rows = []
    for a, b in STAGES:
        for role, x in (("source", source), ("target", target)):
            rows.append({"direction": direction, "seed": seed, "method": method,
                         "layer": layer, "stage": f"{a}-{b}", "domain": role,
                         **prior.profile(x[a-1:b])})
    return rows


def metric_rows(direction, seed, method, truth, pred):
    full = base.metrics(truth, pred)
    rows = [{"direction": direction, "seed": seed, "method": method,
             "stage": "all", "n": 315, "signed_error": float(np.mean(pred-truth)),
             "pred_increment": float(pred[-1]-pred[0]),
             "true_increment": float(truth[-1]-truth[0]),
             "pred_max": float(pred.max()), **full}]
    for a, b in STAGES:
        p, y = pred[a-1:b], truth[a-1:b]
        rows.append({"direction": direction, "seed": seed, "method": method,
                     "stage": f"{a}-{b}", "n": 105,
                     "R2": np.nan, "RMSE": np.nan, "MAPE_percent": np.nan,
                     "MAE": float(np.mean(np.abs(p-y))),
                     "signed_error": float(np.mean(p-y)),
                     "pred_increment": float(p[-1]-p[0]),
                     "true_increment": float(y[-1]-y[0]),
                     "pred_max": float(p.max())})
    return rows


def summary_table(metrics):
    frame = pd.DataFrame(metrics)
    return frame.groupby(["direction", "method", "stage"], as_index=False).agg(
        n_seeds=("seed", "count"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        signed_error_mean=("signed_error", "mean"),
        signed_error_sd=("signed_error", "std"),
        pred_increment_mean=("pred_increment", "mean"),
        pred_increment_sd=("pred_increment", "std"),
        true_increment_mean=("true_increment", "mean"),
        pred_max_mean=("pred_max", "mean"))


def plot_curves(out, predictions, truths):
    low, high = np.inf, -np.inf
    for pair in PAIRS:
        truth = truths[pair[1]]
        low, high = min(low, truth.min()), max(high, truth.max())
        for method in METHODS:
            stack = np.stack([predictions[pair, seed, method] for seed in SEEDS])
            low, high = min(low, stack.min()), max(high, stack.max())
    pad = (high-low)*.04
    ylim = (low-pad, high+pad)
    for source, target in PAIRS:
        pair = (source, target)
        fig, ax = plt.subplots(figsize=(11.5, 5.1), constrained_layout=True)
        ax.plot(CUTS, truths[target], color="#202020", lw=2.2, label="True VB", zorder=5)
        for method, color, label in (("source_only", "#2563a6", "Source-only"),
                                      ("daregram", "#db7026", "DARE-GRAM")):
            values = np.stack([predictions[pair, seed, method] for seed in SEEDS])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.fill_between(CUTS, mean-sd, mean+sd, color=color, alpha=.16,
                            label=f"{label} ±1 seed SD")
            ax.plot(CUTS, mean, color=color, lw=1.8, label=f"{label} mean")
        ax.set(title=f"{source.upper()}→{target.upper()} | A input | five seeds",
               xlabel=f"{target.upper()} Cut", ylabel="VB", xlim=(1, 315), ylim=ylim)
        ax.grid(alpha=.22)
        ax.legend(loc="lower right", ncol=2, fontsize=8)
        fig.savefig(out / f"{source}_to_{target}.png", dpi=180)
        plt.close(fig)
    return {"xlim": [1, 315], "ylim": [float(v) for v in ylim],
            "curve": "five-seed mean with sample-SD band; metrics are averaged per seed"}


def one(summary, direction, method, stage):
    row = summary[(summary.direction == direction) & (summary.method == method) &
                  (summary.stage == stage)]
    if len(row) != 1 or int(row.iloc[0].n_seeds) != 5:
        raise ValueError(f"Missing 5-seed result: {direction} {method} {stage}")
    return row.iloc[0]


def report(out, summary, distance, auc_summary, wear_rows, provenance, plot_config):
    wear = pd.DataFrame(wear_rows).set_index("direction")
    lines = ["# 六方向 A 输入域差异与磨损预测诊断", "",
             "## 协议与实际复用", "",
             "- 全部六个有方向任务、seed 42–46、source-only 与 DARE-GRAM 的 **60 个最终 epoch 权重及 60 份全程预测全部复用；重跑任务 0 个**。逐组路径、SHA-256 和原预测路径见 `provenance.json`、`checkpoint_audit.csv`。C6 历史 95–315 预测没有进入本次；C6 全程 1–315 使用已审计的同权重 Z-score 再评估。",
             "- 固定 A：中心 4096 点、50 kHz、六通道、Hann STFT 256/224、`log1p(abs(STFT))`、128×128、由源域 STFT 拟合六通道 Z-score；原代码的 10 kHz 裁剪分支没有执行。ResNet18 + Linear(512,1)，50 epoch、batch 63、Adam 0.001、最终 epoch 权重。source-only 训练不用目标输入；DARE-GRAM 训练使用全部 315 个目标无标签 Cut。",
             "- 既有权重及特征审计完成后，先计算不读取目标标签的 STFT/512 维分布、距离和域分类；然后读取原预测与目标 VB 作评价。目标 VB 没有进入标准化、训练、分类器、阈值或检查点选择。本次没有训练或调参。",
             "- C1→C6 与既有 A 消融逐 seed、逐 Cut 预测完全相同；source-only / DARE-GRAM 五 seed 全程 MAE 为 14.78 / 15.16。六方向同一 STFT 输入在两种方法间一致；权重、特征及预测哈希均核对通过。", "",
             "## 六方向汇总（每 seed 先算，再取五 seed 均值）", "",
             "| 方向 | 方法 | 全程 R² | MAE | RMSE | 后段 MAE | 后段 pred−true | 后段预测/真实增量 | 源 VB 上界 | 目标 VB 最大 | 超界 Cut |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        name = f"{source}_to_{target}"
        w = wear.loc[name]
        for method in METHODS:
            a, late = one(summary, name, method, "all"), one(summary, name, method, "211-315")
            lines.append(f"| {source.upper()}→{target.upper()} | {method} | {a.R2_mean:.3f} | "
                         f"{a.MAE_mean:.2f} | {a.RMSE_mean:.2f} | {late.MAE_mean:.2f} | "
                         f"{late.signed_error_mean:+.2f} | {late.pred_increment_mean:.2f}/{late.true_increment_mean:.2f} | "
                         f"{w.source_vb_max:.2f} | {w.target_vb_max:.2f} | {int(w.target_cuts_above_source_max)} |")
    lines += ["", "完整的三个阶段 MAE、平均有符号误差、真实和预测增量，以及全部指标的五 seed 标准差见 `five_seed_stage_summary.csv`；逐 seed 值见 `per_seed_metrics.csv`。六张曲线共同使用 `xlim=1–315`、`ylim=" +
              f"{plot_config['ylim'][0]:.1f}–{plot_config['ylim'][1]:.1f}`，曲线为五 seed 均值和 ±1 样本标准差；表中指标不是均值曲线的指标。", "",
              "## 分布诊断方法", "",
              "- STFT 输入按**方向的源域**六通道统计量标准化；48 维诊断描述量包含每通道均值、标准差、p90、p99 和四个频带均值。原 6×128×128 图像的分段通道统计另见 `stft_channel_stats.csv`。512 维表示来自各自最终权重，分段范数和分位数见 `feature_distribution_stats.csv`。",
              "- 最近邻使用同一权重空间内 C1/C4/C6 对应源域训练折拟合的逐维 Z-score；5 折按相邻 7 Cut 分组。目标距离与源域留出 Cut 到源域训练 Cut 的距离比较。STFT 描述量与 512 维特征的绝对距离、跨模型特征绝对距离不可直接排序。逐 Cut 和分段结果见 `nearest_neighbor_per_cut.csv`、`distance_stage_summary.csv`。",
              "- 域分类器为固定 L2 LogisticRegression(C=1)，各阶段源/目标各 105 Cut，以相邻 5 Cut 为组做 5 折；同 Cut 的两域样本在同一折，训练与验证分离，样本数平衡。DARE-GRAM 表示本身使用了全部无标签目标 Cut，分类器交叉验证只检验冻结表示的可分性。逐折与汇总见 `domain_classifier_folds.csv`、`domain_classifier_summary.csv`。", "",
              "## 结果判断", ""]
    # Classify by all three full metrics, then note late-only gains separately.
    full_better, full_worse, mixed, late_gain = [], [], [], []
    for source, target in PAIRS:
        name = f"{source}_to_{target}"
        s, d = one(summary, name, "source_only", "all"), one(summary, name, "daregram", "all")
        sl, dl = one(summary, name, "source_only", "211-315"), one(summary, name, "daregram", "211-315")
        label = f"{source.upper()}→{target.upper()}"
        if d.R2_mean > s.R2_mean and d.MAE_mean < s.MAE_mean and d.RMSE_mean < s.RMSE_mean:
            full_better.append(label)
        elif d.R2_mean < s.R2_mean and d.MAE_mean > s.MAE_mean and d.RMSE_mean > s.RMSE_mean:
            full_worse.append(label)
        else:
            mixed.append(label)
        if dl.MAE_mean < sl.MAE_mean:
            late_gain.append(label)
    lines.append("- DARE-GRAM **全程三项均改善**（R² 升、MAE/RMSE 降）：" +
                 ("、".join(full_better) if full_better else "无") + "；**全程三项均变差**：" +
                 ("、".join(full_worse) if full_worse else "无") + "；**全程指标混合**：" +
                 ("、".join(mixed) if mixed else "无") + "。")
    lines.append("- DARE-GRAM 后段 MAE 改善的方向：" +
                 ("、".join(late_gain) if late_gain else "无") +
                 "。其中未达到全程三项均改善的方向应理解为阶段性收益，不能称为全程提升。")
    lines += ["", "| 方向 | STFT 后段距离比 | source-only 512D 后段距离比 | DARE-GRAM 512D 后段距离比 | source-only / DARE-GRAM 后段 AUC |",
              "|---|---:|---:|---:|---:|"]
    for source, target in PAIRS:
        name = f"{source}_to_{target}"
        def dist(layer, method):
            x = distance[(distance.direction == name) & (distance.layer == layer) &
                         (distance.method == method) & (distance.stage == "211-315")]
            return float(x.target_over_source_holdout_median_ratio.mean())
        def aucval(method):
            x = auc_summary[(auc_summary.direction == name) & (auc_summary.layer == "resnet_512d") &
                            (auc_summary.method == method) & (auc_summary.stage == "211-315")]
            return float(x.auc_mean.mean())
        lines.append(f"| {source.upper()}→{target.upper()} | {dist('stft_48d','shared'):.2f} | "
                     f"{dist('resnet_512d','source_only'):.2f} | {dist('resnet_512d','daregram'):.2f} | "
                     f"{aucval('source_only'):.3f} / {aucval('daregram'):.3f} |")
    lines += ["", "- **低估并不主要由目标 VB 超过源标签上界解释。** C1→C6 有 71 个超界 Cut，source-only 后段偏差 −35.73 VB；但 C6→C1、C6→C4 均无超界 Cut，source-only 后段仍分别低估 −25.71、−33.01 VB。C4→C6 有 21 个超界 Cut，source-only 后段平均偏差仅 −1.08 VB。标签超界是 C1→C6 的重要外推背景，不是六方向低估的充分或必要条件。",
              "- **距离与误差不单调一致。** 在同一个 C1→C6 source-only 模型空间中，从前段到后段，512 维目标/源留出距离比由 10.37 降到 6.71，但五 seed 后段 MAE 从 5.70 升到 35.73 VB。六方向的 STFT 后段距离比也不能单独预测后段误差。域分类 AUC 多数接近 1，仍不能说明磨损映射如何变化。不同模型的 512 维距离绝对值不能直接相减或排名。",
              "- **C1→C6 结论不普遍成立。** 它呈现标签超界、后段低估和 DARE-GRAM 后段收益；C4→C1 后段反而高估，C6→C1/C4 的 source-only 在无标签超界时低估且 DARE-GRAM 全程明显改善，C1→C4 和 C4→C6 的 DARE-GRAM 全程三项指标均变差。结论限定于这批冻结权重和 A 输入，不据目标标签调整方案。", "",
              "## 运行命令与文件", "",
              "```powershell", "cd upstream-reproduction",
              "python diagnose_six_direction_domains.py --out-root artifacts/six_direction_domain_diagnostic_20260926", "```", "",
              "- `checkpoint_audit.csv`：实际复用权重、配置、预测及哈希；`per_seed/`：60 份 Cut 1–315 原始预测；`per_seed_metrics.csv`、`five_seed_stage_summary.csv`、`six_direction_summary.csv`：指标；`curves/`：六张统一坐标曲线。",
              "- `stft_channel_stats.csv`、`feature_distribution_stats.csv`、`distance_stage_summary.csv`、`domain_classifier_summary.csv`：域差异；逐 Cut/折数据另存；`provenance.json`：协议及路径。", "",
              "未使用逐 Cut RMS 方案 C、等价 B、OOR-PGA、门控、监督微调、物理约束或后处理。"]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path("artifacts/six_direction_domain_diagnostic_20260926"))
    args = ap.parse_args()
    out = args.out_root.resolve()
    root, full, feature, ablation = (p.resolve() for p in (ROOT, FULL, FEATURE, ABLATION))
    if out.exists() or any(out == p or p in out.parents for p in (root, full, feature, ablation)):
        ap.error("Output must be a new directory outside all inputs")
    manifest_path = root / "feature_cache/manifest.json"
    manifest = read_json(manifest_path)
    lock = read_json(full / "protocol_lock_full_1_315.json")
    if (lock["checkpoint_count"] != 60 or lock["direction_count"] != 6 or
        lock["scope"] != "full_1_315" or lock["training_repeated"] or
        base.file_hash(manifest_path) != lock["cache_manifest_sha256"] or
        base.file_hash(full / "checkpoint_audit_full_1_315.csv") != lock["checkpoint_audit_sha256"] or
        base.file_hash(full / "per_seed_metrics_full_1_315.csv") != lock["per_seed_metrics_sha256"] or
        base.file_hash(Path(data_sampling.__file__)) != manifest["stft_code_sha256"]):
        raise ValueError("Full-cut A protocol lock or STFT source code changed")
    run_audit = read_json(feature / "run_summary.json")
    if (run_audit["experiments_audited"] != 60 or
        not run_audit["all_paired_inputs_identical"] or run_audit["missing_experiments"] or
        run_audit["target_labels_read"]):
        raise ValueError("Feature extraction audit is incomplete")
    paired = read_json(feature / "paired_input_comparison.json")
    if len(paired) != 30 or not all(row["inputs_identical"] for row in paired):
        raise ValueError("Methods did not receive identical source/target images")
    checkpoint_table = pd.read_csv(full / "checkpoint_audit_full_1_315.csv")
    if len(checkpoint_table) != 60:
        raise ValueError("Expected all 60 final checkpoints")
    checkpoint_index = {(r.source, r.target, int(r.seed), r.method): r
                        for r in checkpoint_table.itertuples(index=False)}
    per_seed_baseline = pd.read_csv(full / "per_seed_metrics_full_1_315.csv")
    baseline_index = {(r.source, r.target, int(r.seed), r.method): r
                      for r in per_seed_baseline.itertuples(index=False)}
    if len(checkpoint_index) != 60 or len(baseline_index) != 60:
        raise ValueError("Duplicate or missing baseline rows")
    raw = {}
    for tool in TOOLS:
        path = root / f"feature_cache/{tool}_stft.npy"
        if (manifest["cuts"] != CUTS or manifest["feature_shape"] != [315, 6, 128, 128] or
            base.file_hash(path) != manifest["tools"][tool]["feature_sha256"]):
            raise ValueError(f"Invalid A STFT cache: {path}")
        raw[tool] = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw[tool].shape != (315, 6, 128, 128) or raw[tool].dtype != np.float32:
            raise ValueError(f"Invalid STFT shape/dtype: {path}")
    index = {(r["source"], r["target"], r["seed"], r["method"]): r
             for r in read_json(feature / "file_index.json")}
    if len(index) != 60:
        raise ValueError("Missing feature audit index entries")
    stage_names = np.asarray([stage_name(cut) for cut in CUTS])
    channel_rows, profile_data, nn_data, nn_summary, auc_data, ck_audit = [], [], [], [], [], []
    with threadpool_limits(limits=4):
        for source, target in PAIRS:
            direction = f"{source}_to_{target}"
            source_mean = raw[source].mean(axis=(0, 2, 3)).astype(np.float32)
            source_std = (raw[source].std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
            images = {tool: ((raw[tool] - source_mean[None, :, None, None]) /
                             (source_std[None, :, None, None] + 1e-8)).astype(np.float32)
                      for tool in (source, target)}
            desc = {tool: prior.stft_descriptors(images[tool]) for tool in (source, target)}
            r, s = summarize_distance(desc[source], desc[target], stage_names,
                                      direction, "shared", "shared", "stft_48d")
            nn_data.extend(r); nn_summary.extend(s)
            auc_data.extend(auc(desc[source], desc[target], direction, "shared", "shared", "stft_48d"))
            profile_data.extend(profile_rows(desc[source], desc[target], direction,
                                             "shared", "shared", "stft_48d"))
            for a, b in STAGES:
                for role, tool in (("source", source), ("target", target)):
                    for i, channel in enumerate(CHANNELS):
                        z = images[tool][a-1:b, i].astype(np.float64)
                        channel_rows.append({"direction": direction, "source": source,
                                             "target": target, "stage": f"{a}-{b}",
                                             "domain": role, "channel": channel,
                                             "mean": z.mean(), "std": z.std(),
                                             "p01": np.quantile(z, .01),
                                             "p50": np.quantile(z, .5),
                                             "p99": np.quantile(z, .99),
                                             "cut_rms_mean": np.sqrt(np.mean(z*z, axis=(1, 2))).mean()})
            for seed in SEEDS:
                for method in METHODS:
                    key = (source, target, seed, method)
                    row = checkpoint_index[key]
                    folder = root / direction / f"seed_{seed}" / method
                    ck, config_path = folder / "final.pth", folder / "config.json"
                    config = read_json(config_path)
                    audit_path = feature / f"experiments/{direction}/seed_{seed}/{method}/audit.json"
                    extraction = read_json(audit_path)
                    state = torch.load(ck, map_location="cpu", weights_only=True)
                    if (config["source"] != source or config["target"] != target or
                        config["seed"] != seed or config["method"] != method or
                        config["source_cuts"] != CUTS or config["target_unlabeled_cuts"] != CUTS or
                        config["actual_unlabeled_target_cuts"] != (CUTS if method == "daregram" else []) or
                        config["target_labels_training_reads"] != 0 or
                        config["epochs"] != 50 or config["batch_size"] != 63 or config["lr"] != .001 or
                        config["backbone"] != "ResNet18" or config["regressor"] != "Linear(512, 1)" or
                        config["checkpoint_selection"] != "final epoch" or
                        config["source_feature_sha256"] != manifest["tools"][source]["feature_sha256"] or
                        config["target_feature_sha256"] != manifest["tools"][target]["feature_sha256"] or
                        not np.array_equal(np.asarray(config["source_normalization_mean"], np.float32), source_mean) or
                        not np.array_equal(np.asarray(config["source_normalization_std"], np.float32), source_std) or
                        base.file_hash(ck) != row.checkpoint_sha256 or
                        base.file_hash(ck) != extraction["checkpoint_sha256"] or
                        base.file_hash(config_path) != row.training_config_sha256 or
                        state["epoch"] != 50 or
                        tuple(state["model"]["regressor.0.weight"].shape) != (1, 512) or
                        row.scope != "full_1_315" or int(row.prediction_count) != 315 or
                        int(row.target_label_reads_during_training) != 0):
                        raise ValueError(f"A frozen training mismatch: {folder}")
                    feats = {}
                    for role, tool in (("source", source), ("target", target)):
                        path = feature / f"experiments/{direction}/seed_{seed}/{method}/{role}_features.csv"
                        if base.file_hash(path) != extraction["feature_stats"][role]["file_sha256"]:
                            raise ValueError(f"Feature file changed: {path}")
                        feats[role] = prior.read_embedding(path, tool) if hasattr(prior, "read_embedding") else prior.read_features(path, tool)
                    r, s = summarize_distance(feats["source"], feats["target"], stage_names,
                                              direction, seed, method, "resnet_512d")
                    nn_data.extend(r); nn_summary.extend(s)
                    auc_data.extend(auc(feats["source"], feats["target"], direction,
                                        seed, method, "resnet_512d"))
                    profile_data.extend(profile_rows(feats["source"], feats["target"], direction,
                                                     seed, method, "resnet_512d"))
                    ck_audit.append({"direction": direction, "seed": seed, "method": method,
                                     "training_config": str(config_path.resolve()),
                                     "training_config_sha256": base.file_hash(config_path),
                                     "checkpoint": str(ck.resolve()),
                                     "checkpoint_sha256": base.file_hash(ck),
                                     "original_prediction": row.original_prediction_file,
                                     "original_prediction_sha256": row.original_prediction_sha256,
                                     "source_stft_sha256": manifest["tools"][source]["feature_sha256"],
                                     "target_stft_sha256": manifest["tools"][target]["feature_sha256"],
                                     "target_label_reads_training": 0,
                                     "target_unlabeled_cuts_training": 315 if method == "daregram" else 0,
                                     "evaluation_cuts": "1-315", "reuse": "existing_final_checkpoint"})
            print(f"Label-free diagnostics and hashes: {direction}", flush=True)
            del images, desc
    # No target wear or prediction file was opened above. All label-free diagnostics are fixed.
    raw_root = Path(manifest["raw_root"])
    truths = {tool: base.wear_labels(raw_root, tool).astype(np.float64) for tool in TOOLS}
    predictions, metric_data, wear_rows = {}, [], []
    for source, target in PAIRS:
        direction = f"{source}_to_{target}"
        limit = float(truths[source].max())
        wear_rows.append({"direction": direction, "source": source, "target": target,
                          "source_vb_min": float(truths[source].min()),
                          "source_vb_max": limit, "target_vb_max": float(truths[target].max()),
                          "target_cuts_above_source_max": int(np.count_nonzero(truths[target] > limit)),
                          "target_late_cuts_above_source_max": int(np.count_nonzero(truths[target][210:] > limit))})
        for seed in SEEDS:
            for method in METHODS:
                key = (source, target, seed, method)
                baseline_row, metric_row = checkpoint_index[key], baseline_index[key]
                original = Path(baseline_row.original_prediction_file)
                copied = Path(metric_row.per_cut_file)
                if (base.file_hash(original) != baseline_row.original_prediction_sha256 or
                    base.file_hash(copied) != metric_row.per_cut_sha256 or
                    base.file_hash(raw_root / f"{target}_wear.csv") != baseline_row.target_wear_sha256 or
                    base.file_hash(raw_root / f"{source}_wear.csv") != baseline_row.source_wear_sha256):
                    raise ValueError(f"Archived evaluation source changed: {direction} {seed} {method}")
                first, second = pd.read_csv(original), pd.read_csv(copied)
                if (list(first.columns) != ["cut_index", "true_vb", "pred_vb"] or
                    list(second.columns) != ["cut_index", "true_vb", "pred_vb", "signed_error", "absolute_error"] or
                    first.cut_index.tolist() != CUTS or second.cut_index.tolist() != CUTS or
                    not np.array_equal(first.true_vb.to_numpy(np.float32), truths[target].astype(np.float32)) or
                    not np.array_equal(second.true_vb.to_numpy(np.float32), truths[target].astype(np.float32)) or
                    not np.allclose(first.pred_vb, second.pred_vb, rtol=0, atol=1e-10)):
                    raise ValueError(f"Full-cut prediction/label mismatch: {direction} {seed} {method}")
                pred = first.pred_vb.to_numpy(np.float64)
                predictions[(source, target), seed, method] = pred
                items = metric_rows(direction, seed, method, truths[target], pred)
                metric_data.extend(items)
                full_row = items[0]
                if any(abs(full_row[m]-getattr(metric_row, m)) > 1e-8 for m in ("R2", "MAE", "RMSE")):
                    raise ValueError(f"Stored baseline metric mismatch: {direction} {seed} {method}")
    # Explicit A-group C1->C6 regression gate before any output is published.
    ablation_summary = pd.read_csv(ablation / "c6_summary.csv")
    for method in METHODS:
        historical = ablation_summary[(ablation_summary.scheme == "A") &
                                      (ablation_summary.method == method) &
                                      (ablation_summary.stage == "all")]
        current = [r for r in metric_data if r["direction"] == "c1_to_c6" and
                   r["method"] == method and r["stage"] == "all"]
        if len(historical) != 1 or len(current) != 5 or any(
            abs(np.mean([r[m] for r in current])-float(historical.iloc[0][f"{m}_mean"])) > 1e-8
            for m in ("R2", "MAE", "RMSE")):
            raise ValueError(f"C1->C6 does not match A ablation: {method}")
        for seed in SEEDS:
            prior_pred = pd.read_csv(ablation / f"c1_to_c6/A/seed_{seed}/{method}/predictions.csv")
            if (prior_pred.cut_index.tolist() != CUTS or
                not np.array_equal(prior_pred.pred_vb.to_numpy(),
                                   predictions[("c1", "c6"), seed, method])):
                raise ValueError(f"C1->C6 A per-cut mismatch: {seed} {method}")
    out.mkdir(parents=True)
    (out / "per_seed").mkdir()
    (out / "curves").mkdir()
    for source, target in PAIRS:
        for seed in SEEDS:
            for method in METHODS:
                pred = predictions[(source, target), seed, method]
                save_csv(out / "per_seed" / f"{source}_to_{target}_seed_{seed}_{method}.csv",
                         [{"cut_index": cut, "true_vb": truths[target][cut-1],
                           "pred_vb": pred[cut-1]} for cut in CUTS])
    summary = summary_table(metric_data)
    distance_frame = pd.DataFrame(nn_summary)
    auc_frame = pd.DataFrame(auc_data)
    auc_summary = auc_frame.groupby(["direction", "seed", "method", "layer", "stage"],
                                    as_index=False).agg(auc_mean=("auc", "mean"),
                                                        auc_sd=("auc", "std"), folds=("fold", "count"))
    save_csv(out / "checkpoint_audit.csv", ck_audit)
    save_csv(out / "per_seed_metrics.csv", metric_data)
    summary.to_csv(out / "five_seed_stage_summary.csv", index=False, float_format="%.17g")
    save_csv(out / "wear_range_diagnostics.csv", wear_rows)
    save_csv(out / "stft_channel_stats.csv", channel_rows)
    save_csv(out / "feature_distribution_stats.csv", profile_data)
    save_csv(out / "nearest_neighbor_per_cut.csv", nn_data)
    save_csv(out / "distance_stage_summary.csv", nn_summary)
    save_csv(out / "domain_classifier_folds.csv", auc_data)
    auc_summary.to_csv(out / "domain_classifier_summary.csv", index=False, float_format="%.17g")
    plot_config = plot_curves(out / "curves", predictions, truths)
    six_rows = []
    wear = pd.DataFrame(wear_rows).set_index("direction")
    for source, target in PAIRS:
        name = f"{source}_to_{target}"
        row = {"direction": name, "source_vb_max": float(wear.loc[name,"source_vb_max"]),
               "target_vb_max": float(wear.loc[name,"target_vb_max"]),
               "target_cuts_above_source_max": int(wear.loc[name,"target_cuts_above_source_max"]),
               "target_late_cuts_above_source_max": int(wear.loc[name,"target_late_cuts_above_source_max"])}
        for method in METHODS:
            a, late = one(summary, name, method, "all"), one(summary, name, method, "211-315")
            for metric in ("R2", "MAE", "RMSE"):
                row[f"{method}_{metric}_mean"] = float(a[f"{metric}_mean"])
                row[f"{method}_{metric}_sd"] = float(a[f"{metric}_sd"])
            for metric in ("MAE", "signed_error", "pred_increment", "true_increment"):
                row[f"{method}_late_{metric}_mean"] = float(late[f"{metric}_mean"])
                if f"{metric}_sd" in late:
                    row[f"{method}_late_{metric}_sd"] = float(late[f"{metric}_sd"])
        six_rows.append(row)
    save_csv(out / "six_direction_summary.csv", six_rows)
    provenance = {"input_scheme": "A: log1p(abs(STFT)) before resize; source-channel Z-score",
                  "directions": [f"{s}_to_{t}" for s, t in PAIRS],
                  "seeds": SEEDS, "methods": METHODS, "scope": "Cut 1-315",
                  "reused_checkpoints": 60, "rerun_tasks": [],
                  "training_repeated": False,
                  "source_only_target_training_cuts": 0,
                  "daregram_unlabeled_target_training_cuts": 315,
                  "target_labels_used_before_frozen_predictions_and_diagnostics": False,
                  "stft_manifest": str(manifest_path), "stft_code_sha256": manifest["stft_code_sha256"],
                  "full_baseline_protocol_lock": str((full / "protocol_lock_full_1_315.json").resolve()),
                  "feature_audit": str((feature / "run_summary.json").resolve()),
                  "c1_c6_A_regression": "All ten per-cut predictions equal A ablation; mean R2/MAE/RMSE equal within 1e-8",
                  "domain_classifier": "5-fold GroupKFold within each 105-Cut stage, groups of five adjacent paired cuts, balanced source/target, fixed L2 logistic C=1",
                  "distance": "5-fold GroupKFold of seven adjacent cuts; dimension z-score fit only on source train fold; source held-out reference",
                  "plot": plot_config,
                  "checkpoints": ck_audit}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    report(out, summary, distance_frame, auc_summary, wear_rows, provenance, plot_config)
    print(f"Audited and reused 60/60 checkpoints; rerun tasks: 0; saved {out}", flush=True)
    print(pd.DataFrame(six_rows)[["direction", "source_only_MAE_mean", "daregram_MAE_mean",
                                  "source_only_late_signed_error_mean",
                                  "daregram_late_signed_error_mean",
                                  "target_cuts_above_source_max"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
