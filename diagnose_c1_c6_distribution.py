"""Read-only C1->C6 full-cut diagnostic of frozen five-seed models.

The first phase uses STFT inputs and frozen 512-D embeddings without C6 labels.
C6 wear is opened only after all distribution metrics have been computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_single_source_pairs import file_hash


SEEDS = range(42, 47)
METHODS = ("source_only", "daregram")
STAGES = ((1, 105), (106, 210), (211, 315))
CUTS = list(range(1, 316))
CHANNELS = ("Fx", "Fy", "Fz", "Vx", "Vy", "Vz")
QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def read_embedding(path, tool):
    frame = pd.read_csv(path)
    cols = [f"feature_{i:03d}" for i in range(512)]
    if list(frame.columns) != ["tool_id", "cut_index", *cols]:
        raise ValueError(f"Wrong feature schema: {path}")
    if frame.tool_id.tolist() != [tool] * 315 or frame.cut_index.tolist() != CUTS:
        raise ValueError(f"Wrong feature cut order: {path}")
    x = frame[cols].to_numpy(np.float64)
    if x.shape != (315, 512) or not np.isfinite(x).all():
        raise ValueError(f"Bad embedding: {path}")
    return x


def profile(x):
    norms = np.linalg.norm(x, axis=1)
    return {
        "n": len(x), "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "norm_p01": float(np.quantile(norms, .01)), "norm_p50": float(np.median(norms)),
        "norm_p99": float(np.quantile(norms, .99)),
        "element_mean": float(x.mean()), "element_std": float(x.std()),
        **{f"element_p{round(q*100):02d}": float(np.quantile(x, q)) for q in QUANTILES},
    }


def stft_descriptors(x):
    """48 fixed, label-free descriptors: four moments and four frequency bands/channel."""
    a = np.asarray(x, dtype=np.float32)
    desc = []
    for c in range(6):
        z = a[:, c]
        desc += [z.mean(axis=(1, 2)), z.std(axis=(1, 2)),
                 np.quantile(z, .9, axis=(1, 2)), np.quantile(z, .99, axis=(1, 2))]
        desc += [band.mean(axis=(1, 2)) for band in np.array_split(z, 4, axis=1)]
    return np.stack(desc, axis=1).astype(np.float64)


def distance_and_oor(xs, xt, stage_indices, seed, method, layer):
    """Five matched-cut folds; C1-only fit, C1 held-out distance reference."""
    rows, oor_rows = [], []
    groups = np.arange(315) // 7
    folds = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(folds.split(xs, groups=groups), 1):
        mu = xs[train].mean(axis=0)
        sigma = xs[train].std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        src = (xs[train] - mu) / sigma
        qs_lo = np.quantile(src, .01, axis=0)
        qs_hi = np.quantile(src, .99, axis=0)
        for domain, points in (("c1_holdout", xs[test]), ("c6", xt[test])):
            z = (points - mu) / sigma
            # Small dense matrices; the source index always excludes C1 test cuts.
            dd = np.maximum(0, (z*z).sum(1)[:, None] + (src*src).sum(1)[None, :]
                            - 2 * (z @ src.T))
            closest = dd.argmin(axis=1)
            distances = np.sqrt(dd[np.arange(len(z)), closest] / xs.shape[1])
            for i, cut_row in enumerate(test):
                rows.append({"seed": seed, "method": method, "layer": layer,
                             "domain": domain, "fold": fold, "cut_index": int(cut_row+1),
                             "stage": stage_indices[cut_row], "distance": float(distances[i]),
                             "nearest_c1_cut": int(train[closest[i]]+1)})
                if domain == "c6":
                    oor_rows.append({"seed": seed, "method": method, "layer": layer,
                                     "fold": fold, "cut_index": int(cut_row+1),
                                     "stage": stage_indices[cut_row],
                                     "outside_dim_fraction": float(np.mean((z[i] < qs_lo) | (z[i] > qs_hi)))})
    return rows, oor_rows


def auc_folds(xs, xt, seed, method, layer):
    records = []
    for lo, hi in STAGES:
        take = np.arange(lo-1, hi)
        # Both domains contribute exactly one observation per cut; fold groups
        # keep paired and nearby cuts on the same side of the split.
        x = np.concatenate((xs[take], xt[take]))
        y = np.r_[np.zeros(len(take), dtype=int), np.ones(len(take), dtype=int)]
        groups = np.tile((take-(lo-1)) // 5, 2)
        for fold, (train, test) in enumerate(GroupKFold(5).split(x, y, groups), 1):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(x[train], y[train])
            score = roc_auc_score(y[test], clf.predict_proba(x[test])[:, 1])
            records.append({"seed": seed, "method": method, "layer": layer,
                            "stage": f"{lo}-{hi}", "fold": fold,
                            "n_c1_train": int((y[train] == 0).sum()),
                            "n_c6_train": int((y[train] == 1).sum()),
                            "n_c1_test": int((y[test] == 0).sum()),
                            "n_c6_test": int((y[test] == 1).sum()),
                            "auc": float(score)})
    return records


def plot_curves(out, truth, preds):
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.plot(CUTS, truth, color="#242424", lw=2.25, label="True VB", zorder=4)
    for method, color, name in (("source_only", "#2563a6", "Source-only"),
                                ("daregram", "#db7026", "DARE-GRAM")):
        values = np.stack([preds[method][seed] for seed in SEEDS])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax.fill_between(CUTS, mean-sd, mean+sd, color=color, alpha=.16, linewidth=0,
                        label=f"{name} ±1 seed SD", zorder=1)
        ax.plot(CUTS, mean, color=color, lw=1.75, label=f"{name} mean", zorder=3)
    ax.set(title="C1 → C6 | five-seed raw predictions | Cut 1–315",
           xlabel="Target cut index", ylabel="Flank wear (VB)", xlim=(1, 315))
    ax.grid(alpha=.22, linewidth=.7)
    ax.legend(loc="lower right", ncol=2, fontsize=8.7, framealpha=.9)
    fig.savefig(out / "raw_prediction_curves.png", dpi=180)
    plt.close(fig)


def make_plots(out, stage_pred, feature_summary, nn_summary, input_rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    stages = [f"{a}-{b}" for a, b in STAGES]
    for i, metric in enumerate(("MAE", "signed_error", "pred_increment")):
        for method, color in (("source_only", "#2563a6"), ("daregram", "#db7026")):
            sub = stage_pred[(stage_pred.method == method) & (stage_pred.seed == "five_seed_mean")]
            vals = [float(sub.loc[sub.stage == s, metric].iloc[0]) for s in stages]
            axes[i].plot(stages, vals, marker="o", label=method, color=color)
        if metric == "pred_increment":
            sub = stage_pred[(stage_pred.method == "source_only") & (stage_pred.seed == "five_seed_mean")]
            axes[i].plot(stages, [float(sub.loc[sub.stage == s, "true_increment"].iloc[0]) for s in stages],
                         marker="s", label="true", color="black")
        axes[i].set(title=metric.replace("_", " "), xlabel="C6 Cut stage")
        axes[i].grid(alpha=.2)
    axes[0].set_ylabel("VB")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.savefig(out / "stage_prediction_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    for ax, layer in zip(axes, ("stft_48d", "resnet_512d")):
        sub = feature_summary[(feature_summary.layer == layer) &
                              (feature_summary.method.isin(["source_only", "daregram"]))]
        if layer == "stft_48d":
            sub = sub[sub.method == "source_only"]
        for method, color in (("source_only", "#2563a6"), ("daregram", "#db7026")):
            for domain, style in (("c1", "--"), ("c6", "-")):
                part = sub[(sub.method == method) & (sub.domain == domain)]
                if len(part):
                    vals = [part.loc[part.stage == s, "norm_mean"].mean() for s in stages]
                    ax.plot(stages, vals, color=color, ls=style, marker="o",
                            label=f"{method} {domain}")
        ax.set(title=layer + " norm", xlabel="Cut stage")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / "feature_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    for ax, layer in zip(axes, ("stft_48d", "resnet_512d")):
        sub = nn_summary[nn_summary.layer == layer]
        for method, color in (("source_only", "#2563a6"), ("daregram", "#db7026")):
            if layer == "stft_48d" and method == "daregram":
                continue
            part = sub[sub.method == method]
            ax.plot(stages, [part.loc[part.stage == s, "c6_distance_median"].mean() for s in stages],
                    color=color, marker="o", label=f"{method} C6")
            ax.plot(stages, [part.loc[part.stage == s, "c1_holdout_distance_median"].mean() for s in stages],
                    color=color, marker="s", ls="--", label=f"{method} C1 holdout")
        ax.set(title=layer + " nearest C1 train", xlabel="Cut stage", ylabel="scaled RMS distance")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / "nearest_neighbor_distances.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    channel_frame = pd.DataFrame(input_rows)
    channel_frame = channel_frame[channel_frame.representation == "model_input_zscore"]
    for channel in ("Fx", "Fy", "Fz", "Vx", "Vy", "Vz"):
        part = channel_frame[(channel_frame.channel == channel) & (channel_frame.domain == "c6")]
        source = channel_frame[(channel_frame.channel == channel) & (channel_frame.domain == "c1")]
        axes[0].plot(stages, [float(part.loc[part.stage == s, "p99"].iloc[0]) /
                             float(source.loc[source.stage == s, "p99"].iloc[0]) for s in stages],
                     marker="o", label=channel)
    axes[0].axhline(1, color="black", lw=.8, ls="--")
    axes[0].set(title="C6 / C1 input p99 (same C1 Z-score)", xlabel="Cut stage", ylabel="ratio")
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    for method, color in (("source_only", "#2563a6"), ("daregram", "#db7026")):
        part = nn_summary[(nn_summary.method == method) & (nn_summary.layer == "resnet_512d")]
        axes[1].plot(stages, [part.loc[part.stage == s,
                                       "c6_outside_c1_full_01_99_dimension_fraction"].mean()
                              for s in stages], marker="o", label=method, color=color)
    axes[1].set(title="512-D C6 outside full C1 [1%,99%]", xlabel="Cut stage", ylabel="dimension fraction")
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out / "input_and_feature_support.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path("artifacts/c1_c6_distribution_diagnostic_20260926"))
    args = ap.parse_args()
    out = args.out_root.resolve()
    source_root = Path("artifacts/five_seed_paired").resolve()
    full_root = Path("artifacts/full_1_315_baseline_zscore_20260925").resolve()
    feature_root = Path("artifacts/five_seed_feature_audit_20260925").resolve()
    prediction_root = Path("artifacts/norm_comparison_20260925/zscore").resolve()
    if out.exists() or any(out == p or p in out.parents for p in
                           (source_root, full_root, feature_root, prediction_root)):
        ap.error("Output must be a new independent directory")
    manifest_path = source_root / "feature_cache/manifest.json"
    manifest = read_json(manifest_path)
    import data_sampling
    if (manifest["cuts"] != CUTS or manifest["feature_shape"] != [315, 6, 128, 128] or
        file_hash(Path(data_sampling.__file__)) != manifest["stft_code_sha256"]):
        raise ValueError("STFT manifest/code mismatch")
    audit_index = {(r["source"], r["target"], r["seed"], r["method"]): r for r in
                   read_json(feature_root / "file_index.json")}
    baseline = pd.read_csv(full_root / "checkpoint_audit_full_1_315.csv")
    baseline = baseline[(baseline.source == "c1") & (baseline.target == "c6")]
    if len(baseline) != 10:
        raise ValueError("Missing full-cut baseline checkpoint audit rows")
    cache = {}
    for tool in ("c1", "c6"):
        path = source_root / f"feature_cache/{tool}_stft.npy"
        if file_hash(path) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"STFT cache changed: {path}")
        cache[tool] = np.load(path, mmap_mode="r", allow_pickle=False)
        if cache[tool].shape != (315, 6, 128, 128) or cache[tool].dtype != np.float32:
            raise ValueError(f"Bad STFT cache: {path}")
    mean = cache["c1"].mean(axis=(0, 2, 3)).astype(np.float32)
    std = (cache["c1"].std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    stage_names = np.asarray([next(f"{a}-{b}" for a, b in STAGES if a <= cut <= b)
                              for cut in CUTS])
    # Mirror existing source-channel Z-score in float32, using only C1 inputs.
    inputs = {tool: ((np.asarray(cache[tool]) - mean[None, :, None, None]) /
                     (std[None, :, None, None] + np.float32(1e-8))).astype(np.float32)
              for tool in ("c1", "c6")}
    stft_desc = {tool: stft_descriptors(inputs[tool]) for tool in ("c1", "c6")}
    feature_summary, dimension_rows, input_rows = [], [], []
    nn_rows, oor_rows, full_oor_rows, auc_rows, provenance = [], [], [], [], []
    embeddings = {}
    with threadpool_limits(limits=4):
        # STFT descriptors are shared across all ten models.
        r, o = distance_and_oor(stft_desc["c1"], stft_desc["c6"], stage_names,
                                "shared", "source_only", "stft_48d")
        nn_rows.extend(r); oor_rows.extend(o)
        auc_rows.extend(auc_folds(stft_desc["c1"], stft_desc["c6"],
                                  "shared", "source_only", "stft_48d"))
        for lo, hi in STAGES:
            for tool in ("c1", "c6"):
                part = inputs[tool][lo-1:hi]
                for representation, cube in (("raw_log_stft", cache[tool][lo-1:hi]),
                                             ("model_input_zscore", part)):
                    flat = np.asarray(cube).reshape(hi-lo+1, -1)
                    rms = np.sqrt(np.mean(flat.astype(np.float64)**2, axis=1))
                    feature_summary.append({"seed": "shared", "method": "source_only", "layer": representation,
                                            "stage": f"{lo}-{hi}", "domain": tool,
                                            "n": len(flat), "norm_mean": rms.mean(), "norm_std": rms.std(),
                                            "norm_p01": np.quantile(rms, .01),
                                            "norm_p50": np.median(rms),
                                            "norm_p99": np.quantile(rms, .99),
                                            "element_mean": flat.mean(), "element_std": flat.std(),
                                            **{f"element_p{round(q*100):02d}": np.quantile(flat, q)
                                               for q in QUANTILES}})
                    for c, channel in enumerate(CHANNELS):
                        z = np.asarray(cube[:, c], dtype=np.float64)
                        input_rows.append({"stage": f"{lo}-{hi}", "domain": tool,
                                           "representation": representation, "channel": channel,
                                           "mean": z.mean(), "std": z.std(),
                                           "p01": np.quantile(z, .01), "p10": np.quantile(z, .1),
                                           "p50": np.quantile(z, .5), "p90": np.quantile(z, .9),
                                           "p99": np.quantile(z, .99),
                                           "cut_rms_mean": np.sqrt(np.mean(z*z, axis=(1, 2))).mean()})
                feature_summary.append({"seed": "shared", "method": "source_only", "layer": "stft_48d",
                                        "stage": f"{lo}-{hi}", "domain": tool,
                                        **profile(stft_desc[tool][lo-1:hi])})
        # Full C1 source training support; unlike the NN reference, this uses all 315 C1 cuts.
        for layer, xs, xt, seed, method in (("stft_48d", stft_desc["c1"], stft_desc["c6"],
                                             "shared", "source_only"),):
            low, high = np.quantile(xs, [.01, .99], axis=0)
            outside = (xt < low) | (xt > high)
            for i in range(315):
                full_oor_rows.append({"seed": seed, "method": method, "layer": layer,
                                      "cut_index": i+1, "stage": stage_names[i],
                                      "outside_dim_fraction": float(outside[i].mean())})
        for seed in SEEDS:
            for method in METHODS:
                folder = source_root / f"c1_to_c6/seed_{seed}/{method}"
                cfg = read_json(folder / "config.json")
                audit_path = feature_root / f"experiments/c1_to_c6/seed_{seed}/{method}/audit.json"
                audit = read_json(audit_path)
                row = baseline[(baseline.seed == seed) & (baseline.method == method)]
                if len(row) != 1:
                    raise ValueError(f"Missing baseline row: {seed} {method}")
                row = row.iloc[0]
                checkpoint = folder / "final.pth"
                ck_hash = file_hash(checkpoint)
                if (cfg["source"] != "c1" or cfg["target"] != "c6" or cfg["seed"] != seed or
                    cfg["method"] != method or cfg["epochs"] != 50 or
                    cfg["backbone"] != "ResNet18" or cfg["regressor"] != "Linear(512, 1)" or
                    cfg["source_cuts"] != CUTS or cfg["target_unlabeled_cuts"] != CUTS or
                    cfg["target_labels_training_reads"] != 0 or
                    cfg["actual_unlabeled_target_cuts"] != (CUTS if method == "daregram" else []) or
                    cfg["source_feature_sha256"] != manifest["tools"]["c1"]["feature_sha256"] or
                    cfg["target_feature_sha256"] != manifest["tools"]["c6"]["feature_sha256"] or
                    not np.array_equal(np.asarray(cfg["source_normalization_mean"], np.float32), mean) or
                    not np.array_equal(np.asarray(cfg["source_normalization_std"], np.float32), std) or
                    ck_hash != audit["checkpoint_sha256"] or ck_hash != row.checkpoint_sha256 or
                    Path(row.checkpoint).resolve() != checkpoint.resolve() or
                    row.scope != "full_1_315" or int(row.prediction_count) != 315):
                    raise ValueError(f"Frozen run provenance mismatch: {folder}")
                full_folder = prediction_root / f"c1_to_c6/seed_{seed}/{method}"
                full_cfg = read_json(full_folder / "config.json")
                pred_file = full_folder / "predictions.csv"
                if (file_hash(full_folder / "final.pth") != ck_hash or
                    full_cfg["evaluation_cuts"] != CUTS or full_cfg["norm_method"] != "zscore" or
                    full_cfg["source_normalization_parameters"] !=
                    {"mean": cfg["source_normalization_mean"], "std": cfg["source_normalization_std"]} or
                    Path(row.original_prediction_file).resolve() != pred_file.resolve() or
                    row.original_prediction_sha256 != file_hash(pred_file)):
                    raise ValueError(f"Full-cut prediction provenance mismatch: {pred_file}")
                feats = {}
                for role, tool in (("source", "c1"), ("target", "c6")):
                    path = feature_root / f"experiments/c1_to_c6/seed_{seed}/{method}/{role}_features.csv"
                    if file_hash(path) != audit["feature_stats"][role]["file_sha256"]:
                        raise ValueError(f"Feature CSV changed: {path}")
                    feats[tool] = read_embedding(path, tool)
                embeddings[method, seed] = feats
                source_all, target_all = feats["c1"], feats["c6"]
                low, high = np.quantile(source_all, [.01, .99], axis=0)
                outside = (target_all < low) | (target_all > high)
                for i in range(315):
                    full_oor_rows.append({"seed": seed, "method": method, "layer": "resnet_512d",
                                          "cut_index": i+1, "stage": stage_names[i],
                                          "outside_dim_fraction": float(outside[i].mean())})
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if state["epoch"] != 50 or state["model"]["regressor.0.weight"].shape != (1, 512):
                    raise ValueError(f"Wrong model head: {checkpoint}")
                head_w = state["model"]["regressor.0.weight"].numpy().reshape(-1).astype(np.float64)
                head_b = float(state["model"]["regressor.0.bias"].numpy()[0])
                provenance.append({"seed": seed, "method": method, "checkpoint": str(checkpoint),
                                   "checkpoint_sha256": ck_hash, "training_config": str(folder / "config.json"),
                                   "full_prediction": str(pred_file), "full_prediction_sha256": file_hash(pred_file),
                                   "feature_audit": str(audit_path), "head_weight_norm": np.linalg.norm(head_w),
                                   "head_bias": head_b})
                for lo, hi in STAGES:
                    source, target = feats["c1"][lo-1:hi], feats["c6"][lo-1:hi]
                    for tool, part in (("c1", source), ("c6", target)):
                        feature_summary.append({"seed": seed, "method": method, "layer": "resnet_512d",
                                                "stage": f"{lo}-{hi}", "domain": tool, **profile(part)})
                    # Per-dimension raw quantiles make overlap and scale changes inspectable.
                    for dim in range(512):
                        sq = np.quantile(source[:, dim], QUANTILES)
                        tq = np.quantile(target[:, dim], QUANTILES)
                        dimension_rows.append({"seed": seed, "method": method, "stage": f"{lo}-{hi}",
                                               "dimension": dim,
                                               **{f"c1_p{round(q*100):02d}": sq[k] for k, q in enumerate(QUANTILES)},
                                               **{f"c6_p{round(q*100):02d}": tq[k] for k, q in enumerate(QUANTILES)},
                                               "median_delta": tq[2]-sq[2],
                                               "iqr90_delta": (tq[3]-tq[1])-(sq[3]-sq[1]),
                                               "c1_full_p01": low[dim], "c1_full_p99": high[dim],
                                               "c6_stage_outside_c1_full_fraction": outside[lo-1:hi, dim].mean(),
                                               "c1_full_mean": source_all[:, dim].mean(),
                                               "c1_full_std": source_all[:, dim].std(),
                                               "head_weight": head_w[dim]})
                r, o = distance_and_oor(feats["c1"], feats["c6"], stage_names,
                                        seed, method, "resnet_512d")
                nn_rows.extend(r); oor_rows.extend(o)
                auc_rows.extend(auc_folds(feats["c1"], feats["c6"], seed, method, "resnet_512d"))
                print(f"Distribution diagnostics: seed {seed} {method}", flush=True)

    # All label-free statistics are complete. Read wear and stored predictions now.
    import run_single_source_pairs as base
    raw_root = Path(manifest["raw_root"])
    y_source = base.wear_labels(raw_root, "c1")
    y_target = base.wear_labels(raw_root, "c6")
    preds = {method: {} for method in METHODS}
    for item in provenance:
        seed, method = item["seed"], item["method"]
        frame = pd.read_csv(item["full_prediction"])
        if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"] or frame.cut_index.tolist() != CUTS:
            raise ValueError(f"Full prediction cuts/schema mismatch: {item['full_prediction']}")
        if not np.array_equal(frame.true_vb.to_numpy(np.float32), y_target):
            raise ValueError(f"C6 true wear mismatch: {item['full_prediction']}")
        pred = frame.pred_vb.to_numpy(np.float64)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite prediction")
        state = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)["model"]
        w = state["regressor.0.weight"].numpy().reshape(-1).astype(np.float64)
        b = float(state["regressor.0.bias"].numpy()[0])
        reconstructed = embeddings[method, seed]["c6"] @ w + b
        item["prediction_head_max_abs_delta"] = float(np.max(np.abs(reconstructed-pred)))
        if item["prediction_head_max_abs_delta"] > 1e-3:
            raise ValueError(f"Features/head do not reproduce predictions: {seed} {method}")
        preds[method][seed] = pred
    pred_by_cut = pd.DataFrame({"cut_index": CUTS, "true_vb": y_target})
    stage_pred = []
    for method in METHODS:
        values = np.stack([preds[method][seed] for seed in SEEDS])
        for seed, vector in [*((str(seed), preds[method][seed]) for seed in SEEDS),
                             ("five_seed_mean", values.mean(0))]:
            if seed != "five_seed_mean":
                pred_by_cut[f"{method}_seed_{seed}"] = vector
            else:
                pred_by_cut[f"{method}_mean"] = vector
                pred_by_cut[f"{method}_seed_sd"] = values.std(0, ddof=1)
            for lo, hi in STAGES:
                p, y = vector[lo-1:hi], y_target[lo-1:hi]
                stage_pred.append({"method": method, "seed": seed, "stage": f"{lo}-{hi}",
                                   "n": len(p), "MAE": np.mean(np.abs(p-y)),
                                   "signed_error": np.mean(p-y),
                                   "pred_increment": p[-1]-p[0], "true_increment": y[-1]-y[0],
                                   "pred_max": np.max(p), "pred_mean": np.mean(p),
                                   "true_mean": np.mean(y), "true_max": np.max(y),
                                   "near_c1_upper_10vb_fraction": np.mean(np.abs(p-y_source.max()) <= 10)})
    nn = pd.DataFrame(nn_rows)
    oor = pd.DataFrame(oor_rows)
    full_oor = pd.DataFrame(full_oor_rows)
    nn_summary = []
    for (seed, method, layer, stage), g in nn.groupby(["seed", "method", "layer", "stage"]):
        s = g[g.domain == "c1_holdout"].distance.to_numpy()
        t = g[g.domain == "c6"].distance.to_numpy()
        og = oor[(oor.seed == seed) & (oor.method == method) &
                 (oor.layer == layer) & (oor.stage == stage)]
        fg = full_oor[(full_oor.seed == seed) & (full_oor.method == method) &
                      (full_oor.layer == layer) & (full_oor.stage == stage)]
        nn_summary.append({"seed": seed, "method": method, "layer": layer, "stage": stage,
                           "c1_holdout_distance_median": np.median(s),
                           "c1_holdout_distance_p95": np.quantile(s, .95),
                           "c6_distance_median": np.median(t),
                           "c6_distance_p95": np.quantile(t, .95),
                           "c6_over_holdout_median_ratio": np.median(t)/np.median(s),
                           "c6_above_c1_holdout_p95_fraction": np.mean(t > np.quantile(s, .95)),
                           "c6_outside_c1_full_01_99_dimension_fraction": fg.outside_dim_fraction.mean(),
                           "c6_outside_c1_fold_01_99_dimension_fraction": og.outside_dim_fraction.mean()})
    nn_summary = pd.DataFrame(nn_summary)
    auc = pd.DataFrame(auc_rows)
    auc_summary = auc.groupby(["seed", "method", "layer", "stage"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), folds=("fold", "count"))
    # Nearest-source labels are a final diagnostic only; they are never used above.
    matched = nn[(nn.domain == "c6") & (nn.layer == "resnet_512d")].copy()
    matched["nearest_c1_true_vb"] = y_source[matched.nearest_c1_cut.to_numpy(int)-1]
    matched["c6_true_vb"] = y_target[matched.cut_index.to_numpy(int)-1]
    matched["c6_minus_nearest_c1_vb"] = matched.c6_true_vb-matched.nearest_c1_true_vb

    out.mkdir(parents=True)
    save_csv(out / "predictions_by_cut.csv", pred_by_cut)
    save_csv(out / "stage_prediction_stats.csv", stage_pred)
    save_csv(out / "stft_channel_stats.csv", input_rows)
    save_csv(out / "feature_distribution_stats.csv", feature_summary)
    save_csv(out / "feature_dimension_quantiles.csv", dimension_rows)
    save_csv(out / "nearest_neighbor_distances.csv", nn_rows)
    save_csv(out / "nearest_neighbor_stage_summary.csv", nn_summary)
    save_csv(out / "outside_source_interval_per_cut.csv", oor_rows)
    save_csv(out / "outside_full_c1_training_interval_per_cut.csv", full_oor_rows)
    save_csv(out / "domain_classifier_folds.csv", auc_rows)
    save_csv(out / "domain_classifier_summary.csv", auc_summary)
    save_csv(out / "nearest_source_wear_comparison.csv", matched)
    (out / "provenance.json").write_text(json.dumps({
        "protocol": "C1 labeled 1-315; C6 unlabeled 1-315 for DARE-GRAM; frozen final checkpoints",
        "evaluation": "C6 1-315", "seeds": list(SEEDS), "methods": METHODS,
        "stft_cache_manifest": str(manifest_path), "stft_code_sha256": manifest["stft_code_sha256"],
        "stft_cache_sha256": {k: manifest["tools"][k]["feature_sha256"] for k in cache},
        "preprocessing": read_json(feature_root / "experiments/c1_to_c6/seed_42/source_only/audit.json")["preprocessing"],
        "c1_channel_mean": mean.tolist(), "c1_channel_std": std.tolist(),
        "c1_wear_csv": str(raw_root / "c1_wear.csv"),
        "c6_wear_csv": str(raw_root / "c6_wear.csv"),
        "checkpoints": provenance,
        "diagnostic_design": {
            "standardizer_and_1_99_interval": "C1 training folds only; no C6 label",
            "full_source_1_99_interval": "All 315 C1 training cuts, dimensionwise, no C6 label",
            "nearest_neighbor": "Euclidean RMS after C1 train-fold per-dimension z-score; C1 held-out comparator",
            "nn_split": "5 GroupKFold folds of 7-cut groups, paired C1/C6 cut test assignment",
            "domain_auc": "5 GroupKFold folds of 5-cut groups within each stage, balanced paired cuts, fixed L2 logistic C=1",
            "stft_classifier_input": "48 fixed descriptors: mean,std,p90,p99 and four frequency-band means per channel",
            "target_label_use": "Read only after all distribution analyses; residuals, metrics, and nearest-source wear comparison"
        }}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    plot_curves(out, y_target, preds)
    make_plots(out, pd.DataFrame(stage_pred), pd.DataFrame(feature_summary), nn_summary, input_rows)
    print(f"Saved C1->C6 diagnostic: {out}")
    print(pd.DataFrame(stage_pred).query("seed == 'five_seed_mean'")[
        ["method", "stage", "MAE", "signed_error", "pred_increment", "true_increment", "pred_max"]].to_string(index=False))
    print(nn_summary[nn_summary.layer == "resnet_512d"].groupby(["method", "stage"])[
        ["c6_over_holdout_median_ratio", "c6_outside_c1_full_01_99_dimension_fraction"]].mean().to_string())
    print(auc_summary[auc_summary.layer == "resnet_512d"].groupby(["method", "stage"]).auc_mean.mean().to_string())


if __name__ == "__main__":
    main()
