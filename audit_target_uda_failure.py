"""Post-training, paired full-cut audit of the frozen five-seed PHM2010 runs.

Target wear is read only after all checkpoint, configuration, feature and
prediction preflight checks have completed. No training or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_single_source_pairs as base

TOOLS = ("c1", "c4", "c6")
SEEDS = range(42, 47)
METHODS = ("source_only", "daregram")
CUTS = list(range(1, 316))
COLS = [f"feature_{i:03d}" for i in range(512)]
SEGMENTS = {"full": (1, 315), "early": (1, 105), "middle": (106, 210), "late": (211, 315)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean_sd(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1))


def measures(truth, pred):
    error = pred - truth
    var = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / var) if var > 1e-8 else np.nan
    return {"R2": r2, "R2_low_variance": bool(var <= 1e-8),
            "true_vb_std": float(np.std(truth)), "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "signed_error": float(error.mean()), "underestimate_fraction": float(np.mean(error < 0))}


def feature_frame(path, tool):
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["tool_id", "cut_index", *COLS] or len(frame) != 315:
        raise ValueError(f"Feature schema/count mismatch: {path}")
    if frame.tool_id.tolist() != [tool] * 315 or frame.cut_index.tolist() != CUTS:
        raise ValueError(f"Feature cut mapping mismatch: {path}")
    if not np.isfinite(frame[COLS].to_numpy()).all():
        raise ValueError(f"Nonfinite features: {path}")
    return frame


def preflight(train, evaluated, features):
    manifest = read_json(train / "feature_cache" / "manifest.json")
    if manifest["cuts"] != CUTS or manifest["feature_shape"] != [315, 6, 128, 128]:
        raise ValueError("Training STFT cache does not cover 1..315")
    feat_run = read_json(features / "run_summary.json")
    if (feat_run["experiments_audited"] != 60 or
            not feat_run["all_paired_inputs_identical"] or feat_run["missing_experiments"]):
        raise ValueError("Historical feature audit incomplete or paired inputs differ")
    records = []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                pair = train / f"{source}_to_{target}" / f"seed_{seed}"
                config, audit = read_json(pair / "config.json"), read_json(pair / "audit.json")
                if ((config["source"], config["target"], config["seed"]) != (source, target, seed)
                        or config["source_cuts"] != CUTS or config["target_unlabeled_cuts"] != CUTS
                        or config["epochs"] != 50 or config["batch_size"] != 63
                        or config["lr"] != 0.001 or config["checkpoint_selection"] != "final epoch"
                        or config["regressor"] != "Linear(512, 1)"
                        or audit["target_label_reads_during_training"] != 0
                        or audit["source_only_training_target_cuts"] != []
                        or not audit["paired_initialization_verified"]
                        or not audit["paired_source_order_verified"]):
                    raise ValueError(f"Training protocol mismatch: {pair}")
                pair_configs = []
                pair_preds = []
                for method in METHODS:
                    folder = pair / method
                    cfg = read_json(folder / "config.json")
                    checkpoint = folder / "final.pth"
                    if (sha(checkpoint) != audit["checkpoint_sha256"][method]
                            or cfg["method"] != method or cfg["seed"] != seed
                            or cfg["actual_unlabeled_target_cuts"] != ([] if method == "source_only" else CUTS)
                            or cfg["source_feature_sha256"] != manifest["tools"][source]["feature_sha256"]
                            or cfg["target_feature_sha256"] != manifest["tools"][target]["feature_sha256"]):
                        raise ValueError(f"Checkpoint/config mismatch: {folder}")
                    pred_path = evaluated / f"{source}_to_{target}" / f"seed_{seed}" / method / "predictions.csv"
                    pred_cfg = read_json(pred_path.parent / "config.json")
                    pred = pd.read_csv(pred_path)
                    if (pred.columns.tolist() != ["cut_index", "true_vb", "pred_vb"]
                            or pred.cut_index.tolist() != CUTS or not np.isfinite(pred.pred_vb).all()
                            or pred_cfg["checkpoint_sha256"] != sha(checkpoint)
                            or pred_cfg["evaluation_cuts"] != CUTS):
                        raise ValueError(f"Full-cut prediction mismatch: {pred_path}")
                    pair_configs.append(cfg)
                    pair_preds.append(pred)
                    feat_dir = features / "experiments" / f"{source}_to_{target}" / f"seed_{seed}" / method
                    fa = read_json(feat_dir / "audit.json")
                    for role, tool in (("source", source), ("target", target)):
                        fpath = feat_dir / f"{role}_features.csv"
                        if sha(fpath) != fa["feature_stats"][role]["file_sha256"]:
                            raise ValueError(f"Feature checksum mismatch: {fpath}")
                        feature_frame(fpath, tool)
                    if fa["checkpoint_sha256"] != sha(checkpoint):
                        raise ValueError(f"Feature/checkpoint mismatch: {feat_dir}")
                    records.append({"source": source, "target": target, "seed": seed, "method": method,
                                    "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint),
                                    "training_config": str((folder / "config.json").resolve()),
                                    "prediction_csv": str(pred_path.resolve()),
                                    "source_features": str((feat_dir / "source_features.csv").resolve()),
                                    "target_features": str((feat_dir / "target_features.csv").resolve()),
                                    "feature_audit": str((feat_dir / "audit.json").resolve())})
                keys = ("source_normalization_mean", "source_normalization_std", "input", "stft_code_sha256",
                        "backbone", "regressor", "epochs", "batch_size", "lr", "source_wear_sha256")
                if any(pair_configs[0][key] != pair_configs[1][key] for key in keys):
                    raise ValueError(f"Paired preprocessing/model differs: {pair}")
                if not np.array_equal(pair_preds[0].true_vb, pair_preds[1].true_vb):
                    raise ValueError(f"Paired target labels differ: {pair}")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/five_seed_full_lifecycle"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/five_seed_feature_audit_20260925"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    args = parser.parse_args()
    train, evaluated, features, out = [p.resolve() for p in
                                      (args.train_root, args.eval_root, args.feature_root, args.out_root)]
    if out.exists():
        parser.error("Output directory already exists")
    provenance = preflight(train, evaluated, features)
    # Target wear is opened below, after all 60 final checkpoints and complete predictions are verified.
    labels = {tool: base.wear_labels(args.raw_root, tool).astype(np.float64) for tool in TOOLS}
    out.mkdir(parents=True)
    (out / "plots").mkdir()
    (out / "features_per_cut").mkdir()
    (out / "regressor_input_512").mkdir()
    provenance.to_csv(out / "provenance.csv", index=False)
    per_cut, segments, source_fit, training_log, feature_index = [], [], [], [], []
    for record in provenance.to_dict("records"):
        source, target, seed, method = (record[k] for k in ("source", "target", "seed", "method"))
        pred = pd.read_csv(record["prediction_csv"])
        truth = labels[target]
        if not np.allclose(pred.true_vb, truth, rtol=0, atol=1e-5):
            raise ValueError(f"Target labels disagree with raw wear CSV: {record['prediction_csv']}")
        source_max = float(labels[source].max())
        errors = pred.pred_vb.to_numpy() - truth
        beyond = truth > source_max
        for cut in CUTS:
            i = cut - 1
            per_cut.append({"source": source, "target": target, "seed": seed, "method": method,
                            "cut_index": cut, "true_vb": truth[i], "pred_vb": pred.pred_vb.iloc[i],
                            "signed_error": errors[i], "absolute_error": abs(errors[i]),
                            "above_source_train_max": bool(beyond[i]),
                            "source_train_max_vb": source_max,
                            "checkpoint_sha256": record["checkpoint_sha256"]})
        for segment, (first, last) in SEGMENTS.items():
            idx = slice(first - 1, last)
            metrics = measures(truth[idx], pred.pred_vb.to_numpy()[idx])
            segments.append({"source": source, "target": target, "seed": seed, "method": method,
                             "segment": segment, "first_cut": first, "last_cut": last,
                             "R2_low_variance_relative": bool(metrics["true_vb_std"] < .1 * np.std(truth)),
                             "source_train_max_vb": source_max, "target_max_vb": float(truth.max()),
                             "beyond_source_cut_count": int(beyond[idx].sum()),
                             "beyond_source_MAE": float(np.mean(np.abs(errors[idx][beyond[idx]]))) if beyond[idx].any() else np.nan,
                             "beyond_source_signed_error": float(np.mean(errors[idx][beyond[idx]])) if beyond[idx].any() else np.nan,
                             **metrics})
        # These are training-set reconstruction errors; the protocol has no held-out source validation split.
        fsource = feature_frame(Path(record["source_features"]), source)
        ckpt = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
        weight = ckpt["model"]["regressor.0.weight"].numpy().reshape(-1).astype(np.float64)
        bias = float(ckpt["model"]["regressor.0.bias"].item())
        src_pred = fsource[COLS].to_numpy(dtype=np.float64) @ weight + bias
        source_fit.append({"source": source, "target": target, "seed": seed, "method": method,
                           "evaluation_type": "source training cuts, in-sample; no held-out source validation in protocol",
                           **measures(labels[source], src_pred)})
        log_path = Path(record["checkpoint"]).parent / "run.log"
        pattern = re.compile(r"^(source_only|daregram) epoch=(\d+)/50 source_MSE=([\d.eE+-]+) gram=([\d.eE+-]+) source_order_sha256=([a-f0-9]{64}) target_unique=(\d+)$")
        history = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            hit = pattern.match(line)
            if hit:
                history.append({"epoch": int(hit[2]), "supervised_mse": float(hit[3]),
                                "domain_gram": float(hit[4]), "source_order_sha256": hit[5],
                                "target_unique": int(hit[6])})
        if [x["epoch"] for x in history] != list(range(1, 51)) or any(
                x["target_unique"] != (0 if method == "source_only" else 315) for x in history):
            raise ValueError(f"Incomplete training loss history: {log_path}")
        for row in history:
            training_log.append({"source": source, "target": target, "seed": seed, "method": method,
                                 "checkpoint_sha256": record["checkpoint_sha256"], **row})
        # One center STFT window per cut. Thus sample-level and cut-aggregated vectors are identical.
        ft = feature_frame(Path(record["target_features"]), target)
        feature_pred = ft[COLS].to_numpy(dtype=np.float64) @ weight + bias
        linear_replay_max_error = float(np.max(np.abs(feature_pred - pred.pred_vb.to_numpy())))
        if linear_replay_max_error > .01:
            raise ValueError(f"Feature/head reconstruction differs materially: {record['checkpoint']}")
        merged = pd.concat([fsource, ft], ignore_index=True)
        merged.insert(0, "domain", ["source"] * 315 + ["target"] * 315)
        merged.insert(3, "window_index", 0)
        merged.insert(4, "windows_in_cut", 1)
        name = f"{source}_to_{target}_seed_{seed}_{method}.csv"
        path = out / "features_per_cut" / name
        merged.to_csv(path, index=False, float_format="%.9g")
        regressor_path = out / "regressor_input_512" / name
        shutil.copyfile(path, regressor_path)
        feature_index.append({"source": source, "target": target, "seed": seed, "method": method,
                              "resnet18_512_source": record["source_features"],
                              "resnet18_512_target": record["target_features"],
                              "regressor_input_512": str(regressor_path.resolve()),
                              "aggregated_per_cut_512": str(path.resolve()),
                              "sample_to_cut": "one centered window per cut; window_index=0; arithmetic mean of one vector",
                              "shape_per_domain": "315x512", "checkpoint_sha256": record["checkpoint_sha256"],
                              "target_linear_replay_max_abs_vb_difference": linear_replay_max_error})
    cuts = pd.DataFrame(per_cut)
    seg = pd.DataFrame(segments)
    fit = pd.DataFrame(source_fit)
    cuts.to_csv(out / "per_cut_errors.csv", index=False)
    seg.to_csv(out / "segment_metrics.csv", index=False)
    fit.to_csv(out / "source_fit.csv", index=False)
    pd.DataFrame(training_log).to_csv(out / "training_epoch_losses.csv", index=False)
    pd.DataFrame(feature_index).to_csv(out / "feature_index.csv", index=False)
    seed_rows, direction_rows = [], []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            direction = seg[(seg.source == source) & (seg.target == target)]
            fits = fit[(fit.source == source) & (fit.target == target)]
            target_truth = labels[target]
            source_max = float(labels[source].max())
            beyond = target_truth > source_max
            for seed in SEEDS:
                pair = direction[direction.seed == seed]
                row = {"source": source, "target": target, "seed": seed,
                       "source_train_max_vb": source_max, "target_max_vb": float(target_truth.max()),
                       "beyond_source_cut_count": int(beyond.sum()),
                       "first_beyond_source_cut": int(np.flatnonzero(beyond)[0] + 1) if beyond.any() else np.nan}
                for method in METHODS:
                    full = pair[(pair.method == method) & (pair.segment == "full")].iloc[0]
                    late = pair[(pair.method == method) & (pair.segment == "late")].iloc[0]
                    sf = fits[(fits.seed == seed) & (fits.method == method)].iloc[0]
                    for key in ("MAE", "RMSE", "R2", "signed_error", "underestimate_fraction",
                                "beyond_source_MAE", "beyond_source_signed_error"):
                        row[f"{method}_{key}"] = full[key]
                    row[f"{method}_late_signed_error"] = late.signed_error
                    row[f"{method}_source_train_MAE"] = sf.MAE
                    row[f"{method}_source_train_RMSE"] = sf.RMSE
                for key in ("MAE", "RMSE", "R2"):
                    row[f"delta_{key}"] = row[f"daregram_{key}"] - row[f"source_only_{key}"]
                seed_rows.append(row)
            subset = pd.DataFrame(seed_rows[-5:])
            summary = {"source": source, "target": target, "n_seeds": 5,
                       "source_train_max_vb": source_max, "target_max_vb": float(target_truth.max()),
                       "beyond_source_cut_count": int(beyond.sum()),
                       "first_beyond_source_cut": int(np.flatnonzero(beyond)[0] + 1) if beyond.any() else np.nan,
                       "daregram_worse_MAE_seeds": int((subset.delta_MAE > 0).sum())}
            for key in subset.columns:
                if key in ("source", "target", "seed", "source_train_max_vb", "target_max_vb",
                           "beyond_source_cut_count", "first_beyond_source_cut"):
                    continue
                summary[f"{key}_mean"], summary[f"{key}_sd"] = mean_sd(subset[key]) if subset[key].notna().all() else (np.nan, np.nan)
            direction_rows.append(summary)
            pc = cuts[(cuts.source == source) & (cuts.target == target)]
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
            axes[0].plot(CUTS, target_truth, color="black", label="true VB", linewidth=2)
            for method, color in (("source_only", "tab:blue"), ("daregram", "tab:orange")):
                matrix = np.stack([pc[(pc.method == method) & (pc.seed == seed)].pred_vb.to_numpy() for seed in SEEDS])
                avg, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
                axes[0].plot(CUTS, avg, color=color, label=f"{method} mean")
                axes[0].fill_between(CUTS, avg - sd, avg + sd, color=color, alpha=.15)
                axes[1].plot(CUTS, avg - target_truth, color=color, label=f"{method} signed error")
            axes[0].axhline(source_max, color="grey", linestyle="--", label="source training max VB")
            axes[1].axhline(0, color="black", linewidth=.8)
            if beyond.any():
                for ax in axes:
                    ax.axvline(int(np.flatnonzero(beyond)[0] + 1), color="red", linestyle=":", label="first beyond source max")
            axes[0].set(ylabel="VB", title=f"{source.upper()} to {target.upper()} | five paired seeds | cuts 1–315")
            axes[1].set(xlabel="target cut", ylabel="pred − true VB")
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
            fig.savefig(out / "plots" / f"{source}_to_{target}.png", dpi=170)
            plt.close(fig)
    pd.DataFrame(seed_rows).to_csv(out / "seed_summary.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(out / "direction_summary.csv", index=False)
    print(f"Saved audit to {out}: 60 runs, {len(cuts)} per-cut rows")


if __name__ == "__main__":
    main()
