"""Five-seed full-cut trend ablation and report-only physical prediction audit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_five_seed_pairs as five
import run_single_source_pairs as base
from trend_physics import physical_audit, source_delta


METHODS = ("source_only", "daregram")
CUTS = base.ALL_CUTS


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/trend_physics_full_1_315_20260925"))
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/five_seed_paired/feature_cache"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--trend-lambda", type=float, default=0.0)
    parser.add_argument("--source", choices=base.TOOLS)
    parser.add_argument("--target", choices=base.TOOLS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(five.SEEDS))
    parser.add_argument("--mark-declines", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (args.source is None) != (args.target is None) or args.source == args.target and args.source is not None:
        parser.error("Specify distinct source and target, or neither for all directions")
    if not np.isfinite(args.trend_lambda) or args.trend_lambda < 0 or not set(args.seeds) <= set(five.SEEDS):
        parser.error("Invalid weight or seed")
    if args.out_root.resolve() == args.baseline_root.resolve():
        parser.error("Output must differ from frozen baseline")
    return args


def cache(cache_root, manifest, tool):
    path = cache_root / f"{tool}_stft.npy"
    if base.file_hash(path) != manifest["tools"][tool]["feature_sha256"]:
        raise ValueError(f"Cache digest mismatch: {path}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (315, 6, 128, 128) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"Invalid STFT cache: {path}")
    return x


def normalize(x, mean, std):
    result = ((x - mean[None, :, None, None]) /
              (std[None, :, None, None] + 1e-8)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized features")
    return result


def plot_curve(path, source, target, seed, method, cuts, truth, pred, audit, mark):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(cuts, truth, color="black", lw=1.5, label="True VB")
    ax.plot(cuts, pred, color="#2563a6", lw=1.3, label="Raw prediction")
    if mark:
        bad = audit.loc[audit.decline_exceeds_delta, "cut_index"].to_numpy(int) - 1
        ax.scatter(cuts[bad], pred[bad], color="#c0263d", s=15, label="Decline > source δ", zorder=4)
    ax.set(xlabel="Cut", ylabel="VB", title=f"{source.upper()}→{target.upper()} seed {seed}: {method}")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_one(args, manifest, source, target, seed):
    folder = args.out_root / f"{source}_to_{target}" / f"seed_{seed}"
    if (folder / "complete.json").exists():
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if cfg["trend_lambda"] != args.trend_lambda:
            raise ValueError(f"Completed run has different weight: {folder}")
        logging.info("Already completed: %s", folder)
        return
    if folder.exists():
        raise FileExistsError(f"Incomplete run; inspect before rerun: {folder}")
    folder.mkdir(parents=True)
    for method in METHODS:
        (folder / method).mkdir()
    device = torch.device(args.device)
    raw_source = cache(args.cache_root, manifest, source)
    mean = raw_source.mean(axis=(0, 2, 3)).astype(np.float32)
    std = (raw_source.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    xs = normalize(raw_source, mean, std)
    del raw_source
    ys = base.wear_labels(args.raw_root, source)
    delta, source_edges = source_delta(ys, CUTS)
    config = {"source": source, "target": target, "seed": seed, "trend_lambda": args.trend_lambda,
              "delta_vb": delta, "delta_method": "0.25 times median absolute adjacent source training-label VB change over 314 pairs; fixed before target evaluation",
              "delta_source_adjacent_pairs": source_edges,
              "prediction_and_delta_units": "original flank wear VB; no label scaling/inverse transform",
              "trend_stage": "all 50 epochs including DARE-GRAM ramp from zero; unchanged epoch/step budget",
              "trend_domain_composition": "mean of available source and target batch trend losses; source-only source alone",
              "epochs": five.EPOCHS, "batch_size": five.BATCH_SIZE, "lr": five.LR,
              "source_feature_sha256": manifest["tools"][source]["feature_sha256"],
              "target_feature_sha256": manifest["tools"][target]["feature_sha256"],
              "source_wear_sha256": base.file_hash(args.raw_root / f"{source}_wear.csv"),
              "source_label_max_vb": float(np.max(ys)), "target_labels_training_reads": 0,
              "source_only_training_target_cuts": [], "daregram_training_target_cuts": CUTS,
              "cut_pairing": "true same-tool i,i+1 pairs from cut IDs within each independently shuffled batch; no cross-tool/missing/duplicate pairing",
              "normalization_mean": mean.tolist(), "normalization_std": std.tolist()}
    five.json_write(folder / "config.json", config)
    logging.info("Training %s->%s seed=%d delta=%.6f lambda=%.4f", source, target, seed, delta, args.trend_lambda)
    traces = {}
    traces["source_only"] = five.train("source_only", xs, ys, None, source, target, seed,
                                        folder, device, args.trend_lambda, delta)
    # Source-only has already finished before any target input is opened.
    xt = normalize(cache(args.cache_root, manifest, target), mean, std)
    traces["daregram"] = five.train("daregram", xs, ys, xt, source, target, seed,
                                     folder, device, args.trend_lambda, delta)
    if traces["source_only"]["initial_model_sha256"] != traces["daregram"]["initial_model_sha256"]:
        raise ValueError("Paired methods have different initial weights")
    if traces["source_only"]["source_order_sha256_by_epoch"] != traces["daregram"]["source_order_sha256_by_epoch"]:
        raise ValueError("Paired methods have different source shuffle orders")
    if traces["source_only"]["actual_unlabeled_target_cuts"] or traces["daregram"]["actual_unlabeled_target_cuts"] != CUTS:
        raise ValueError("Target training usage mismatch")
    pred = {method: five.predict(method, xt, CUTS, seed, folder, device) for method in METHODS}
    # Both checkpoints and all model predictions are fixed before target labels are read.
    truth = base.wear_labels(args.raw_root, target, evaluation_dir=folder).astype(np.float64)
    rows = []
    for method in METHODS:
        sub = folder / method
        pd.DataFrame({"cut_index": CUTS, "true_vb": truth, "pred_vb": pred[method]}).to_csv(
            sub / "predictions.csv", index=False, float_format="%.17g")
        audit_frame, audit = physical_audit(CUTS, pred[method], delta, float(np.max(ys)))
        audit_frame.to_csv(sub / "physical_per_cut.csv", index=False, float_format="%.17g")
        five.json_write(sub / "physical_summary.json", audit)
        metric = (five.metrics_from_csv(sub / "predictions.csv", CUTS) if np.isfinite(pred[method]).all()
                  else {name: None for name in base.METRICS})
        five.json_write(sub / "metrics.json", metric)
        plot_curve(sub / "prediction.png", source, target, seed, method, np.asarray(CUTS),
                   truth, pred[method], audit_frame, args.mark_declines)
        five.json_write(sub / "training_trace.json", {k: v for k, v in traces[method].items() if k != "epoch_losses"})
        rows.append({"source": source, "target": target, "seed": seed, "method": method + "_trend",
                     **metric, **{k: v for k, v in audit.items() if not isinstance(v, list)}})
        baseline = args.baseline_root / "per_seed" / f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
        frame = pd.read_csv(baseline)
        if frame.cut_index.tolist() != CUTS or not np.allclose(frame.true_vb.to_numpy(), truth, atol=1e-6, rtol=0):
            raise ValueError(f"Frozen baseline cut/label mismatch: {baseline}")
        baseline_pred = frame.pred_vb.to_numpy(float)
        base_cut, base_audit = physical_audit(CUTS, baseline_pred, delta, float(np.max(ys)))
        base_cut.to_csv(sub / "baseline_physical_per_cut.csv", index=False, float_format="%.17g")
        five.json_write(sub / "baseline_physical_summary.json", base_audit)
        base_metric = base.metrics(truth, baseline_pred)
        rows.append({"source": source, "target": target, "seed": seed, "method": method,
                     **base_metric, **{k: v for k, v in base_audit.items() if not isinstance(v, list)}})
        if args.trend_lambda == 0:
            diff = float(np.max(np.abs(pred[method] - baseline_pred)))
            if diff > 0.005:
                raise ValueError(f"Zero-weight baseline prediction mismatch: {method} max diff={diff}")
    pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
    five.json_write(folder / "complete.json", {"source": source, "target": target, "seed": seed,
                                            "trend_lambda": args.trend_lambda,
                                            "baseline_root": str(args.baseline_root.resolve())})
    logging.info("Completed %s->%s seed=%d", source, target, seed)


def summarize(args):
    tables = [pd.read_csv(path) for path in sorted(args.out_root.glob("*_to_*/seed_*/comparison.csv"))]
    if not tables:
        return
    table = pd.concat(tables, ignore_index=True)
    # Early completed runs may predate count columns; their per-method JSON retains the lists.
    for field in ("missing_cut_count", "duplicate_cut_count"):
        if field not in table:
            table[field] = 0
        else:
            table[field] = table[field].fillna(0)
    table.to_csv(args.out_root / "per_seed_comparison.csv", index=False)
    columns = ["R2", "MAE", "RMSE", "adjacent_decline_count", "decline_over_delta_count",
               "max_decline_vb", "max_upward_jump_vb", "prediction_variance_vb2",
               "negative_count", "nonfinite_count", "missing_cut_count", "duplicate_cut_count",
               "above_source_max_oor_count"]
    summary = table.groupby(["source", "target", "method"], as_index=False)[columns].agg(["mean", "std"])
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(args.out_root / "direction_summary.csv", index=False)
    five.json_write(args.out_root / "direction_summary.json", summary.to_dict(orient="records"))


def main():
    args = arguments()
    args.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.out_root / "run.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["cuts"] != CUTS or manifest["stft_code_sha256"] != base.file_hash(Path(base.sampling.__file__)):
        raise ValueError("Cached cut list or STFT code differs")
    pairs = [(args.source, args.target)] if args.source else five.PAIRS
    try:
        for source, target in pairs:
            for seed in args.seeds:
                run_one(args, manifest, source, target, seed)
                summarize(args)
    finally:
        summarize(args)


if __name__ == "__main__":
    main()
