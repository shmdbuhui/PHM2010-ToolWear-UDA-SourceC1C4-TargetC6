"""Zero-delta trend training followed by label-free per-seed PAVA output projection."""

from __future__ import annotations

import argparse
import hashlib
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
from run_trend_physics_comparison import cache, normalize
from trend_physics import physical_audit, pava_nondecreasing


CUTS = base.ALL_CUTS
METHODS = ("source_only", "daregram")
DEFAULT_OUT = Path("artifacts/monotone_pava_delta0_full_1_315_20260926")
LAMBDA = 0.1  # Fixed before any target-label evaluation; same weight as the previous trend ablation.


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/five_seed_paired/feature_cache"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--source", choices=base.TOOLS)
    parser.add_argument("--target", choices=base.TOOLS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(five.SEEDS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (args.source is None) != (args.target is None) or (args.source is not None and args.source == args.target):
        parser.error("Specify distinct source and target, or neither for six directions")
    if not set(args.seeds) <= set(five.SEEDS):
        parser.error("Seeds must be selected from 42..46")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def plot_one(path, source, target, seed, method, truth, pred, label):
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.plot(CUTS, truth, color="black", lw=1.8, label="True VB")
    ax.plot(CUTS, pred, color="#2563a6" if label == "Raw prediction" else "#b12d58",
            lw=1.5, label=label)
    ax.set(xlabel="Target cut", ylabel="VB", title=f"{source.upper()}->{target.upper()} seed {seed}: {method}")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def measures(truth, raw, mono):
    raw_metrics = base.metrics(truth, raw)
    mono_metrics = base.metrics(truth, mono)
    changes = np.abs(mono - raw)
    raw_step = np.diff(raw)
    mono_step = np.diff(mono)
    suffix = np.asarray(CUTS) >= 250
    return {"raw": raw_metrics, "monotone": mono_metrics,
            "raw_decline_count": int(np.count_nonzero(raw_step < -1e-8)),
            "raw_max_decline_vb": float(np.maximum(-raw_step, 0).max(initial=0)),
            "monotone_decline_count": int(np.count_nonzero(mono_step < -1e-8)),
            "monotone_max_decline_vb": float(np.maximum(-mono_step, 0).max(initial=0)),
            "mean_abs_output_change_vb": float(changes.mean()),
            "max_abs_output_change_vb": float(changes.max()),
            "squared_output_change_sum_vb2": float(np.square(mono - raw).sum()),
            "cut_250_315": {
                "raw_mean_signed_error_vb": float(np.mean(raw[suffix] - truth[suffix])),
                "monotone_mean_signed_error_vb": float(np.mean(mono[suffix] - truth[suffix])),
                "raw_underestimated_cut_count": int(np.count_nonzero(raw[suffix] < truth[suffix])),
                "monotone_underestimated_cut_count": int(np.count_nonzero(mono[suffix] < truth[suffix])),
                "raw_suffix_RMSE": float(np.sqrt(np.mean(np.square(raw[suffix] - truth[suffix])))),
                "monotone_suffix_RMSE": float(np.sqrt(np.mean(np.square(mono[suffix] - truth[suffix]))))}}


def run_one(args, manifest, source, target, seed):
    folder = args.out_root / f"{source}_to_{target}" / f"seed_{seed}"
    if (folder / "complete.json").is_file():
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if cfg["trend_delta_vb"] != 0 or cfg["trend_lambda"] != LAMBDA or cfg["projection"] != "equal-weight PAVA per seed over target cuts 1..315":
            raise ValueError(f"Completed config mismatch: {folder}")
        logging.info("Already completed %s", folder)
        return
    if folder.exists():
        raise FileExistsError(f"Incomplete result requires inspection: {folder}")
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
    config = {"source": source, "target": target, "seed": seed,
              "trend_delta_vb": 0.0, "trend_lambda": LAMBDA,
              "trend_loss": "mean(relu(pred_cut_i - pred_cut_i_plus_1)^2) for true same-tool adjacent cuts in each shuffled batch",
              "trend_stage": "all 50 epochs; original DARE-GRAM ramp and five optimizer steps per epoch retained",
              "projection": "equal-weight PAVA per seed over target cuts 1..315",
              "projection_input": "cut IDs and raw model predictions only; before target wear labels are opened",
              "projection_objective": "minimize sum((monotone_pred - raw_pred)^2) subject to adjacent differences >= 0",
              "output_unit": "original VB; no wear-label normalization or inverse transform",
              "epochs": five.EPOCHS, "batch_size": five.BATCH_SIZE, "lr": five.LR,
              "seeds": list(five.SEEDS), "target_label_training_reads": 0,
              "source_only_training_target_cuts": [], "daregram_training_target_cuts": CUTS,
              "source_wear_sha256": base.file_hash(args.raw_root / f"{source}_wear.csv"),
              "source_feature_sha256": manifest["tools"][source]["feature_sha256"],
              "target_feature_sha256": manifest["tools"][target]["feature_sha256"],
              "source_input_mean": mean.tolist(), "source_input_std": std.tolist(),
              "smoke_performance_gate": "five-seed mean monotone RMSE <= 1.25 * five-seed mean raw RMSE for each method; defined before evaluation"}
    write_json(folder / "config.json", config)
    logging.info("Training %s->%s seed=%d delta=0 lambda=%.3f", source, target, seed, LAMBDA)
    source_trace = five.train("source_only", xs, ys, None, source, target, seed, folder, device, LAMBDA, 0.0)
    # The source-only checkpoint exists before any target training feature is opened.
    xt = normalize(cache(args.cache_root, manifest, target), mean, std)
    dare_trace = five.train("daregram", xs, ys, xt, source, target, seed, folder, device, LAMBDA, 0.0)
    if source_trace["initial_model_sha256"] != dare_trace["initial_model_sha256"] or source_trace["source_order_sha256_by_epoch"] != dare_trace["source_order_sha256_by_epoch"]:
        raise ValueError("Initial model/source shuffle mismatch")
    if source_trace["actual_unlabeled_target_cuts"] or dare_trace["actual_unlabeled_target_cuts"] != CUTS:
        raise ValueError("Source-only or full-target usage mismatch")
    raw = {method: five.predict(method, xt, CUTS, seed, folder, device) for method in METHODS}
    # This entire projection block runs before opening the target wear CSV.
    monotone = {method: pava_nondecreasing(raw[method]) for method in METHODS}
    for method in METHODS:
        if raw[method].shape != (315,) or not np.isfinite(raw[method]).all() or np.any(np.diff(monotone[method]) < -1e-8):
            raise ValueError(f"Invalid raw or projected prediction: {method}")
        write_json(folder / method / "projection_prelabel.json", {
            "cut_index": CUTS, "raw_prediction_sha256": hashlib.sha256(raw[method].tobytes()).hexdigest(),
            "monotone_prediction_sha256": hashlib.sha256(monotone[method].tobytes()).hexdigest(),
            "target_label_reads_so_far": 0, "minimum_adjacent_difference_vb": float(np.diff(monotone[method]).min())})
    truth = base.wear_labels(args.raw_root, target, evaluation_dir=folder).astype(np.float64)
    rows = []
    for method, trace in (("source_only", source_trace), ("daregram", dare_trace)):
        sub = folder / method
        r, m = raw[method], monotone[method]
        frame = pd.DataFrame({"cut_index": CUTS, "true_vb": truth, "raw_pred": r, "monotone_pred": m})
        frame.to_csv(sub / "predictions.csv", index=False, float_format="%.17g")
        result = measures(truth, r, m)
        write_json(sub / "metrics_projection.json", result)
        for name, pred in (("raw", r), ("monotone", m)):
            per_cut, audit = physical_audit(CUTS, pred, 0.0, float(np.max(ys)))
            per_cut = per_cut.rename(columns={"pred_vb_raw": f"{name}_pred"})
            per_cut.to_csv(sub / f"physical_{name}_per_cut.csv", index=False, float_format="%.17g")
            write_json(sub / f"physical_{name}.json", audit)
        plot_one(sub / "raw_prediction.png", source, target, seed, method, truth, r, "Raw prediction")
        plot_one(sub / "monotone_prediction.png", source, target, seed, method, truth, m,
                 "Monotone-projected prediction")
        write_json(sub / "training_trace.json", {key: value for key, value in trace.items() if key != "epoch_losses"})
        rows.append({"source": source, "target": target, "seed": seed, "method": method,
                     **{f"raw_{key}": value for key, value in result["raw"].items()},
                     **{f"monotone_{key}": value for key, value in result["monotone"].items()},
                     **{key: value for key, value in result.items() if key not in ("raw", "monotone", "cut_250_315")},
                     **result["cut_250_315"]})
    pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
    write_json(folder / "complete.json", {"source": source, "target": target, "seed": seed,
                                          "methods": list(METHODS), "trend_delta_vb": 0.0,
                                          "projection": "PAVA per seed before target label read"})
    logging.info("Completed %s->%s seed=%d", source, target, seed)


def summarize(args):
    files = sorted(args.out_root.glob("*_to_*/seed_*/comparison.csv"))
    if not files:
        return
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame.to_csv(args.out_root / "per_seed_comparison.csv", index=False)
    metrics = ["raw_R2", "raw_MAE", "raw_RMSE", "monotone_R2", "monotone_MAE", "monotone_RMSE",
               "raw_decline_count", "raw_max_decline_vb", "monotone_decline_count", "monotone_max_decline_vb",
               "mean_abs_output_change_vb", "max_abs_output_change_vb",
               "raw_mean_signed_error_vb", "monotone_mean_signed_error_vb",
               "raw_underestimated_cut_count", "monotone_underestimated_cut_count",
               "raw_suffix_RMSE", "monotone_suffix_RMSE"]
    metrics = [m for m in metrics if m in frame.columns]
    summary = frame.groupby(["source", "target", "method"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(args.out_root / "direction_summary.csv", index=False)
    write_json(args.out_root / "direction_summary.json",
               summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"))


def main():
    args = arguments()
    args.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.out_root / "run.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["cuts"] != CUTS or manifest["stft_code_sha256"] != base.file_hash(Path(base.sampling.__file__)):
        raise ValueError("STFT cache provenance mismatch")
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
