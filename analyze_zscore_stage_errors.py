"""Read-only per-cut and stage-error audit for the six-direction five-seed Z-score run.

Uses original full-cut predictions where available. Target-C6 full-cut predictions
come from the later read-only re-evaluation of the same checkpoint. If a prediction
file is absent, the corresponding final checkpoint is evaluated without retraining.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from networks.resnet import ResNet18


TOOLS = ("c1", "c4", "c6")
METHODS = ("source_only", "daregram")
SEEDS = range(42, 47)
METRICS = ("MAE", "RMSE", "signed_bias", "R2")
SCOPES = {"full_1_315": ((1, 315), (("early", 1, 105), ("middle", 106, 210), ("late", 211, 315))),
          "suffix_95_315": ((95, 315), (("early", 95, 168), ("middle", 169, 241), ("late", 242, 315)))}
COLORS = {"source_only": "#2364a0", "daregram": "#d47b24"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labels(raw_root: Path, tool: str) -> tuple[np.ndarray, list[dict], dict]:
    path = raw_root / f"{tool}_wear.csv"
    frame = pd.read_csv(path)
    if list(frame.columns) != ["cut", "flute_1", "flute_2", "flute_3"]:
        raise ValueError(f"Wrong wear CSV schema: {path}")
    # Preserve exact cut-index matching and the experiment's float32 mean conversion.
    if len(frame) != 315 or frame.cut.astype(int).tolist() != list(range(1, 316)):
        raise ValueError(f"Missing, duplicate, or unordered wear cut: {path}")
    flutes = frame[["flute_1", "flute_2", "flute_3"]].to_numpy(dtype=np.float64)
    has_label = np.isfinite(flutes).all(axis=1)
    y = frame[["flute_1", "flute_2", "flute_3"]].mean(axis=1).to_numpy(dtype=np.float32).astype(np.float64)
    y[~has_label] = np.nan
    rows = [{"tool": tool, "cut_index": cut, "has_true_label": bool(has_label[cut - 1]),
             "flute_1": float(flutes[cut - 1, 0]) if np.isfinite(flutes[cut - 1, 0]) else "",
             "flute_2": float(flutes[cut - 1, 1]) if np.isfinite(flutes[cut - 1, 1]) else "",
             "flute_3": float(flutes[cut - 1, 2]) if np.isfinite(flutes[cut - 1, 2]) else "",
             "true_vb_three_flute_mean": float(y[cut - 1]) if has_label[cut - 1] else "",
             "label_file": str(path.resolve()), "label_file_sha256": sha(path)}
            for cut in range(1, 316)]
    return y, rows, {"tool": tool, "path": str(path.resolve()), "sha256": sha(path),
                     "cuts_1_315_with_label": int(has_label.sum()),
                     "missing_cuts": (np.flatnonzero(~has_label) + 1).tolist(),
                     "label_definition": "float32(mean(flute_1, flute_2, flute_3))"}


def csv_predictions(path: Path, true_values: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"]:
        raise ValueError(f"Wrong prediction columns: {path}")
    cuts = frame.cut_index.astype(int).to_numpy()
    if len(np.unique(cuts)) != len(cuts) or not np.isin(cuts, np.arange(1, 316)).all():
        raise ValueError(f"Invalid prediction cut indices: {path}")
    pred = np.full(315, np.nan)
    pred[cuts - 1] = frame.pred_vb.to_numpy(dtype=np.float64)
    recorded_y = frame.true_vb.to_numpy(dtype=np.float64)
    if not np.isfinite(pred[cuts - 1]).all():
        raise ValueError(f"Nonfinite existing predictions: {path}")
    common = np.isfinite(true_values[cuts - 1])
    if not np.allclose(recorded_y[common], true_values[cuts[common] - 1], rtol=0, atol=1e-6):
        raise ValueError(f"Prediction file labels disagree with original wear CSV: {path}")
    return pred


def infer_missing(checkpoint_path: Path, config: dict, raw_cache: Path,
                  device: torch.device) -> np.ndarray:
    if sha(raw_cache) != config["target_feature_sha256"]:
        raise ValueError(f"Target input cache changed: {raw_cache}")
    raw = np.load(raw_cache, mmap_mode="r", allow_pickle=False)
    mean = np.asarray(config["source_normalization_mean"], dtype=np.float32)
    std = np.asarray(config["source_normalization_std"], dtype=np.float32)
    x = ((raw - mean[None, :, None, None]) /
         (std[None, :, None, None] + 1e-8)).astype(np.float32)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != config["epochs"]:
        raise ValueError("Not a final-epoch checkpoint")
    model = nn.Module()
    model.feature_extractor = ResNet18()
    model.regressor = nn.Sequential(nn.Linear(512, 1))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, 315, 63):
            xb = torch.from_numpy(x[start:start + 63]).to(device)
            chunks.append(model.regressor(model.feature_extractor(xb)).cpu().numpy().reshape(-1))
    return np.concatenate(chunks).astype(np.float64)


def stage_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return {"n": 0, "MAE": "", "RMSE": "", "signed_bias": "", "R2": "",
                "R2_status": "no_labeled_predictions"}
    yy, pp = y[valid], p[valid]
    error = pp - yy
    sst = np.sum((yy - yy.mean()) ** 2)
    r2_valid = len(yy) >= 2 and sst > 1e-12
    return {"n": int(len(yy)), "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "signed_bias": float(error.mean()),
            "R2": float(1 - np.sum(error ** 2) / sst) if r2_valid else "",
            "R2_status": "defined" if r2_valid else "insufficient_label_variation"}


def plot_scope(path: Path, source: str, target: str, scope: str, y: np.ndarray,
               predictions: dict, global_limits: dict) -> None:
    (lo, hi), stages = SCOPES[scope]
    cuts = np.arange(lo, hi + 1)
    subset_y = y[lo - 1:hi]
    # Every plotted point is an original cut; lines connect cuts in order without smoothing.
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(cuts, subset_y, color="black", lw=2.0, label="true VB")
    for method in METHODS:
        matrix = np.stack([predictions[source, target, seed, method][lo - 1:hi] for seed in SEEDS])
        for line in matrix:
            axes[0].plot(cuts, line, color=COLORS[method], alpha=0.10, lw=0.7)
        mean_pred = matrix.mean(axis=0)
        axes[0].plot(cuts, mean_pred, color=COLORS[method], lw=1.6,
                     label=f"{method} five-seed mean")
        errors = matrix - subset_y[None, :]
        axes[1].plot(cuts, errors.mean(axis=0), color=COLORS[method], lw=1.35,
                     label=f"{method} mean signed error")
        axes[2].plot(cuts, np.abs(errors).mean(axis=0), color=COLORS[method], lw=1.35,
                     label=f"{method} mean absolute error")
    axes[0].set(ylabel="VB", ylim=global_limits["prediction"])
    axes[1].axhline(0, color="black", lw=0.7, alpha=0.5)
    axes[1].set(ylabel="prediction − truth", ylim=global_limits["signed"])
    axes[2].set(xlabel="target cut index", ylabel="mean absolute error", ylim=global_limits["absolute"])
    boundaries = [stages[0][2] + 0.5, stages[1][2] + 0.5]
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="#6b6b6b", ls="--", lw=0.9)
        if target == "c6" and scope == "full_1_315":
            ax.axvline(94.5, color="#6d3b9c", ls=":", lw=1.1,
                       label="C6 original evaluation starts at 95" if ax is axes[0] else None)
        ax.set_xlim(lo, hi)
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"{source.upper()} → {target.upper()} | {scope} | original cut order; no smoothing")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--full-c6-root", type=Path, default=Path("artifacts/norm_comparison_20260925/zscore"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/stage_error_analysis_zscore_20260925"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root, full_root, raw_root, out = (p.resolve() for p in
                                      (args.experiment_root, args.full_c6_root,
                                       args.raw_root, args.out_root))
    if out.exists() or out == root or root in out.parents or out == full_root or full_root in out.parents:
        parser.error("Output must be a new independent directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    truth, availability, label_sources = {}, [], []
    for tool in TOOLS:
        truth[tool], rows, source_info = labels(raw_root, tool)
        availability.extend(rows)
        label_sources.append(source_info)
    out.mkdir(parents=True)
    (out / "per_cut").mkdir()
    (out / "figures").mkdir()
    write_csv(out / "label_availability_cuts_1_315.csv", availability)
    predictions, provenance, original_suffix_comparison = {}, [], []
    device = torch.device(args.device)
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                pair = root / f"{source}_to_{target}" / f"seed_{seed}"
                for method in METHODS:
                    folder = pair / method
                    config = json_read(folder / "config.json")
                    if (config["source"], config["target"], config["seed"], config["method"],
                        config["checkpoint_selection"]) != (source, target, seed, method, "final epoch"):
                        raise ValueError(f"Checkpoint config mismatch: {folder}")
                    expected_original = list(range(95, 316)) if target == "c6" else list(range(1, 316))
                    if config["evaluation_cuts"] != expected_original:
                        raise ValueError(f"Original evaluation cuts changed: {folder}")
                    checkpoint = folder / "final.pth"
                    checkpoint_hash = sha(checkpoint)
                    original_file = folder / "predictions.csv"
                    original_pred = csv_predictions(original_file, truth[target]) if original_file.is_file() else None
                    full_file = original_file
                    source_kind = "original_full_prediction_csv"
                    if target == "c6":
                        full_folder = full_root / f"{source}_to_{target}" / f"seed_{seed}" / method
                        full_checkpoint = full_folder / "final.pth"
                        if full_checkpoint.is_file() and sha(full_checkpoint) != checkpoint_hash:
                            raise ValueError(f"Full C6 prediction checkpoint differs: {full_folder}")
                        full_config = json_read(full_folder / "config.json") if (full_folder / "config.json").is_file() else None
                        if full_config is not None:
                            if (full_config["norm_method"], full_config["source"],
                                full_config["target"], full_config["seed"],
                                full_config["method"], full_config["evaluation_cuts"]) != (
                                    "zscore", source, target, seed, method, list(range(1, 316))):
                                raise ValueError(f"Full C6 config mismatch: {full_folder}")
                            if (full_config["source_normalization_parameters"]["mean"] != config["source_normalization_mean"] or
                                full_config["source_normalization_parameters"]["std"] != config["source_normalization_std"]):
                                raise ValueError(f"Full C6 normalization differs: {full_folder}")
                        full_file = full_folder / "predictions.csv"
                        source_kind = "existing_full_C6_reevaluation_csv"
                    if full_file.is_file():
                        pred = csv_predictions(full_file, truth[target])
                    else:
                        pred = infer_missing(checkpoint, config,
                                             root / "feature_cache" / f"{target}_stft.npy", device)
                        source_kind = "inferred_from_original_final_checkpoint"
                    if pred.shape != (315,) or not np.isfinite(pred).all():
                        raise ValueError(f"Full 315 predictions unavailable: {source} {target} {seed} {method}")
                    predictions[source, target, seed, method] = pred
                    max_overlap = ""
                    if target == "c6" and original_pred is not None:
                        max_overlap = float(np.max(np.abs(pred[94:] - original_pred[94:])))
                        original_suffix_comparison.append({"source": source, "target": target,
                                                           "seed": seed, "method": method,
                                                           "original_suffix_prediction_file": str(original_file.resolve()),
                                                           "full_prediction_file": str(full_file.resolve()),
                                                           "max_overlap_prediction_abs_difference": max_overlap})
                    provenance.append({"source": source, "target": target, "seed": seed,
                                       "method": method, "source_wear_csv": str((raw_root / f"{source}_wear.csv").resolve()),
                                       "target_wear_csv": str((raw_root / f"{target}_wear.csv").resolve()),
                                       "checkpoint": str(checkpoint.resolve()),
                                       "checkpoint_sha256": checkpoint_hash,
                                       "config": str((folder / "config.json").resolve()),
                                       "prediction_source": source_kind,
                                       "prediction_file": str(full_file.resolve()) if full_file.is_file() else "",
                                       "prediction_file_sha256": sha(full_file) if full_file.is_file() else "",
                                       "original_evaluation_cut_first": expected_original[0],
                                       "original_evaluation_cut_last": expected_original[-1],
                                       "saved_prediction_cut_first": 1, "saved_prediction_cut_last": 315,
                                       "overlap_prediction_max_abs_difference": max_overlap,
                                       "source_normalization_mean": json.dumps(config["source_normalization_mean"]),
                                       "source_normalization_std": json.dumps(config["source_normalization_std"])})
                print(f"Matched predictions {source}->{target} seed {seed}", flush=True)
    write_csv(out / "checkpoint_prediction_provenance.csv", provenance)
    if original_suffix_comparison:
        write_csv(out / "c6_original_suffix_cuts_95_315_vs_full_cuts_1_315_prediction.csv", original_suffix_comparison)

    metric_rows, extrapolation_rows, top_error_rows = [], [], []
    for source in TOOLS:
        source_max = float(np.nanmax(truth[source]))
        for target in TOOLS:
            if source == target:
                continue
            target_y = truth[target]
            for method in METHODS:
                matrix = np.stack([predictions[source, target, seed, method] for seed in SEEDS])
                per_cut = []
                for cut in range(1, 316):
                    y = target_y[cut - 1]
                    row = {"source": source, "target": target, "method": method,
                           "cut_index": cut, "has_true_label": bool(np.isfinite(y)),
                           "true_vb": float(y) if np.isfinite(y) else "",
                           "above_source_training_label_max": bool(y > source_max) if np.isfinite(y) else "",
                           "source_training_label_max": source_max}
                    for j, seed in enumerate(SEEDS):
                        pred = float(matrix[j, cut - 1])
                        row[f"seed_{seed}_pred_vb"] = pred
                        row[f"seed_{seed}_signed_error"] = pred - y if np.isfinite(y) else ""
                        row[f"seed_{seed}_absolute_error"] = abs(pred - y) if np.isfinite(y) else ""
                    row["five_seed_mean_pred_vb"] = float(matrix[:, cut - 1].mean())
                    row["five_seed_mean_signed_error"] = float(matrix[:, cut - 1].mean() - y) if np.isfinite(y) else ""
                    row["five_seed_mean_absolute_error"] = float(np.abs(matrix[:, cut - 1] - y).mean()) if np.isfinite(y) else ""
                    row["absolute_error_of_five_seed_mean_prediction"] = abs(matrix[:, cut - 1].mean() - y) if np.isfinite(y) else ""
                    per_cut.append(row)
                write_csv(out / "per_cut" / f"{source}_to_{target}_{method}_cuts_1_315.csv", per_cut)
                for seed in SEEDS:
                    pred = predictions[source, target, seed, method]
                    for scope, (scope_range, stages) in SCOPES.items():
                        if scope == "suffix_95_315" and target != "c6":
                            continue
                        for segment, lo, hi in (("all", *scope_range), *stages):
                            result = stage_metrics(target_y[lo - 1:hi], pred[lo - 1:hi])
                            metric_rows.append({"source": source, "target": target, "method": method,
                                                "seed": seed, "scope": scope, "segment": segment,
                                                "cut_first": lo, "cut_last": hi, **result})
                        lo, hi = scope_range
                        ys, ps = target_y[lo - 1:hi], pred[lo - 1:hi]
                        valid = np.isfinite(ys) & np.isfinite(ps)
                        error2 = np.where(valid, (ps - ys) ** 2, 0)
                        total = float(error2.sum())
                        ordered = np.argsort(-error2)
                        for rank, offset in enumerate(ordered[:10], 1):
                            top_error_rows.append({"source": source, "target": target, "method": method,
                                                   "seed": seed, "scope": scope, "rank": rank,
                                                   "cut_index": lo + int(offset),
                                                   "squared_error": float(error2[offset]),
                                                   "share_of_total_squared_error": float(error2[offset] / total) if total else "",
                                                   "top_5_share": float(error2[ordered[:5]].sum() / total) if total else "",
                                                   "top_10_share": float(error2[ordered[:10]].sum() / total) if total else ""})
                    for group, mask in (("at_or_below_source_max", target_y <= source_max),
                                        ("above_source_max", target_y > source_max)):
                        result = stage_metrics(target_y[mask], pred[mask])
                        extrapolation_rows.append({"source": source, "target": target,
                                                   "method": method, "seed": seed,
                                                   "source_training_label_max": source_max,
                                                   "target_group": group, **result})
    write_csv(out / "per_seed_stage_metrics_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv", metric_rows)
    write_csv(out / "per_seed_extrapolation_metrics_full_cuts_1_315.csv", extrapolation_rows)
    write_csv(out / "top_squared_error_cuts_per_seed_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv", top_error_rows)

    key = lambda r: (r["source"], r["target"], r["scope"], r["segment"], r["seed"])
    indexed = {(key(r), r["method"]): r for r in metric_rows}
    paired = []
    for r in metric_rows:
        if r["method"] != "source_only":
            continue
        d = indexed[key(r), "daregram"]
        row = {"source": r["source"], "target": r["target"], "seed": r["seed"],
               "scope": r["scope"], "segment": r["segment"],
               "cut_first": r["cut_first"], "cut_last": r["cut_last"],
               "n_source_only": r["n"], "n_daregram": d["n"]}
        for metric in METRICS:
            row[f"source_only_{metric}"] = r[metric]
            row[f"daregram_{metric}"] = d[metric]
            row[f"delta_{metric}"] = float(d[metric]) - float(r[metric]) if r[metric] != "" and d[metric] != "" else ""
        paired.append(row)
    write_csv(out / "paired_seed_stage_metrics_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv", paired)
    summary, paired_summary = [], []
    groups = sorted({(r["source"], r["target"], r["scope"], r["segment"]) for r in metric_rows})
    for source, target, scope, segment in groups:
        for method in METHODS:
            selected = [r for r in metric_rows if (r["source"], r["target"], r["scope"], r["segment"], r["method"]) ==
                        (source, target, scope, segment, method)]
            row = {"source": source, "target": target, "scope": scope,
                   "segment": segment, "method": method, "n_seeds": len(selected),
                   "n_per_seed": selected[0]["n"], "cut_first": selected[0]["cut_first"],
                   "cut_last": selected[0]["cut_last"]}
            for metric in METRICS:
                values = np.array([float(r[metric]) for r in selected if r[metric] != ""])
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else ""
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else ""
                row[f"{metric}_defined_seed_count"] = len(values)
            summary.append(row)
        selected = [r for r in paired if (r["source"], r["target"], r["scope"], r["segment"]) ==
                    (source, target, scope, segment)]
        row = {"source": source, "target": target, "scope": scope,
               "segment": segment, "n_seeds": len(selected),
               "n_per_seed": selected[0]["n_source_only"],
               "cut_first": selected[0]["cut_first"], "cut_last": selected[0]["cut_last"]}
        for metric in METRICS:
            values = np.array([float(r[f"delta_{metric}"]) for r in selected if r[f"delta_{metric}"] != ""])
            row[f"delta_{metric}_mean"] = float(values.mean()) if len(values) else ""
            row[f"delta_{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else ""
            row[f"delta_{metric}_defined_seed_count"] = len(values)
        paired_summary.append(row)
    write_csv(out / "stage_summary_by_method_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv", summary)
    write_csv(out / "paired_stage_summary_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv", paired_summary)

    global_y = np.concatenate([truth[t][np.isfinite(truth[t])] for t in TOOLS] + list(predictions.values()))
    global_error = np.concatenate([predictions[s, t, seed, m] - truth[t]
                                   for s in TOOLS for t in TOOLS if s != t
                                   for seed in SEEDS for m in METHODS])
    pmin, pmax = float(np.nanmin(global_y)), float(np.nanmax(global_y))
    emax = float(np.nanmax(np.abs(global_error)))
    limits = {"prediction": (pmin - 5, pmax + 5), "signed": (-emax * 1.04, emax * 1.04),
              "absolute": (0, emax * 1.04)}
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for scope in ("full_1_315", "suffix_95_315") if target == "c6" else ("full_1_315",):
                plot_scope(out / "figures" / f"{source}_to_{target}_{scope}_curves_and_errors.png",
                           source, target, scope, truth[target], predictions, limits)
    manifest = {"experiment_root": str(root), "full_C6_prediction_root": str(full_root),
                "label_sources": label_sources, "evaluation_scopes": {
                    name: {"all": list(all_cuts), "segments": [list(x) for x in stages]}
                    for name, (all_cuts, stages) in SCOPES.items()},
                "primary_original_evaluation": "C1/C4 targets 1..315; C6 targets 95..315",
                "additional_C6_full_evaluation": "1..315, using existing re-evaluation of identical final checkpoints",
                "target_unlabeled_adaptation_protocol_changed": False,
                "models_retrained_or_checkpoints_reselected": False,
                "prediction_fallback_used": sum(r["prediction_source"] == "inferred_from_original_final_checkpoint" for r in provenance),
                "global_plot_limits": limits,
                "R2_rule": "defined only for >=2 labeled cuts and sum((y-mean(y))**2)>1e-12",
                "sample_standard_deviation_ddof": 1}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed: {out}; {len(metric_rows)} per-seed stage rows, {len(paired)} paired rows", flush=True)


if __name__ == "__main__":
    main()
