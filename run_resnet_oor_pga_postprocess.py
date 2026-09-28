"""Exploratory OOR-PGA and PGA-like postprocessing of frozen ResNet runs.

Uses source labels and unlabeled source/target STFT only until predictions are
written. Target labels enter solely in the subsequent evaluation phase.
These transfer EEMD mechanisms, not its 43-feature threshold calibration or
its independently trained XGBoost baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TOOLS = ("c1", "c4", "c6")
METHODS = ("source_only", "daregram")
SEEDS = (42, 43, 44, 45, 46)
CUTS = np.arange(1, 316)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--prediction-root", type=Path, default=Path("artifacts/five_seed_full_lifecycle"))
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing c1_wear.csv etc.")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/resnet_oor_pga_postprocess"))
    parser.add_argument("--lambda-pga", type=float, default=0.75)
    parser.add_argument("--tl", type=float, default=3 / 43)
    parser.add_argument("--th", type=float, default=6 / 43)
    parser.add_argument("--late-start", type=float, default=0.70)
    args = parser.parse_args()
    if not (0 <= args.tl < args.th <= 1 and args.lambda_pga >= 0 and 0 <= args.late_start < 1):
        parser.error("Require 0 <= TL < TH <= 1, lambda >= 0 and 0 <= late-start < 1")
    if args.out_root.exists():
        parser.error("Output directory already exists; use a new path to preserve previous results")
    return args


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def read_cache(path: Path) -> np.ndarray:
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.shape != (315, 6, 128, 128) or data.dtype != np.float32:
        raise ValueError(f"Expected float32 (315,6,128,128) STFT cache: {path}")
    return data


def band_features(cache: np.ndarray) -> np.ndarray:
    """48 fixed features: mean log1p STFT intensity in 8 bands per channel."""
    features = np.empty((315, 48), dtype=np.float64)
    for start in range(0, 315, 15):
        # Frequency is the 128-row axis; time is averaged within each band.
        block = np.asarray(cache[start:start + 15], dtype=np.float64)
        features[start:start + len(block)] = block.reshape(len(block), 6, 8, 16, 128).mean(axis=(3, 4)).reshape(len(block), 48)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite STFT band summary")
    return features


def source_wear(root: Path, tool: str) -> np.ndarray:
    frame = pd.read_csv(root / f"{tool}_wear.csv")
    if list(frame.columns) != ["cut", "flute_1", "flute_2", "flute_3"] or frame.cut.tolist() != CUTS.tolist():
        raise ValueError(f"Unexpected source wear labels for {tool}")
    wear = frame[["flute_1", "flute_2", "flute_3"]].mean(axis=1).to_numpy(dtype=np.float64)
    if not np.isfinite(wear).all():
        raise ValueError(f"Nonfinite source wear labels for {tool}")
    return wear


def score_and_increment(source_features: np.ndarray, target_features: np.ndarray,
                        wear: np.ndarray, tl: float, th: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lower, upper = source_features.min(axis=0), source_features.max(axis=0)
    rate = ((target_features < lower) | (target_features > upper)).mean(axis=1)
    gate = np.clip((rate - tl) / (th - tl), 0.0, 1.0)
    tau = (CUTS - 1) / 314.0
    # EEMD PGA-like uses a source-only least-squares linear trend.
    slope, _ = np.polyfit(tau, wear, 1)
    # Do not create a negative "anti-wear" correction from a noisy source fit.
    increment = np.full(315, max(0.0, float(slope)) / 314.0)
    increment[0] = 0.0
    return rate, gate, increment, float(slope)


def corrections(gate: np.ndarray, increment: np.ndarray, lam: float,
                late_start: float) -> dict[str, np.ndarray]:
    tau = (CUTS - 1) / 314.0
    late_gate = gate * (tau >= late_start)
    return {
        "oor_pga_like": np.cumsum(lam * gate * increment),
        "late_oor_pga_like": np.cumsum(lam * late_gate * increment),
    }


def power_exponent(wear: np.ndarray) -> float:
    """EEMD's robust log-log source wear exponent, without target labels."""
    tau = (CUTS - 1) / 314.0
    usable = (tau > 0) & (wear > 0)
    if usable.sum() < 3:
        raise ValueError("At least three positive-time, positive-wear source labels are required")
    slope, _, _, _ = theilslopes(np.log(wear[usable]), np.log(tau[usable]))
    return float(slope)


def oor_pga_factor(gate: np.ndarray, exponent: float) -> tuple[np.ndarray, int | None]:
    """EEMD OOR-triggered PGA: first fully open gate after tau=0, then (t/t_OOR)^m."""
    tau = (CUTS - 1) / 314.0
    candidates = np.flatnonzero((gate >= 1 - 1e-12) & (tau > 0))
    factor = np.ones(315, dtype=np.float64)
    if len(candidates) == 0:
        return factor, None
    index = int(candidates[0])
    active = np.arange(315) > index
    factor[active] = (tau[active] / tau[index]) ** exponent
    return factor, int(CUTS[index])


def read_baseline(path: Path) -> np.ndarray:
    # Explicit usecols prevents target labels in the existing CSV from entering correction.
    frame = pd.read_csv(path, usecols=["cut_index", "pred_vb"])
    if frame.cut_index.tolist() != CUTS.tolist() or frame.cut_index.duplicated().any():
        raise ValueError(f"Missing or reordered prediction cuts: {path}")
    prediction = frame.pred_vb.to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite predictions: {path}")
    return prediction


def metrics(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    true, pred = truth[mask], prediction[mask]
    if len(true) < 2:
        raise ValueError("Evaluation mask too small")
    return {"n": int(len(true)), "MAE": float(mean_absolute_error(true, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
            "R2": float(r2_score(true, pred)) if np.ptp(true) > 0 else float("nan"),
            "bias": float(np.mean(pred - true))}


def plot_direction(args: argparse.Namespace, source: str, target: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for method in METHODS:
        tables = [pd.read_csv(args.out_root / f"{source}_to_{target}" / f"seed_{seed}" /
                              method / "evaluated_predictions.csv") for seed in SEEDS]
        truth = tables[0].true_vb.to_numpy(dtype=float)
        if any(not np.allclose(frame.true_vb.to_numpy(dtype=float), truth, rtol=0, atol=1e-12)
               for frame in tables[1:]):
            raise ValueError(f"Target labels differ across seeds: {source}->{target}")
        means = {name: np.mean([frame[name].to_numpy(dtype=float) for frame in tables], axis=0)
                 for name in ("baseline", "oor_pga", "oor_pga_like", "late_oor_pga_like")}
        gate = tables[0].gate.to_numpy(dtype=float)
        fig, (ax, gate_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(CUTS, truth, color="#222222", linewidth=1.8, label="True wear")
        ax.plot(CUTS, means["baseline"], color="#0072B2", alpha=.85, label=method)
        ax.plot(CUTS, means["oor_pga"], color="#9467BD", alpha=.8, label="OOR-PGA")
        ax.plot(CUTS, means["oor_pga_like"], color="#D55E00", alpha=.85, label="OOR-PGA-like")
        ax.plot(CUTS, means["late_oor_pga_like"], color="#009E73", alpha=.85,
                label="Late-guarded OOR-PGA-like")
        for boundary in (105.5, 210.5):
            ax.axvline(boundary, color="gray", alpha=.35, linestyle=":")
        ax.set(ylabel="VB", title=f"{source.upper()} to {target.upper()} | {method} | five-seed mean")
        ax.grid(alpha=.2)
        ax.legend(ncol=2, fontsize=8)
        gate_ax.plot(CUTS, gate, color="#9467BD", linewidth=1, label="Unlabeled STFT OOR gate")
        gate_ax.set(xlabel="Cut index", ylabel="Gate", xlim=(1, 315), ylim=(-.05, 1.05))
        gate_ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(args.out_root / f"{source}_to_{target}" / f"{method}_comparison.png", dpi=180)
        plt.close(fig)


def validate_provenance(args: argparse.Namespace) -> dict:
    manifest_path = args.trained_root / "feature_cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cuts") != CUTS.tolist() or manifest.get("feature_shape") != [315, 6, 128, 128]:
        raise ValueError("Training feature-cache manifest differs from 315-cut STFT protocol")
    if manifest.get("raw_root") != str(args.raw_root.resolve()):
        raise ValueError("Raw root differs from recorded training data")
    for tool in TOOLS:
        cache = args.trained_root / "feature_cache" / f"{tool}_stft.npy"
        if digest(cache) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"Feature cache checksum mismatch: {cache}")
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                folder = args.trained_root / f"{source}_to_{target}" / f"seed_{seed}"
                config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
                if (config.get("source"), config.get("target"), config.get("seed")) != (source, target, seed):
                    raise ValueError(f"Wrong training configuration: {folder}")
                if config.get("source_feature_sha256") != manifest["tools"][source]["feature_sha256"] or config.get("target_feature_sha256") != manifest["tools"][target]["feature_sha256"]:
                    raise ValueError(f"Wrong cache provenance: {folder}")
                for method in METHODS:
                    checkpoint = folder / method / "final.pth"
                    if digest(checkpoint) != audit["checkpoint_sha256"][method]:
                        raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
                    prediction = args.prediction_root / f"{source}_to_{target}" / f"seed_{seed}" / method / "predictions.csv"
                    if not prediction.is_file():
                        raise FileNotFoundError(f"Run evaluate_full_lifecycle.py first: {prediction}")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = validate_provenance(args)
    # Read every baseline without target-label columns before any target label evaluation.
    baselines = {}
    for source in TOOLS:
        for target in TOOLS:
            if source != target:
                for seed in SEEDS:
                    for method in METHODS:
                        path = args.prediction_root / f"{source}_to_{target}" / f"seed_{seed}" / method / "predictions.csv"
                        baselines[source, target, seed, method] = read_baseline(path)
    args.out_root.mkdir(parents=True)
    source_features = {tool: band_features(read_cache(args.trained_root / "feature_cache" / f"{tool}_stft.npy")) for tool in TOOLS}
    source_labels = {tool: source_wear(args.raw_root, tool) for tool in TOOLS}
    parameter_rows = []
    # All corrected predictions are frozen on disk before target labels are read.
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            rate, gate, increment, slope = score_and_increment(
                source_features[source], source_features[target], source_labels[source], args.tl, args.th)
            increments = corrections(gate, increment, args.lambda_pga, args.late_start)
            exponent = power_exponent(source_labels[source])
            factor, trigger_cut = oor_pga_factor(gate, exponent)
            direction = args.out_root / f"{source}_to_{target}"
            direction.mkdir()
            parameter_rows.append({"source": source, "target": target, "source_trend_slope": slope,
                                   "source_power_exponent": exponent, "oor_pga_trigger_cut": trigger_cut,
                                   "source_label_max": float(source_labels[source].max()),
                                   "mean_oor_rate": float(rate.mean()), "first_gate_cut": int(CUTS[np.flatnonzero(gate > 0)[0]]) if np.any(gate > 0) else None,
                                   "max_correction": float(increments["oor_pga_like"][-1]),
                                   "max_late_correction": float(increments["late_oor_pga_like"][-1])})
            for seed in SEEDS:
                for method in METHODS:
                    output = direction / f"seed_{seed}" / method
                    output.mkdir(parents=True)
                    baseline = baselines[source, target, seed, method]
                    table = pd.DataFrame({"cut_index": CUTS, "baseline": baseline,
                                          "oor_rate": rate, "gate": gate,
                                          "source_trend_increment": increment,
                                          "oor_pga_factor": factor,
                                          "oor_pga": baseline * factor,
                                          "oor_pga_like": baseline + increments["oor_pga_like"],
                                          "late_oor_pga_like": baseline + increments["late_oor_pga_like"]})
                    table.to_csv(output / "predictions_before_evaluation.csv", index=False, float_format="%.17g")
    (args.out_root / "parameters.json").write_text(json.dumps({
        "reference": ["EEMD/run_oor_triggered_pga.py: multiplicative OOR-PGA",
                      "EEMD/run_oor_adaptive_pga_like.py: OOR-weighted cumulative additive PGA-like"],
        "feature": "48 means of log1p STFT: 6 channels x 8 frequency bands",
        "tl": args.tl, "th": args.th, "lambda": args.lambda_pga,
        "late_start": args.late_start, "late_start_provenance": "exploratory choice informed by earlier target-stage diagnosis; not independent validation",
        "trained_root": str(args.trained_root.resolve()),
        "prediction_root": str(args.prediction_root.resolve()),
        "feature_manifest_sha256": digest(args.trained_root / "feature_cache" / "manifest.json"),
        "directions": parameter_rows,
        "target_label_use_before_prediction_freeze": False}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Evaluate all frozen methods using labels from the existing, checked full-cut CSV.
    rows = []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                for method in METHODS:
                    prediction_file = args.prediction_root / f"{source}_to_{target}" / f"seed_{seed}" / method / "predictions.csv"
                    existing = pd.read_csv(prediction_file)
                    if list(existing.columns) != ["cut_index", "true_vb", "pred_vb"] or existing.cut_index.tolist() != CUTS.tolist():
                        raise ValueError(f"Invalid evaluation CSV: {prediction_file}")
                    baseline = baselines[source, target, seed, method]
                    if not np.allclose(existing.pred_vb.to_numpy(dtype=np.float64), baseline, atol=1e-12, rtol=0):
                        raise ValueError(f"Baseline prediction mismatch: {prediction_file}")
                    truth = existing.true_vb.to_numpy(dtype=np.float64)
                    if not np.isfinite(truth).all():
                        raise ValueError(f"Nonfinite target labels: {prediction_file}")
                    output = args.out_root / f"{source}_to_{target}" / f"seed_{seed}" / method
                    corrected = pd.read_csv(output / "predictions_before_evaluation.csv")
                    for name in ("baseline", "oor_pga", "oor_pga_like", "late_oor_pga_like"):
                        pred = corrected[name].to_numpy(dtype=np.float64)
                        masks = {"full_1_315": np.ones(315, bool),
                                 "early_1_105": CUTS <= 105, "middle_106_210": (CUTS >= 106) & (CUTS <= 210),
                                 "late_211_315": CUTS >= 211}
                        if target == "c6":
                            masks["original_95_315"] = CUTS >= 95
                        for scope, mask in masks.items():
                            rows.append({"source": source, "target": target, "seed": seed,
                                         "base_method": method, "correction": name, "scope": scope,
                                         **metrics(truth, pred, mask)})
                    corrected.insert(1, "true_vb", truth)
                    corrected.to_csv(output / "evaluated_predictions.csv", index=False, float_format="%.17g")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.out_root / "per_seed_metrics.csv", index=False)
    keys = ["source", "target", "base_method", "correction", "scope"]
    summary = per_seed.groupby(keys, as_index=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        bias_mean=("bias", "mean"), bias_sd=("bias", "std"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"))
    baseline = per_seed.loc[per_seed.correction == "baseline", [*keys[:3], "seed", "scope", "RMSE"]].rename(columns={"RMSE": "baseline_RMSE"})
    paired = per_seed.merge(baseline, on=[*keys[:3], "seed", "scope"], validate="many_to_one")
    paired["delta_RMSE"] = paired.RMSE - paired.baseline_RMSE
    paired.groupby(keys, as_index=False).agg(delta_RMSE_mean=("delta_RMSE", "mean"),
                                               delta_RMSE_sd=("delta_RMSE", "std"),
                                               seeds_improved=("delta_RMSE", lambda value: int((value < 0).sum()))) \
        .merge(summary, on=keys).to_csv(args.out_root / "summary.csv", index=False)
    for source in TOOLS:
        for target in TOOLS:
            if source != target:
                plot_direction(args, source, target)
    print(f"Saved frozen predictions and paired metrics to {args.out_root}")
    print("Exploratory comparison only: no six-direction results can be inferred without the local artifacts.")


if __name__ == "__main__":
    main()
