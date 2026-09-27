"""Five-seed, label-isolated diagnosis of the frozen C1->C6 fusion.

predict uses C6 wear rows 1..5 only; score reads rows 6..315 after SHA locking.
No model training or checkpoint mutation occurs.
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

import run_c6_five_point_model_fusion as fusion
import run_multi_window_ablation as experiment

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "artifacts/multi_window_25_50_75_20260926"
DEFAULT_OUT = ROOT / "artifacts/c6_five_point_five_seed_diagnostic_20260926"
SEEDS = (42, 43, 44, 45, 46)
DETAIL_CUTS = (6, 10, 50, 100, 200, 315)
ALPHA, DELTA, DECAY = 0.1, 2.0, 20.0
DIVERGENCE_ABS_VB = 1000.0


def frozen_model_predictions(seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    folder = EXPERIMENT / "c1_to_c6" / f"seed_{seed}"
    config_path = folder / "config.json"
    checkpoint = folder / "daregram/final.pth"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["source"], config["target"], config["seed"]) != ("c1", "c6", seed):
        raise ValueError(f"Wrong run configuration for seed {seed}")
    digest = fusion.sha256(checkpoint)
    if digest != config["traces"]["daregram"]["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
    mean = np.asarray(config["source_zscore_mean"], dtype=np.float32)
    std = np.asarray(config["source_zscore_std"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,) or (std <= 0).any():
        raise ValueError("Invalid source normalization")
    cache = experiment.cache_view(EXPERIMENT, "c6")
    windows = experiment.predict_windows("daregram", cache, mean, std, seed, folder, device)
    if windows.shape != (315, 3) or not np.isfinite(windows).all():
        raise ValueError(f"Invalid inference shape for seed {seed}")
    cut_mean = windows.mean(axis=1)
    if seed == 42:
        _, audited = fusion.model_predictions()
        if not np.allclose(cut_mean, audited, atol=1e-4, rtol=0):
            raise ValueError("Seed 42 inference disagrees with label-free checkpoint audit")
    return cut_mean, {"seed": seed, "checkpoint_sha256": digest,
                      "configuration_sha256": fusion.sha256(config_path),
                      "seed_42_audit_max_abs_difference": float(np.max(np.abs(cut_mean - audited)))
                      if seed == 42 else None}


def assert_history(trace: pd.DataFrame) -> None:
    if trace.cut_index.tolist() != fusion.CUTS:
        raise ValueError("Cut order must be 6..315")
    for row in trace.itertuples(index=False):
        cut = int(row.cut_index)
        if (row.input_cut_start, row.input_cut_end) != (cut - 5, cut - 1):
            raise ValueError(f"Wrong five-cut window at {cut}")
        for j in range(1, 6):
            prior_cut = getattr(row, f"input_cut_{j}")
            expected_source = "true_initial" if prior_cut <= 5 else "predicted"
            if prior_cut != cut - 6 + j or getattr(row, f"input_source_{j}") != expected_source:
                raise ValueError(f"Wrong window source/order at {cut}")
            if prior_cut > 5:
                prior = trace.iloc[prior_cut - 6]
                if not np.isclose(getattr(row, f"input_vb_{j}"), prior.pred_vb, atol=1e-12):
                    raise ValueError(f"Predicted VB not appended before cut {cut}")


def predict(args: argparse.Namespace) -> None:
    out = args.out_root
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if fusion.sha256(fusion.CHECKPOINT) != fusion.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Original seed-42 checkpoint changed")
    prior_parameters = json.loads((fusion.DEFAULT_OUTPUT / "selected_parameters.json").read_text(encoding="utf-8"))
    if (prior_parameters["alpha"], prior_parameters["delta"]) != (ALPHA, DELTA):
        raise ValueError("Current fusion parameters changed")
    manifest = json.loads((EXPERIMENT / "feature_cache/manifest.json").read_text(encoding="utf-8"))
    cache_path = EXPERIMENT / "feature_cache/c6_stft.npy"
    if fusion.sha256(cache_path) != manifest["tools"]["c6"]["feature_sha256"]:
        raise ValueError("Unlabeled C6 feature cache changed")
    initial_rows = fusion.read_initial_five(args.target_wear)
    initial = {cut: row["vb"] for cut, row in initial_rows.items()}
    device = torch.device(args.device)
    traces, provenance, offsets = [], [], []
    for seed in SEEDS:
        model, info = frozen_model_predictions(seed, device)
        trace = fusion.recurse(initial, model, ALPHA, DELTA)
        assert_history(trace)
        raw = trace.q_t.to_numpy(float) - trace.b_t.to_numpy(float)
        clipped = np.clip(raw, -DELTA, DELTA)
        correction = trace.pred_vb.to_numpy(float) - trace.b_t.to_numpy(float)
        if not np.allclose(trace.clipped_q_minus_b, clipped, atol=1e-12, rtol=0) or \
                not np.allclose(correction, ALPHA * clipped, atol=1e-12, rtol=0):
            raise ValueError(f"Fusion arithmetic mismatch at seed {seed}")
        r = float(np.median(np.asarray(list(initial.values())) - model[:5]))
        calibrated = model[5:] + r * np.exp(-(np.asarray(fusion.CUTS) - 5) / DECAY)
        trace.insert(0, "seed", seed)
        trace["raw_q_minus_b"] = raw
        trace["clip_triggered"] = np.abs(raw) > DELTA
        trace["actual_fusion_correction"] = correction
        trace["initial_offset_r"] = r
        trace["short_calibration_pred_vb"] = calibrated
        traces.append(trace)
        provenance.append(info)
        offsets.append({"seed": seed, "initial_offset_r": r,
                        "initial_vb_unit": "mean of three flute wear values, same VB unit as model"})
    combined = pd.concat(traces, ignore_index=True)
    if len(combined) != 5 * 310:
        raise ValueError("Expected 1550 cut predictions")
    pure = fusion.recurse(initial, [0.0] * 5 + traces[0].b_t.to_numpy(float).tolist(),
                          1.0, float("inf"))
    # In this control, the model array is irrelevant: alpha=1 and no clipping
    # algebraically returns q_t, which is then appended to the next window.
    first_negative = int(pure.loc[pure.pred_vb < 0, "cut_index"].iloc[0])
    above = pure.loc[pure.pred_vb.abs() > DIVERGENCE_ABS_VB, "cut_index"]
    first_explosion = int(above.iloc[0]) if len(above) else None
    out.mkdir(parents=True)
    files = {"unscored_predictions_sha256": out / "unscored_predictions.csv",
             "selected_cuts_sha256": out / "selected_cuts_unscored.csv",
             "pure_recursive_sha256": out / "pure_recursive_unscored.csv",
             "initial_five_sha256": out / "initial_five.csv",
             "checkpoint_provenance_sha256": out / "checkpoint_provenance.json"}
    combined.to_csv(files["unscored_predictions_sha256"], index=False, float_format="%.17g")
    combined.loc[combined.cut_index.isin(DETAIL_CUTS)].to_csv(
        files["selected_cuts_sha256"], index=False, float_format="%.17g")
    pure[["cut_index", "pred_vb"]].rename(columns={"pred_vb": "pure_five_point_q_t"}).to_csv(
        files["pure_recursive_sha256"], index=False, float_format="%.17g")
    pd.DataFrame({"cut_index": list(initial), "true_vb_initial": list(initial.values())}).to_csv(
        files["initial_five_sha256"], index=False, float_format="%.17g")
    fusion.write_json(files["checkpoint_provenance_sha256"], {"runs": provenance, "offsets": offsets})
    fusion.write_json(out / "prediction_lock.json", {
        "prediction_complete": True, "seeds": list(SEEDS), "prediction_cuts": fusion.CUTS,
        "C6_true_VB_read_before_prediction_complete": [1, 2, 3, 4, 5],
        "target_suffix_labels_read_before_prediction_complete": False,
        "alpha": ALPHA, "delta": DELTA, "calibration_decay_cuts": DECAY,
        "unclipped_divergence_diagnostic_abs_vb": DIVERGENCE_ABS_VB,
        "pure_first_negative_cut": first_negative,
        "pure_first_abs_gt_1000_cut": first_explosion,
        **{key: fusion.sha256(path) for key, path in files.items()},
        "cache_manifest_sha256": fusion.sha256(EXPERIMENT / "feature_cache/manifest.json"),
        "seed_42_audited_prediction_sha256": fusion.sha256(fusion.PREDICTIONS),
        "prior_selected_parameters_sha256": fusion.sha256(fusion.DEFAULT_OUTPUT / "selected_parameters.json")})
    print(combined.loc[combined.cut_index.isin(DETAIL_CUTS),
                       ["seed", "cut_index", "b_t", "q_t", "raw_q_minus_b", "clipped_q_minus_b",
                        "actual_fusion_correction", "pred_vb", "short_calibration_pred_vb"]].to_string(index=False))
    print(f"pure five-point: first negative cut {first_negative}; first |q|>1000 cut {first_explosion}")


def score(args: argparse.Namespace) -> None:
    out = args.out_root
    if (out / "metrics_by_seed.csv").exists():
        raise FileExistsError("Scored results already exist")
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    if not lock["prediction_complete"] or lock["seeds"] != list(SEEDS) or \
            lock["prediction_cuts"] != fusion.CUTS or lock["target_suffix_labels_read_before_prediction_complete"]:
        raise ValueError("Incomplete prediction lock")
    paths = {"unscored_predictions_sha256": out / "unscored_predictions.csv",
             "selected_cuts_sha256": out / "selected_cuts_unscored.csv",
             "pure_recursive_sha256": out / "pure_recursive_unscored.csv",
             "initial_five_sha256": out / "initial_five.csv",
             "checkpoint_provenance_sha256": out / "checkpoint_provenance.json",
             "cache_manifest_sha256": EXPERIMENT / "feature_cache/manifest.json",
             "seed_42_audited_prediction_sha256": fusion.PREDICTIONS,
             "prior_selected_parameters_sha256": fusion.DEFAULT_OUTPUT / "selected_parameters.json"}
    for key, path in paths.items():
        if fusion.sha256(path) != lock[key]:
            raise ValueError(f"Locked file changed: {path}")
    provenance = json.loads((out / "checkpoint_provenance.json").read_text(encoding="utf-8"))
    for entry in provenance["runs"]:
        folder = EXPERIMENT / "c1_to_c6" / f"seed_{entry['seed']}"
        if fusion.sha256(folder / "daregram/final.pth") != entry["checkpoint_sha256"] or \
                fusion.sha256(folder / "config.json") != entry["configuration_sha256"]:
            raise ValueError("Frozen checkpoint/configuration changed")
    frame = pd.read_csv(out / "unscored_predictions.csv")
    for seed in SEEDS:
        part = frame.loc[frame.seed == seed]
        if part.cut_index.tolist() != fusion.CUTS or not np.isfinite(
                part[["b_t", "q_t", "pred_vb", "short_calibration_pred_vb"]].to_numpy(float)).all():
            raise ValueError(f"Invalid predictions for seed {seed}")
    # Only now may C6 cuts 6..315 be read.
    truth = fusion.read_all_wear_after_predictions(args.target_wear)
    initial = pd.read_csv(out / "initial_five.csv")
    if initial.cut_index.tolist() != list(range(1, 6)) or any(
            not np.isclose(row.true_vb_initial, truth[row.cut_index], atol=1e-11)
            for row in initial.itertuples(index=False)):
        raise ValueError("Initial five labels differ")
    frame["true_vb"] = [truth[cut] for cut in frame.cut_index]
    frame.to_csv(out / "predictions_scored.csv", index=False, float_format="%.17g")
    methods = (("daregram", "b_t"), ("five_point_fusion", "pred_vb"),
               ("short_initial_calibration", "short_calibration_pred_vb"))
    rows = []
    for seed in SEEDS:
        part = frame.loc[frame.seed == seed]
        for method, column in methods:
            record = {"seed": seed, "method": method,
                      **fusion.metrics(part.true_vb.to_numpy(float), part[column].to_numpy(float))}
            for name, start, end in fusion.SEGMENTS[1:]:
                segment = part.loc[part.cut_index.between(start, end)]
                record[f"{name}_signed_error"] = float(
                    (segment[column] - segment.true_vb).mean())
            rows.append(record)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False, float_format="%.17g")
    summary = metrics.groupby("method", sort=False).agg(
        **{f"{column}_{stat}": (column, stat) for column in
           ("R2", "MAE", "RMSE", "early_6_105_signed_error",
            "middle_106_210_signed_error", "late_211_315_signed_error")
           for stat in ("mean", "std")}).reset_index()
    summary.to_csv(out / "metrics_five_seed_summary.csv", index=False, float_format="%.17g")
    diagnostics = []
    for seed in SEEDS:
        part = frame.loc[frame.seed == seed]
        abs_raw = part.raw_q_minus_b.abs()
        abs_correction = part.actual_fusion_correction.abs()
        record = {"seed": seed, "clip_count": int(part.clip_triggered.sum()),
                  "clip_fraction": float(part.clip_triggered.mean()),
                  "mean_abs_fusion_correction": float(abs_correction.mean()),
                  "max_abs_fusion_correction": float(abs_correction.max()),
                  "fused_q_min": float(part.q_t.min()), "fused_q_max": float(part.q_t.max()),
                  "first_fused_abs_q_gt_1000": int(part.loc[part.q_t.abs() > 1000, "cut_index"].iloc[0])
                  if (part.q_t.abs() > 1000).any() else None,
                  "first_abs_q_minus_b_gt_2": int(part.loc[abs_raw > DELTA, "cut_index"].iloc[0])
                  if (abs_raw > DELTA).any() else None}
        for name, start, end in fusion.SEGMENTS[1:]:
            segment = part.loc[part.cut_index.between(start, end)]
            record[f"{name}_clip_fraction"] = float(segment.clip_triggered.mean())
        diagnostics.append(record)
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(out / "fusion_diagnostics_by_seed.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True, constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        part = frame.loc[frame.seed == seed]
        ax.plot(part.cut_index, part.true_vb, color="black", lw=1.4, label="C6 true VB")
        ax.plot(part.cut_index, part.b_t, lw=1, label="DARE-GRAM")
        ax.plot(part.cut_index, part.pred_vb, lw=1, label="five-point fusion")
        ax.plot(part.cut_index, part.short_calibration_pred_vb, lw=1,
                label="short initial calibration")
        ax.set(ylabel=f"seed {seed}\nVB")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("C6 cut")
    fig.savefig(out / "five_seed_prediction_curves.png", dpi=180)
    plt.close(fig)
    fusion.write_json(out / "score_audit.json", {
        "target_suffix_first_read": "after five-seed predictions were complete and SHA-256 locked",
        "target_wear_sha256_after_prediction_complete": fusion.sha256(args.target_wear),
        "true_VB_unit": "mean of flute_1, flute_2, flute_3; same unit as model output",
        "pure_five_point_first_negative_cut": lock["pure_first_negative_cut"],
        "pure_five_point_first_abs_gt_1000_cut": lock["pure_first_abs_gt_1000_cut"],
        "fused_q_divergence_threshold_abs_vb": 1000.0,
        "parameters_fixed_before_target_scoring": {"alpha": ALPHA, "delta": DELTA,
                                                    "calibration_decay_cuts": DECAY}})
    print(summary.to_string(index=False))
    print(diagnostic.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score"))
    parser.add_argument("--target-wear", type=Path,
                        default=Path(r"E:\QLP\source\source_mill\c6_wear.csv"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
