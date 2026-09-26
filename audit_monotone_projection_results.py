"""Audit zero-delta training, label-free per-seed PAVA, and five-seed mean curves."""

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
from sklearn.isotonic import IsotonicRegression

import run_five_seed_pairs as five
import run_single_source_pairs as base
from run_monotone_projection_comparison import DEFAULT_OUT, LAMBDA, measures, write_json
from trend_physics import pava_nondecreasing


def draw(path, source, target, method, truth, curve, label):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(base.ALL_CUTS, truth, color="black", lw=1.8, label="True VB")
    ax.plot(base.ALL_CUTS, curve, color="#b12d58" if label.startswith("Monotone") else "#2563a6",
            lw=1.5, label=label)
    ax.set(xlabel="Target cut", ylabel="VB", title=f"{source.upper()}->{target.upper()} {method}: five-seed mean")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    records, missing = [], []
    plot_dir = args.result_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    smoke = {"direction": "c1_to_c6", "criterion": "five-seed mean monotone RMSE <= 1.25 times raw RMSE",
             "methods": {}, "pass": None}
    for source, target in five.PAIRS:
        for method in ("source_only", "daregram"):
            raw_seeds, mono_seeds, truth = [], [], None
            for seed in five.SEEDS:
                folder = args.result_root / f"{source}_to_{target}" / f"seed_{seed}"
                if not (folder / "complete.json").is_file():
                    missing.append(f"{source}_to_{target}/seed_{seed}")
                    continue
                cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                if (cfg["trend_delta_vb"] != 0 or cfg["trend_lambda"] != LAMBDA or
                    cfg["target_label_training_reads"] != 0 or cfg["source_only_training_target_cuts"] != [] or
                    cfg["daregram_training_target_cuts"] != base.ALL_CUTS):
                    raise ValueError(f"Protocol mismatch: {folder}")
                sub = folder / method
                frame = pd.read_csv(sub / "predictions.csv", float_precision="round_trip")
                if list(frame.columns) != ["cut_index", "true_vb", "raw_pred", "monotone_pred"] or frame.cut_index.tolist() != base.ALL_CUTS:
                    raise ValueError(f"Prediction schema/cuts mismatch: {sub}")
                values = frame[["true_vb", "raw_pred", "monotone_pred"]].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite prediction/label: {sub}")
                y, raw, mono = values.T
                if truth is None:
                    truth = y
                elif not np.allclose(truth, y, rtol=0, atol=1e-6):
                    raise ValueError(f"Target labels differ across seeds: {sub}")
                if np.any(np.diff(mono) < -1e-8):
                    raise ValueError(f"Hard monotonicity failed: {sub}")
                if not np.isclose(mono.mean(), raw.mean(), rtol=0, atol=1e-10):
                    raise ValueError(f"Equal-weight PAVA should preserve the global prediction mean: {sub}")
                reference = IsotonicRegression(increasing=True).fit_transform(np.asarray(base.ALL_CUTS), raw)
                if not np.allclose(mono, pava_nondecreasing(raw), rtol=0, atol=1e-10) or not np.allclose(mono, reference, rtol=0, atol=1e-10):
                    raise ValueError(f"PAVA projection differs from L2 isotonic solution: {sub}")
                pre = json.loads((sub / "projection_prelabel.json").read_text(encoding="utf-8"))
                if pre["target_label_reads_so_far"] != 0 or pre["cut_index"] != base.ALL_CUTS or pre["minimum_adjacent_difference_vb"] < -1e-8:
                    raise ValueError(f"Prelabel projection provenance mismatch: {sub}")
                if (hashlib.sha256(raw.tobytes()).hexdigest() != pre["raw_prediction_sha256"] or
                    hashlib.sha256(mono.tobytes()).hexdigest() != pre["monotone_prediction_sha256"]):
                    raise ValueError(f"Prediction changed after projection: {sub}")
                result = json.loads((sub / "metrics_projection.json").read_text(encoding="utf-8"))
                check = measures(y, raw, mono)
                for metric_group in ("raw", "monotone"):
                    for name in ("R2", "MAE", "RMSE"):
                        if not np.isclose(result[metric_group][name], check[metric_group][name], atol=1e-9, rtol=1e-9):
                            raise ValueError(f"Metric mismatch {metric_group}/{name}: {sub}")
                    physical = pd.read_csv(sub / f"physical_{metric_group}_per_cut.csv")
                    expected_pred = raw if metric_group == "raw" else mono
                    if (f"{metric_group}_pred" not in physical or
                        physical.cut_index.tolist() != base.ALL_CUTS or
                        not np.allclose(physical[f"{metric_group}_pred"], expected_pred, rtol=0, atol=1e-10)):
                        raise ValueError(f"Physical per-cut prediction mismatch: {sub}/{metric_group}")
                loss = pd.read_csv(sub / "epoch_losses.csv")
                if (len(loss) != five.EPOCHS or not (loss.trend_weight == LAMBDA).all() or
                    not (loss.source_valid_pairs > 0).all() or not (loss.supervised_mse > 0).all() or
                    not (loss.trend_grad_active_steps > 0).all() or
                    not (loss.trend_grad_norm_regressor_weight > 0).all() or
                    (method == "daregram" and not (loss.target_valid_pairs > 0).all()) or
                    (method == "source_only" and (loss.target_valid_pairs != 0).any())):
                    raise ValueError(f"Trend training gradient/pair audit failed: {sub}")
                trace = json.loads((sub / "training_trace.json").read_text(encoding="utf-8"))
                if trace["actual_unlabeled_target_cuts"] != (base.ALL_CUTS if method == "daregram" else []):
                    raise ValueError(f"Target input use mismatch: {sub}")
                raw_seeds.append(raw)
                mono_seeds.append(mono)
                records.append({"source": source, "target": target, "seed": seed, "method": method,
                                "minimum_monotone_step_vb": float(np.diff(mono).min()),
                                "raw_RMSE": result["raw"]["RMSE"], "monotone_RMSE": result["monotone"]["RMSE"],
                                "mean_abs_output_change_vb": result["mean_abs_output_change_vb"],
                                "max_abs_output_change_vb": result["max_abs_output_change_vb"],
                                "cut_250_315_raw_bias_vb": result["cut_250_315"]["raw_mean_signed_error_vb"],
                                "cut_250_315_monotone_bias_vb": result["cut_250_315"]["monotone_mean_signed_error_vb"],
                                "trend_gradient_active_epochs": int((loss.trend_grad_active_steps > 0).sum())})
            if len(raw_seeds) != len(five.SEEDS):
                continue
            raw_mean = np.mean(np.stack(raw_seeds), axis=0)
            mono_mean = np.mean(np.stack(mono_seeds), axis=0)
            if np.any(np.diff(mono_mean) < -1e-8):
                raise ValueError(f"Five-seed mean is not monotone: {source}->{target} {method}")
            stem = f"{source}_to_{target}_{method}"
            pd.DataFrame({"cut_index": base.ALL_CUTS, "true_vb": truth,
                          "raw_mean": raw_mean, "monotone_mean": mono_mean,
                          "raw_seed_sd": np.std(raw_seeds, axis=0, ddof=1),
                          "monotone_seed_sd": np.std(mono_seeds, axis=0, ddof=1)}).to_csv(
                              plot_dir / f"{stem}_curves.csv", index=False, float_format="%.17g")
            draw(plot_dir / f"{stem}_raw.png", source, target, method, truth, raw_mean, "Raw prediction")
            draw(plot_dir / f"{stem}_monotone.png", source, target, method, truth, mono_mean,
                 "Monotone-projected prediction")
            if (source, target) == ("c1", "c6"):
                pair = [row for row in records if (row["source"], row["target"], row["method"]) == (source, target, method)]
                raw_rmse = float(np.mean([row["raw_RMSE"] for row in pair]))
                mono_rmse = float(np.mean([row["monotone_RMSE"] for row in pair]))
                smoke["methods"][method] = {"raw_RMSE_mean": raw_rmse, "monotone_RMSE_mean": mono_rmse,
                                             "ratio": mono_rmse / raw_rmse,
                                             "pass": bool(mono_rmse <= 1.25 * raw_rmse)}
    if len(smoke["methods"]) == 2:
        smoke["pass"] = all(item["pass"] for item in smoke["methods"].values())
    write_json(args.result_root / "smoke_gate.json", smoke)
    pd.DataFrame(records).to_csv(args.result_root / "validation.csv", index=False)
    write_json(args.result_root / "validation.json", {"completed_method_runs": len(records),
               "expected_method_runs": 60, "missing_pair_seeds": sorted(set(missing)),
               "checks_pass_for_completed_runs": True, "smoke_gate": smoke})
    print(f"Validated {len(records)}/60 method runs, missing pair-seeds={len(set(missing))}, smoke={smoke['pass']}")


if __name__ == "__main__":
    main()
