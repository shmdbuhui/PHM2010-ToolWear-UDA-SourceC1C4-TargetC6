"""Five recent current-cut wear estimates -> next-cut linear extrapolation.

This read-only reuses the C1->C6 seed-42 current-cut source-only checkpoint.
Inference and scoring are separate stages to keep C6 wear out of fitting.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import run_single_source_pairs as base

CUTS = list(range(1, 316))
PREDICTION_CUTS = list(range(6, 316))
STAGES = (("early_6_105", 6, 105), ("middle_106_210", 106, 210),
          ("late_211_315", 211, 315))
METHODS = ("persistence", "local_linear", "k5_gru_source_only")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("infer", "score", "all"), default="all")
    parser.add_argument("--out-root", type=Path,
                        default=Path("artifacts/local_five_pred_current_extrapolation_c1_c6_seed42_20260926"))
    parser.add_argument("--current-run", type=Path,
                        default=Path("artifacts/five_seed_paired/c1_to_c6/seed_42"))
    parser.add_argument("--gru-run", type=Path,
                        default=Path("artifacts/historical_next_cut_seed42_20260926/c1_to_c6/seed_42"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("artifacts/five_seed_paired/feature_cache"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def current_artifacts(args):
    config_path = args.current_run / "config.json"
    method_config_path = args.current_run / "source_only" / "config.json"
    checkpoint_path = args.current_run / "source_only" / "final.pth"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method_config = json.loads(method_config_path.read_text(encoding="utf-8"))
    assert (config["source"], config["target"], config["seed"]) == ("c1", "c6", 42)
    assert config["source_count"] == 315 and config["source_cuts"] == CUTS
    assert config["backbone"] == "ResNet18" and config["regressor"] == "Linear(512, 1)"
    assert method_config["actual_unlabeled_target_cuts"] == []
    assert config["stft_code_sha256"] == base.file_hash(Path(base.sampling.__file__))
    assert str(args.raw_root.resolve()).lower() == config["raw_root"].lower()
    assert config["target_feature_sha256"] == base.file_hash(args.cache_root / "c6_stft.npy")
    return config, checkpoint_path, base.file_hash(checkpoint_path)


def predict_current(args, config, checkpoint_path):
    cache_manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    assert cache_manifest["cuts"] == CUTS
    assert cache_manifest["tools"]["c6"]["feature_sha256"] == config["target_feature_sha256"]
    raw = np.load(args.cache_root / "c6_stft.npy", mmap_mode="r", allow_pickle=False)
    assert raw.shape == (315, 6, 128, 128) and raw.dtype == np.float32
    mean = np.asarray(config["source_normalization_mean"], dtype=np.float32)
    std = np.asarray(config["source_normalization_std"], dtype=np.float32)
    assert mean.shape == std.shape == (6,) and np.all(std > 0)
    model, _ = base.build_model(42, torch.device(args.device))
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    assert checkpoint["epoch"] == 50
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, 315, 63):
            # Each current-cut prediction receives exactly that cut's STFT.
            x = np.asarray(raw[offset:offset + 63])
            x = ((x - mean[None, :, None, None]) /
                 (std[None, :, None, None] + 1e-8)).astype(np.float32)
            tensor = torch.from_numpy(x).to(args.device)
            output = model.regressor(model.feature_extractor(tensor))
            values.extend(output.cpu().numpy().reshape(-1).tolist())
    pred = np.asarray(values, dtype=np.float64)
    assert pred.shape == (315,) and np.isfinite(pred).all()
    return pred


def fit_history(pred_current):
    rows = []
    for prediction_cut in PREDICTION_CUTS:
        input_cuts = np.arange(prediction_cut - 5, prediction_cut, dtype=np.float64)
        input_estimates = pred_current[input_cuts.astype(int) - 1]
        assert len(input_cuts) == len(input_estimates) == 5
        assert int(input_cuts[-1]) + 1 == prediction_cut
        assert np.all(input_cuts < prediction_cut)
        assert np.array_equal(input_cuts, np.arange(prediction_cut - 5, prediction_cut))
        # Ordinary least squares with an intercept and actual cut indices.
        design = np.column_stack((input_cuts, np.ones(5, dtype=np.float64)))
        slope, intercept = np.linalg.lstsq(design, input_estimates, rcond=None)[0]
        row = {"input_cut_start": int(input_cuts[0]),
               "input_cut_end": int(input_cuts[-1]),
               "prediction_cut": prediction_cut}
        row.update({f"input_pred_{j}": float(value) for j, value in enumerate(input_estimates, 1)})
        row.update({"ols_slope": float(slope), "ols_intercept": float(intercept),
                    "pred_persistence": float(input_estimates[-1]),
                    "pred_local_linear": float(slope * prediction_cut + intercept)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame) == 310 and frame.prediction_cut.tolist() == PREDICTION_CUTS
    assert frame.iloc[0][["input_cut_start", "input_cut_end", "prediction_cut"]].tolist() == [1, 5, 6]
    assert frame.iloc[-1][["input_cut_start", "input_cut_end", "prediction_cut"]].tolist() == [310, 314, 315]
    assert (frame.input_cut_end + 1 == frame.prediction_cut).all()
    return frame


def infer(args):
    if args.out_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {args.out_root}")
    # Validation and model predictions happen before any C6 wear read.
    config, checkpoint_path, checkpoint_hash = current_artifacts(args)
    pred_current = predict_current(args, config, checkpoint_path)
    frame = fit_history(pred_current)
    args.out_root.mkdir(parents=True)
    current_path = args.out_root / "pred_current_unscored.csv"
    forecast_path = args.out_root / "forecast_unscored.csv"
    pd.DataFrame({"cut_index": CUTS, "pred_current": pred_current}).to_csv(
        current_path, index=False, float_format="%.17g")
    frame.to_csv(forecast_path, index=False, float_format="%.17g")
    write_json(args.out_root / "inference_audit.json", {
        "model_role": "current-cut X_i -> VB_i source-only estimator; C1 labels in original training",
        "source": "c1", "target": "c6", "seed": 42,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "config_sha256": base.file_hash(args.current_run / "config.json"),
        "target_feature_sha256": config["target_feature_sha256"],
        "normalization": "original C1 source per-channel z-score",
        "prediction_cuts": PREDICTION_CUTS,
        "first_window": {"inputs": [1, 2, 3, 4, 5], "prediction_cut": 6},
        "last_window": {"inputs": [310, 311, 312, 313, 314], "prediction_cut": 315},
        "future_values_used_in_each_fit": 0, "target_wear_reads_before_forecast_fixed": 0,
        "pred_current_sha256": base.file_hash(current_path),
        "forecast_unscored_sha256": base.file_hash(forecast_path),
    })
    print("First window: [1, 2, 3, 4, 5] -> 6")
    print("Last window: [310, 311, 312, 313, 314] -> 315")


def score(args):
    out = args.out_root
    final_path = out / "predictions_per_cut.csv"
    if final_path.exists() or (out / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite scored outputs in {out}")
    audit = json.loads((out / "inference_audit.json").read_text(encoding="utf-8"))
    assert base.file_hash(out / "pred_current_unscored.csv") == audit["pred_current_sha256"]
    assert base.file_hash(out / "forecast_unscored.csv") == audit["forecast_unscored_sha256"]
    config, checkpoint_path, checkpoint_hash = current_artifacts(args)
    assert checkpoint_hash == audit["checkpoint_sha256"]
    assert base.file_hash(args.current_run / "config.json") == audit["config_sha256"]
    current = pd.read_csv(out / "pred_current_unscored.csv")
    frame = pd.read_csv(out / "forecast_unscored.csv")
    assert current.cut_index.tolist() == CUTS
    assert frame.prediction_cut.tolist() == PREDICTION_CUTS
    for prediction_cut in PREDICTION_CUTS:
        row = frame.iloc[prediction_cut - 6]
        xs = np.arange(prediction_cut - 5, prediction_cut)
        ys = current.pred_current.to_numpy()[xs - 1]
        assert np.allclose(row[[f"input_pred_{j}" for j in range(1, 6)]].to_numpy(dtype=float), ys, atol=1e-12)
        assert row.input_cut_start == xs[0] and row.input_cut_end == xs[-1]
        assert np.isclose(row.pred_persistence, ys[-1], atol=1e-12)
        assert np.isclose(row.pred_local_linear, row.ols_slope * prediction_cut + row.ols_intercept, atol=1e-10)
    gru_config = json.loads((args.gru_run / "config.json").read_text(encoding="utf-8"))
    assert (gru_config["source"], gru_config["target"], gru_config["seed"]) == ("c1", "c6", 42)
    gru_checkpoint = args.gru_run / "k5_source_only" / "final.pth"
    gru_prediction_file = args.gru_run / "k5_source_only" / "predictions_all_cuts.csv"
    assert gru_checkpoint.is_file() and gru_prediction_file.is_file()
    gru = pd.read_csv(gru_prediction_file, usecols=["cut_index", "pred_vb"])
    assert gru.cut_index.tolist() == CUTS and gru.loc[:4, "pred_vb"].isna().all()
    assert gru.loc[5:, "pred_vb"].notna().all()
    frame["pred_k5_gru_source_only"] = gru.loc[5:, "pred_vb"].to_numpy(dtype=np.float64)
    # The first C6 wear read in this runner occurs only after the unscored
    # current and next-cut predictions have been written and hash-locked.
    y = base.wear_labels(args.raw_root, "c6")[5:].astype(np.float64)
    frame["true_vb"] = y
    frame.to_csv(final_path, index=False, float_format="%.17g")
    scored = pd.read_csv(final_path)
    assert scored.prediction_cut.tolist() == PREDICTION_CUTS
    columns = {"persistence": "pred_persistence", "local_linear": "pred_local_linear",
               "k5_gru_source_only": "pred_k5_gru_source_only"}
    results = {}
    for method, col in columns.items():
        pred = scored[col].to_numpy(dtype=np.float64)
        truth = scored.true_vb.to_numpy(dtype=np.float64)
        signed = pred - truth
        stages = {name: {"cuts": f"{start}..{end}",
                         "n": int(((scored.prediction_cut >= start) & (scored.prediction_cut <= end)).sum()),
                         "mean_signed_error": float(signed[(scored.prediction_cut >= start) &
                                                             (scored.prediction_cut <= end)].mean())}
                  for name, start, end in STAGES}
        results[method] = {"cuts": "6..315", "n": 310,
                           "R2": float(r2_score(truth, pred)),
                           "MAE": float(mean_absolute_error(truth, pred)),
                           "RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
                           "mean_signed_error": float(signed.mean()), "stages": stages}
    write_json(out / "metrics.json", results)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(scored.prediction_cut, scored.true_vb, color="black", linewidth=1.6, label="C6 true VB")
    ax.plot(scored.prediction_cut, scored.pred_persistence, linewidth=1, label="K=1 persistence")
    ax.plot(scored.prediction_cut, scored.pred_local_linear, linewidth=1, label="K=5 local OLS")
    ax.plot(scored.prediction_cut, scored.pred_k5_gru_source_only, linewidth=1, label="K=5 ResNet18+GRU")
    ax.set(xlabel="C6 prediction cut", ylabel="VB", title="C1→C6 next-cut forecasts, seed 42")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prediction_curves.png", dpi=200)
    plt.close(fig)
    write_json(out / "score_audit.json", {
        "target_wear_first_read_stage": "score, after forecast_unscored.csv was hash-locked",
        "target_wear_used_for_fitting": False,
        "future_cut_signal_or_prediction_used_for_fitting": False,
        "existing_gru_method": "k5_source_only only; no DARE-GRAM data",
        "existing_gru_checkpoint_sha256": base.file_hash(gru_checkpoint),
        "existing_gru_prediction_sha256": base.file_hash(gru_prediction_file),
        "stage_definition": {name: [start, end] for name, start, end in STAGES},
        "signed_error_definition": "prediction minus true VB",
    })
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    args = arguments()
    if args.stage in ("infer", "all"):
        infer(args)
    if args.stage in ("score", "all"):
        score(args)


if __name__ == "__main__":
    main()
