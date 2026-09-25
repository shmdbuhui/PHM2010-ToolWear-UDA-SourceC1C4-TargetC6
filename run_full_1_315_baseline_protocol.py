"""Audit frozen Z-score runs and publish the six-direction full-cut baseline.

No training is performed. A fresh directory is required to preserve old evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import analyze_zscore_stage_errors as stage
import run_single_source_pairs as base


PAIRS = tuple((s, t) for s in stage.TOOLS for t in stage.TOOLS if s != t)
SEEDS = tuple(stage.SEEDS)
METHODS = stage.METHODS
CUTS = list(range(1, 316))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def measure(y: np.ndarray, pred: np.ndarray) -> dict:
    if y.shape != (315,) or pred.shape != (315,) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("All 315 labels and predictions must be finite")
    return dict(base.metrics(y, pred), signed_bias=float(np.mean(pred - y)))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (s, t, method), frame in rows.groupby(["source", "target", "method"], sort=True):
        if sorted(frame.seed.tolist()) != list(SEEDS):
            raise ValueError(f"Missing paired seed: {s} {t} {method}")
        row = {"source": s, "target": t, "method": method, "scope": "full_1_315", "n_seeds": 5}
        for metric in ("R2", "MAE", "RMSE", "MAPE_percent", "signed_bias"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        result.append(row)
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--full-c6-root", type=Path, default=Path("artifacts/norm_comparison_20260925/zscore"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    args = parser.parse_args()
    original, full_c6, raw, out = (p.resolve() for p in
                                   (args.original_root, args.full_c6_root, args.raw_root, args.out_root))
    if out.exists() or any(out == p or p in out.parents for p in (original, full_c6)):
        parser.error("Output must be a new independent directory")
    cache = original / "feature_cache"
    manifest_path = cache / "manifest.json"
    manifest = stage.json_read(manifest_path)
    if manifest["cuts"] != CUTS or manifest["feature_shape"] != [315, 6, 128, 128]:
        raise ValueError("STFT cache cut/shape mismatch")
    if manifest["stft_code_sha256"] != stage.sha(Path(base.sampling.__file__)):
        raise ValueError("STFT implementation changed")
    source_stats = {}
    for tool in stage.TOOLS:
        cache_file = cache / f"{tool}_stft.npy"
        if stage.sha(cache_file) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"STFT cache hash mismatch: {tool}")
        x = np.load(cache_file, mmap_mode="r", allow_pickle=False)
        if x.shape != (315, 6, 128, 128) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"STFT cache invalid: {tool}")
        source_stats[tool] = (x.mean(axis=(0, 2, 3)).astype(np.float32),
                              (x.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32))
    truth = {}
    for tool in stage.TOOLS:
        truth[tool], _, audit = stage.labels(raw, tool)
        if audit["cuts_1_315_with_label"] != 315:
            raise ValueError(f"Missing true wear labels for {tool}")

    records, predictions = [], {}
    for source, target in PAIRS:
        for seed in SEEDS:
            pair = original / f"{source}_to_{target}" / f"seed_{seed}"
            pair_config = stage.json_read(pair / "config.json")
            pair_audit = stage.json_read(pair / "audit.json")
            if (pair_config["source"], pair_config["target"], pair_config["seed"]) != (source, target, seed):
                raise ValueError(f"Pair config mismatch: {pair}")
            if (pair_config["source_cuts"] != CUTS or pair_config["target_unlabeled_cuts"] != CUTS or
                pair_audit["source_only_training_target_cuts"] != [] or
                pair_audit["target_unlabeled_cuts"] != CUTS or
                pair_audit["target_label_reads_during_training"] != 0):
                raise ValueError(f"Training data protocol mismatch: {pair}")
            if (pair_config["epochs"], pair_config["batch_size"], pair_config["lr"],
                pair_config["backbone"], pair_config["regressor"], pair_config["optimizer"],
                pair_config["source_loss"], pair_config["daregram_loss"],
                pair_config["checkpoint_selection"]) != (
                    50, 63, 0.001, "ResNet18", "Linear(512, 1)", "Adam", "MSE",
                    "original inverse Gram loss with exp epoch tradeoff", "final epoch"):
                raise ValueError(f"Model/training settings mismatch: {pair}")
            if (pair_config["source_feature_sha256"] != manifest["tools"][source]["feature_sha256"] or
                pair_config["target_feature_sha256"] != manifest["tools"][target]["feature_sha256"] or
                pair_config["source_wear_sha256"] != stage.sha(raw / f"{source}_wear.csv") or
                pair_config["stft_code_sha256"] != manifest["stft_code_sha256"]):
                raise ValueError(f"Source data/cache mismatch: {pair}")
            mean, std = source_stats[source]
            if (not np.array_equal(np.asarray(pair_config["source_normalization_mean"], dtype=np.float32), mean) or
                not np.array_equal(np.asarray(pair_config["source_normalization_std"], dtype=np.float32), std)):
                raise ValueError(f"Source-only Z-score statistics mismatch: {pair}")
            if (pair_config["evaluation_cuts"] !=
                    (list(range(95, 316)) if target == "c6" else CUTS)):
                raise ValueError(f"Historical evaluation cut mismatch: {pair}")
            for method in METHODS:
                folder = pair / method
                cfg = stage.json_read(folder / "config.json")
                ckpt = folder / "final.pth"
                digest = stage.sha(ckpt)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                if state["epoch"] != 50 or not state["model"]:
                    raise ValueError(f"Not a final model: {ckpt}")
                if (cfg["source"], cfg["target"], cfg["seed"], cfg["method"],
                    cfg["actual_unlabeled_target_cuts"], cfg["target_labels_training_reads"]) != (
                        source, target, seed, method, CUTS if method == "daregram" else [], 0):
                    raise ValueError(f"Method config/provenance mismatch: {folder}")
                if not np.array_equal(np.asarray(cfg["source_normalization_mean"], dtype=np.float32), mean):
                    raise ValueError(f"Method mean mismatch: {folder}")
                if not np.array_equal(np.asarray(cfg["source_normalization_std"], dtype=np.float32), std):
                    raise ValueError(f"Method std mismatch: {folder}")
                audit_hash = pair_audit["checkpoint_sha256"][method]
                if digest != audit_hash:
                    raise ValueError(f"Checkpoint/audit hash mismatch: {ckpt}")
                prediction_file = folder / "predictions.csv"
                prediction_cfg = cfg
                if target == "c6":
                    full_folder = full_c6 / f"{source}_to_{target}" / f"seed_{seed}" / method
                    prediction_file = full_folder / "predictions.csv"
                    prediction_cfg = stage.json_read(full_folder / "config.json")
                    if (stage.sha(full_folder / "final.pth") != digest or
                        prediction_cfg["evaluation_cuts"] != CUTS or
                        prediction_cfg["norm_method"] != "zscore" or
                        prediction_cfg["source_normalization_parameters"] !=
                        {"mean": cfg["source_normalization_mean"], "std": cfg["source_normalization_std"]}):
                        raise ValueError(f"C6 full prediction provenance mismatch: {full_folder}")
                frame = pd.read_csv(prediction_file)
                if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"] or frame.cut_index.tolist() != CUTS:
                    raise ValueError(f"Expected unique ordered 315 predictions: {prediction_file}")
                if not np.allclose(frame.true_vb.to_numpy(float), truth[target], atol=1e-6, rtol=0):
                    raise ValueError(f"Target true labels mismatch: {prediction_file}")
                pred = frame.pred_vb.to_numpy(float)
                if not np.isfinite(pred).all():
                    raise ValueError(f"Nonfinite prediction: {prediction_file}")
                predictions[source, target, seed, method] = pred
                records.append({"source": source, "target": target, "seed": seed, "method": method,
                                "scope": "full_1_315", "training_reused": True,
                                "source_only_training_target_cuts": 0,
                                "daregram_training_target_cuts": 315,
                                "target_label_reads_during_training": 0,
                                "historical_evaluation_scope": "old_95_315" if target == "c6" else "full_1_315",
                                "checkpoint": str(ckpt), "checkpoint_sha256": digest,
                                "training_config": str(folder / "config.json"),
                                "training_config_sha256": stage.sha(folder / "config.json"),
                                "original_prediction_file": str(prediction_file),
                                "original_prediction_sha256": stage.sha(prediction_file),
                                "source_stft_cache": str(cache / f"{source}_stft.npy"),
                                "source_stft_sha256": manifest["tools"][source]["feature_sha256"],
                                "target_stft_cache": str(cache / f"{target}_stft.npy"),
                                "target_stft_sha256": manifest["tools"][target]["feature_sha256"],
                                "source_wear_file": str(raw / f"{source}_wear.csv"),
                                "source_wear_sha256": stage.sha(raw / f"{source}_wear.csv"),
                                "target_wear_file": str(raw / f"{target}_wear.csv"),
                                "target_wear_sha256": stage.sha(raw / f"{target}_wear.csv"),
                                "source_zscore_mean": json.dumps(mean.tolist()),
                                "source_zscore_std": json.dumps(std.tolist()),
                                "prediction_count": 315})
    if len(records) != 60:
        raise ValueError("Expected 60 fully audited checkpoints")
    out.mkdir(parents=True)
    (out / "per_seed").mkdir()
    pd.DataFrame(records).to_csv(out / "checkpoint_audit_full_1_315.csv", index=False)
    metrics = []
    for source, target in PAIRS:
        for seed in SEEDS:
            for method in METHODS:
                pred = predictions[source, target, seed, method]
                path = out / "per_seed" / f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
                pd.DataFrame({"cut_index": CUTS, "true_vb": truth[target], "pred_vb": pred,
                              "signed_error": pred - truth[target],
                              "absolute_error": np.abs(pred - truth[target])}).to_csv(
                                  path, index=False, float_format="%.17g")
                saved = pd.read_csv(path)
                if saved.cut_index.tolist() != CUTS:
                    raise ValueError(f"Roundtrip cut mismatch: {path}")
                result = measure(saved.true_vb.to_numpy(float), saved.pred_vb.to_numpy(float))
                metrics.append({"source": source, "target": target, "seed": seed, "method": method,
                                "scope": "full_1_315", "n": 315, **result,
                                "per_cut_file": str(path.resolve()), "per_cut_sha256": stage.sha(path)})
    table = pd.DataFrame(metrics)
    table.to_csv(out / "per_seed_metrics_full_1_315.csv", index=False)
    summarize(table).to_csv(out / "five_seed_summary_full_1_315.csv", index=False)
    write_json(out / "protocol_lock_full_1_315.json", {
        "status": "audited_baseline_frozen_before_OOR_PGA", "direction_count": 6,
        "checkpoint_count": 60, "scope": "full_1_315", "seeds": list(SEEDS),
        "training_repeated": False, "STFT_code_sha256": manifest["stft_code_sha256"],
        "cache_manifest": str(manifest_path), "cache_manifest_sha256": stage.sha(manifest_path),
        "checkpoint_audit_sha256": stage.sha(out / "checkpoint_audit_full_1_315.csv"),
        "per_seed_metrics_sha256": stage.sha(out / "per_seed_metrics_full_1_315.csv"),
        "five_seed_summary_sha256": stage.sha(out / "five_seed_summary_full_1_315.csv"),
        "historical_C6_95_315": "Retained unchanged in original five_seed_paired directory as old evaluation scope",
        "command": "python run_full_1_315_baseline_protocol.py"})
    print(f"Audited 60 frozen checkpoints; saved six-direction full-cut baseline: {out}")


if __name__ == "__main__":
    main()
