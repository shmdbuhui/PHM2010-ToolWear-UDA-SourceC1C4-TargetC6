"""Verify the completed normalization comparison and write a compact report."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_five_seed_pairs as five
import run_single_source_pairs as base
from run_norm_comparison import (DEFAULT_OUTPUT, DEFAULT_CACHE, NORMS,
                                 evaluation_cuts, read_cache, source_parameters)


COMMON_FIELDS = (
    "source", "target", "seed", "epochs", "batch_size", "lr", "device",
    "raw_root", "split_record", "cache_root", "stft_code_sha256",
    "source_feature_sha256", "target_feature_sha256", "source_wear_sha256",
    "source_count", "source_cuts", "target_unlabeled_cuts", "evaluation_cuts",
    "evaluation_cut_definition", "input", "backbone", "regressor", "optimizer",
    "scheduler", "source_loss", "daregram_loss", "checkpoint_selection",
    "protocol", "target_labels_training_reads", "initial_model_sha256",
    "source_order_sha256_by_epoch", "source_order_sha256_all_epochs",
    "actual_unlabeled_target_cuts", "target_training_information", "method",
)


def load(folder):
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def verify(root):
    records = []
    manifest = json.loads((DEFAULT_CACHE / "manifest.json").read_text(encoding="utf-8"))
    expected_parameters = {}
    for source in base.TOOLS:
        source_cache = read_cache(DEFAULT_CACHE, manifest, source)
        for norm in NORMS:
            expected_parameters[(source, norm)] = source_parameters(source_cache, norm)
    for source, target in five.PAIRS:
        expected_cuts, _ = evaluation_cuts(
            target, Path(r"E:\QLP\test-9.3\outputs\predictions_comparison.csv"))
        for seed in five.SEEDS:
            for method in five.METHODS:
                configs = {}
                bn_hashes = {}
                for norm in NORMS:
                    folder = root / norm / f"{source}_to_{target}" / f"seed_{seed}" / method
                    if not (folder.parent / "complete.json").is_file():
                        raise FileNotFoundError(f"Incomplete experiment: {folder}")
                    config = load(folder)
                    if config["norm_method"] != norm or config["normalization_source"] != source or \
                            config["normalization_axis"] != [0, 2, 3]:
                        raise ValueError(f"Normalization fitting audit failed: {folder}")
                    for key, expected in expected_parameters[(source, norm)].items():
                        saved = np.asarray(config["source_normalization_parameters"][key], dtype=np.float32)
                        if not np.array_equal(saved, expected):
                            raise ValueError(f"Source-fitted {key} differs from the STFT cache: {folder}")
                    if config["evaluation_cuts"] != expected_cuts or \
                            config["target_unlabeled_cuts"] != base.ALL_CUTS or \
                            config["source_cuts"] != base.ALL_CUTS:
                        raise ValueError(f"Cut protocol mismatch: {folder}")
                    if config["actual_unlabeled_target_cuts"] != (
                        [] if method == "source_only" else base.ALL_CUTS):
                        raise ValueError(f"Target training usage mismatch: {folder}")
                    if config["target_labels_training_reads"] != 0:
                        raise ValueError(f"Training read target labels: {folder}")
                    metrics = five.metrics_from_csv(folder / "predictions.csv", expected_cuts)
                    recorded = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    if not all(np.isclose(metrics[key], recorded[key], atol=1e-12, rtol=1e-12)
                               for key in base.METRICS):
                        raise ValueError(f"Metrics mismatch: {folder}")
                    checkpoint = torch.load(folder / "final.pth", map_location="cpu", weights_only=True)
                    if checkpoint["epoch"] != five.EPOCHS:
                        raise ValueError(f"Wrong checkpoint epoch: {folder}")
                    if base.file_hash(folder / "final.pth") != config["checkpoint_sha256"]:
                        raise ValueError(f"Checkpoint changed after training: {folder}")
                    digest = hashlib.sha256()
                    for name, tensor in sorted(checkpoint["model"].items()):
                        if "running_mean" in name or "running_var" in name:
                            digest.update(name.encode())
                            digest.update(tensor.numpy().tobytes())
                    bn_hashes[norm] = digest.hexdigest()
                    configs[norm] = config
                differences = [field for field in COMMON_FIELDS
                               if configs["zscore"].get(field) != configs["minmax"].get(field)]
                if differences:
                    raise ValueError(f"Other configuration differences in {source}->{target} {seed} {method}: {differences}")
                if configs["zscore"]["checkpoint_sha256"] == configs["minmax"]["checkpoint_sha256"]:
                    raise ValueError(f"Identical final weights across normalizations: {source}->{target} {seed} {method}")
                if bn_hashes["zscore"] == bn_hashes["minmax"]:
                    raise ValueError(f"Identical BatchNorm running buffers: {source}->{target} {seed} {method}")
                records.append({"source": source, "target": target, "seed": seed, "method": method,
                                "same_shared_config": True, "same_initial_model": True,
                                "same_source_batch_order": True,
                                "different_final_checkpoint": True,
                                "different_batchnorm_running_buffers": True,
                                "zscore_batchnorm_sha256": bn_hashes["zscore"],
                                "minmax_batchnorm_sha256": bn_hashes["minmax"],
                                "evaluation_count": len(expected_cuts),
                                "zscore_target_outside_0_1_fraction": configs["zscore"]["target_outside_0_1_fraction"],
                                "minmax_target_outside_0_1_fraction": configs["minmax"]["target_outside_0_1_fraction"]})
    pd.DataFrame(records).to_csv(root / "protocol_audit.csv", index=False)
    return records


def report(root, records):
    summary = pd.read_csv(root / "summary.csv")
    delta = pd.read_csv(root / "deltas.csv")
    if len(summary) != 24 or len(delta) != 12:
        raise ValueError("Expected 24 normalization summaries and 12 direction-method deltas")
    lines = ["# PHM2010 input normalization comparison", "",
             "All 120 trainings completed: 2 normalizations × 6 directions × 5 seeds × 2 model methods.",
             "Values are five-seed mean ± sample SD (ddof=1). Δ is Min–Max minus Z-score, paired by seed.",
             "DARE-GRAM used unlabeled target cuts 1–315. All metrics and prediction plots use cuts 1–315, including target C6.",
             "Target labels were read after both model checkpoints and predictions were fixed.", "",
             "The C6 full-lifecycle request arrived after the Z-score training run. Its target-C6 final checkpoints",
             "were kept byte-for-byte; the existing prediction and metric functions were rerun on cuts 1–315.",
             "Min–Max used cuts 1–315 during its original evaluation. No C6 model was retrained for this cut change.", "",
             "| Direction | Method | Metric | Z-score | Min–Max | Δ Min–Max − Z-score |",
             "|---|---|---|---:|---:|---:|"]
    for source, target in five.PAIRS:
        for method in five.METHODS:
            z = summary.loc[(summary.norm_method == "zscore") & (summary.source == source) &
                            (summary.target == target) & (summary.method == method)].iloc[0]
            m = summary.loc[(summary.norm_method == "minmax") & (summary.source == source) &
                            (summary.target == target) & (summary.method == method)].iloc[0]
            d = delta.loc[(delta.source == source) & (delta.target == target) & (delta.method == method)].iloc[0]
            for metric in ("R2", "MAE", "RMSE"):
                lines.append(f"| {source.upper()}→{target.upper()} | {method} | {metric} | "
                             f"{z[f'{metric}_mean']:.4f} ± {z[f'{metric}_std']:.4f} | "
                             f"{m[f'{metric}_mean']:.4f} ± {m[f'{metric}_std']:.4f} | "
                             f"{d[f'minmax_minus_zscore_{metric}_mean']:+.4f} ± "
                             f"{d[f'minmax_minus_zscore_{metric}_std']:.4f} |")
    lines += ["", "## Execution", "",
              "Run in `upstream-reproduction`:", "",
              "```powershell",
              "python run_norm_comparison.py --preflight",
              "python run_norm_comparison.py --norm-method zscore",
              "python reevaluate_c6_full_lifecycle.py --norm-method zscore",
              "python run_norm_comparison.py --norm-method minmax",
              "python audit_norm_comparison.py",
              "```", "",
              "The preflight JSON records the C1→C4 seed 42 conv1 input check.",
              "`seed_metrics.csv`, `summary.csv`, and `deltas.csv` contain all numerical results.",
              "`protocol_audit.csv` records the 60 configuration and checkpoint comparisons.",
              "Each experiment is under `<norm_method>/<source>_to_<target>/seed_<seed>/<method>/`.",
              "Its config records source fitted parameters, source and target model input ranges,",
              "and the fraction of target values outside [0,1]. The Min–Max target input is not clipped.",
              "The compared configurations, initial weights, source batch orders, training cuts,",
              "and evaluation cuts match; the final checkpoint hashes differ across normalizations.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    root = DEFAULT_OUTPUT
    records = verify(root)
    report(root, records)
    print(f"Verified {len(records)} matched direction-seed-method experiments; wrote {root / 'README.md'}")


if __name__ == "__main__":
    main()
