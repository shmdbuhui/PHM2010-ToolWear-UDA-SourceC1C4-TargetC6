"""Verify completed DANN/MMD runs and paired full-cut baseline provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_single_source_pairs as base
from models.DANN import Discriminator
from run_dann_mmd_joint import BASELINES, EPOCHS, METHODS, PAIRS, SEEDS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/dann_mmd_joint_20260926"))
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--full-baseline-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    parser.add_argument("--expect-methods", nargs="+", choices=METHODS, default=list(METHODS[:4]))
    a = parser.parse_args()
    summary = pd.read_csv(a.root / "per_seed_metrics.csv")
    summary = summary[summary.method.isin((*BASELINES, *a.expect_methods))]
    deltas = pd.read_csv(a.root / "paired_deltas.csv")
    deltas = deltas[deltas.method.isin((*BASELINES, *a.expect_methods))]
    if len(summary) != (len(BASELINES) + len(a.expect_methods)) * len(PAIRS) * len(SEEDS):
        raise AssertionError("Wrong number of per-seed metric rows")
    if len(deltas) != 2 * len(summary):
        raise AssertionError("Wrong number of paired deltas")
    baseline_audit = pd.read_csv(a.full_baseline_root / "checkpoint_audit_full_1_315.csv")
    checked = []
    for source, target in PAIRS:
        for seed in SEEDS:
            original = a.baseline_root / f"{source}_to_{target}" / f"seed_{seed}" / "source_only" / "config.json"
            base_cfg = json.loads(original.read_text(encoding="utf-8"))
            for method in a.expect_methods:
                folder = a.root / "new_runs" / f"{source}_to_{target}" / f"seed_{seed}" / method
                config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                checkpoint = torch.load(folder / "final.pth", map_location="cpu", weights_only=True)
                losses = pd.read_csv(folder / "epoch_losses.csv")
                prediction = pd.read_csv(folder / "predictions.csv")
                saved_metric = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                calculated = base.metrics(prediction.true_vb.to_numpy(), prediction.pred_vb.to_numpy())
                if (checkpoint["epoch"] != EPOCHS or len(losses) != EPOCHS or
                    prediction.cut_index.tolist() != base.ALL_CUTS or
                    config["evaluation_cuts"] != base.ALL_CUTS or
                    config["target_label_reads_during_training"] != 0 or
                    (losses.target_label_reads != 0).any() or
                    (losses.source_cuts != 315).any() or (losses.unlabeled_target_cuts != 315).any() or
                    config["initial_model_sha256"] != base_cfg["initial_model_sha256"] or
                    config["source_order_sha256_by_epoch"] != base_cfg["source_order_sha256_by_epoch"] or
                    config["raw_root"] != base_cfg["raw_root"] or
                    config["source_wear_sha256"] != base.file_hash(Path(config["raw_root"]) / f"{source}_wear.csv") or
                    config["target_wear_sha256_for_evaluation_only"] != base.file_hash(Path(config["raw_root"]) / f"{target}_wear.csv") or
                    config["stft_code_sha256"] != base_cfg["stft_code_sha256"] or
                    config["checkpoint_sha256"] != base.file_hash(folder / "final.pth") or
                    not np.isfinite(losses.select_dtypes(include=[np.number]).to_numpy()).all()):
                    raise AssertionError(f"Protocol/checkpoint/loss mismatch: {folder}")
                for name in ("gram", "mmd", "dann"):
                    expected = (name == "gram" and method.startswith("daregram")) or name in method
                    if expected and not (losses[name].abs() > 0).any():
                        raise AssertionError(f"Missing {name} loss: {folder}")
                if "dann" in method:
                    if config["grl_steps"] != 250 or checkpoint["discriminator"] is None:
                        raise AssertionError(f"Domain discriminator absent: {folder}")
                    # build_model resets the seed and consumes exactly the model
                    # initialization draws used before Discriminator() in training.
                    base.build_model(seed, torch.device("cpu"))
                    initial_domain = Discriminator().state_dict()
                    if not any(not torch.equal(initial_domain[key], value.cpu())
                               for key, value in checkpoint["discriminator"].items()):
                        raise AssertionError(f"Domain discriminator did not change: {folder}")
                if any(not np.isclose(saved_metric[key], calculated[key], rtol=1e-12, atol=1e-12)
                       for key in ("R2", "MAE", "RMSE")):
                    raise AssertionError(f"Metrics mismatch: {folder}")
                row = summary.loc[(summary.source == source) & (summary.target == target) &
                                  (summary.seed == seed) & (summary.method == method)]
                if len(row) != 1 or any(not np.isclose(row.iloc[0][key], saved_metric[key]) for key in ("R2", "MAE", "RMSE")):
                    raise AssertionError(f"Summary mismatch: {folder}")
                checked.append(str(folder))
            for method in BASELINES:
                row = baseline_audit.loc[(baseline_audit.source == source) &
                                         (baseline_audit.target == target) &
                                         (baseline_audit.seed == seed) &
                                         (baseline_audit.method == method)]
                if len(row) != 1 or row.iloc[0].checkpoint_sha256 != base.file_hash(Path(row.iloc[0].checkpoint)):
                    raise AssertionError("Frozen baseline checkpoint changed")
    result = {"status": "passed", "checked_new_runs": len(checked), "checked_baseline_runs": 60,
              "metric_rows": len(summary), "paired_delta_rows": len(deltas),
              "expected_methods": a.expect_methods, "evaluation_cuts": [1, 315],
              "no_target_label_reads_during_training": True,
              "initialization_and_source_order_match_frozen_baseline": True,
              "all_dann_discriminators_changed_from_seeded_initialization": True,
              "checkpoints_and_metrics_match_files": True}
    write_json(a.root / "output_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
