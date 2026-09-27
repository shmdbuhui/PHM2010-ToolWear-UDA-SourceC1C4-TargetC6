"""Read-only protocol and numerical audit for the custom comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_zscore_stage_errors as stage
import run_full_1_315_baseline_protocol as baseline
import run_inverse_gram_subspace as run


def main():
    root = run.DEFAULT_ROOT
    run.verify_frozen()
    lock = stage.json_read(root / "prediction_lock_before_target_labels.json")
    if len(lock["predictions"]) != 30:
        raise ValueError("Expected 30 locked predictions")
    batches, preflights = [], []
    for source, target in baseline.PAIRS:
        score = pd.read_csv(run.OOR / "oor_scores" /
                            f"{source}_to_{target}_oor_score_full_1_315.csv")
        factor = score.multiplicative_factor.to_numpy(float)
        if score.cut_index.tolist() != run.CUTS or not np.isfinite(factor).all():
            raise ValueError(f"OOR score invalid: {source}->{target}")
        for seed in baseline.SEEDS:
            folder = run.run_path(root, source, target, seed)
            key = f"{source}_to_{target}_seed_{seed}"
            cfg = stage.json_read(folder / "config.json")
            old = stage.json_read(run.HISTORICAL / f"{source}_to_{target}" /
                                  f"seed_{seed}" / "daregram" / "config.json")
            if (cfg["initial_model_sha256"] != old["initial_model_sha256"] or
                cfg["source_order_sha256_by_epoch"] != old["source_order_sha256_by_epoch"] or
                cfg["evaluation_cuts"] != run.CUTS or cfg["unlabeled_target_cuts"] != run.CUTS or
                cfg["target_label_reads_during_training"] != 0 or
                cfg["checkpoint_sha256"] != stage.sha(folder / "final.pth")):
                raise ValueError(f"Training protocol mismatch: {folder}")
            raw, corrected = (pd.read_csv(folder / name) for name in
                              ("raw_predictions.csv", "oor_pga_predictions.csv"))
            hashes = lock["predictions"][key]
            if (stage.sha(folder / "raw_predictions.csv") != hashes["raw_sha256"] or
                stage.sha(folder / "oor_pga_predictions.csv") != hashes["corrected_sha256"] or
                raw.cut_index.tolist() != run.CUTS or corrected.cut_index.tolist() != run.CUTS or
                not np.allclose(raw.pred_vb.to_numpy(float), corrected.raw_pred_vb.to_numpy(float),
                                rtol=1e-12, atol=1e-12) or
                not np.allclose(corrected.oor_pga_pred_vb.to_numpy(float),
                                raw.pred_vb.to_numpy(float) * factor, rtol=1e-12, atol=1e-12)):
                raise ValueError(f"Prediction/factor mismatch: {folder}")
            old_oor = pd.read_csv(run.OOR / "unlabeled_predictions" /
                                  f"{key}_full_1_315.csv")
            if not np.allclose(old_oor.daregram_pred_vb.to_numpy(float) * factor,
                               old_oor.daregram_oor_pga_pred_vb.to_numpy(float),
                               rtol=1e-12, atol=1e-12):
                raise ValueError(f"Factor does not reproduce frozen DARE-GRAM OOR result: {folder}")
            batch = pd.read_csv(folder / "batch_diagnostics.csv")
            numeric = batch.select_dtypes(include="number").to_numpy()
            if (len(batch) != 250 or not np.isfinite(numeric).all() or
                batch.k.min() < 1 or
                (batch.k > batch[["rank_source", "rank_target"]].min(axis=1)).any() or
                not (batch.epsilon_source > 0).all() or not (batch.epsilon_target > 0).all()):
                raise ValueError(f"Batch diagnostic failure: {folder}")
            batches.append(batch)
            preflight = stage.json_read(folder / "real_batch_preflight.json")
            if not all(np.isfinite(preflight[k]) and preflight[k] > 0 for k in
                       ("alignment_feature_grad_norm", "grad_norm", "conv1_update_norm")):
                raise ValueError(f"Real batch preflight failed: {folder}")
            preflights.append(preflight)
    all_batches = pd.concat(batches, ignore_index=True)
    result = {"method": run.METHOD, "scope": "full_1_315", "runs": len(batches),
              "batches": len(all_batches), "real_batch_preflights": len(preflights),
              "failure_files": len(list(root.glob("*_to_*/seed_*/failure.json"))),
              "all_batch_numbers_finite": True, "all_initializations_match_daregram": True,
              "all_source_orders_match_daregram": True,
              "all_oor_factors_reproduce_frozen_daregram_correction": True,
              "all_315_raw_and_corrected_predictions_locked": True,
              "k_values": sorted(int(k) for k in all_batches.k.unique()),
              "ranges": {name: {"min": float(all_batches[name].min()),
                                "max": float(all_batches[name].max())}
                         for name in ("source_mse", "alignment_loss", "weighted_alignment_loss",
                                      "gradient_norm", "sigma_source_max", "sigma_source_min",
                                      "sigma_source_kept_min", "sigma_target_max", "sigma_target_min",
                                      "sigma_target_kept_min", "epsilon_source", "epsilon_target")},
              "preflight_alignment_feature_grad_norm_range":
                  [min(p["alignment_feature_grad_norm"] for p in preflights),
                   max(p["alignment_feature_grad_norm"] for p in preflights)],
              "preflight_conv1_update_norm_range":
                  [min(p["conv1_update_norm"] for p in preflights),
                   max(p["conv1_update_norm"] for p in preflights)]}
    run.write_json(root / "numerical_and_protocol_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
