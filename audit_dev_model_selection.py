"""Recheck held-out-source DEV artifacts without retraining or relabeling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_dev_model_selection as dev
import run_single_source_pairs as base


def audit_one(root: Path) -> dict:
    split = pd.read_csv(root / "source_split.csv")
    if len(split) != 315 or set(split.cut) != set(dev.CUTS):
        raise AssertionError(f"Invalid split: {root}")
    train = split.loc[split.split == "train", "cut"].to_numpy(np.int64)
    val = split.loc[split.split == "val", "cut"].to_numpy(np.int64)
    if len(train) != 252 or len(val) != 63 or set(train) & set(val):
        raise AssertionError(f"Source training/validation overlap: {root}")
    stage_counts = split.groupby(["wear_stage", "split"]).size().to_dict()
    if stage_counts != {(stage, part): n for stage in ("early", "middle", "late")
                        for part, n in (("train", 84), ("val", 21))}:
        raise AssertionError(f"Stage distribution differs: {root}")
    selected = pd.read_csv(root / "selected_models.csv")
    frozen = json.loads((root / "selection_frozen.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(root / "dev_scores.csv")
    if set(scores.method) != set(dev.METHODS) or len(scores) != 6:
        raise AssertionError(f"Missing DEV candidate: {root}")
    variant = selected.iloc[0].selection_variant
    if variant not in ("DEV", "DEV-guard"):
        raise AssertionError(f"Unrecognized selection variant: {root}")
    score_col = "dev_score" if variant == "DEV" else "dev_guard_score"
    status_col = "status" if variant == "DEV" else "dev_guard_status"
    if (scores.status == "ok").all() != (variant == "DEV"):
        raise AssertionError(f"Guard use did not match raw DEV validity: {root}")
    eligible = scores.loc[scores[status_col] == "ok"].sort_values([score_col, "method"])
    if eligible.empty or selected.iloc[0].selected_method != eligible.iloc[0].method:
        raise AssertionError(f"Selection does not match frozen DEV ordering: {root}")
    if frozen["selected_method"] != selected.iloc[0].selected_method or frozen["selection_uses_target_vb"]:
        raise AssertionError(f"Invalid selection freeze: {root}")
    for method in dev.METHODS:
        folder = root / method
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if (cfg["source_train_cuts"] != train.tolist() or cfg["source_val_cuts"] != val.tolist() or
                cfg["normalization_fit_cuts"] != train.tolist() or cfg["target_label_reads_before_selection"] != 0):
            raise AssertionError(f"Cut or target label leakage: {folder}")
        if cfg["checkpoint_sha256"] != base.file_hash(folder / "final.pth"):
            raise AssertionError(f"Changed checkpoint: {folder}")
        order_loader = DataLoader(torch.from_numpy(train), batch_size=dev.joint.BATCH_SIZE,
                                  shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]))
        expected = []
        for _ in range(dev.joint.EPOCHS):
            cuts = np.concatenate([batch.numpy() for batch in order_loader])
            if sorted(cuts.tolist()) != train.tolist():
                raise AssertionError(f"Training order includes a validation cut: {folder}")
            expected.append(hashlib.sha256(cuts.astype(np.int32).tobytes()).hexdigest())
        if expected != cfg["source_order_sha256_by_epoch"]:
            raise AssertionError(f"Training order checksum mismatch: {folder}")
        arr = np.load(folder / "dev_arrays.npz", allow_pickle=False)
        if (str(arr["source"]) != cfg["source"] or str(arr["target"]) != cfg["target"] or
                str(arr["method"]) != method or int(arr["seed"]) != cfg["seed"] or
                arr["source_train_cut"].tolist() != train.tolist() or
                arr["source_val_cut"].tolist() != val.tolist() or
                arr["target_cut"].tolist() != dev.CUTS.tolist() or
                arr["F_s_train"].shape != (252, 512) or arr["F_s_val"].shape != (63, 512) or
                arr["F_t"].shape != (315, 512)):
            raise AssertionError(f"Feature export identity/shape mismatch: {folder}")
        if (scores.set_index("method").loc[method, "domain_train_source_n"] != 201 or
                scores.set_index("method").loc[method, "domain_train_target_n"] != 252 or
                scores.set_index("method").loc[method, "domain_holdout_source_n"] != 51 or
                scores.set_index("method").loc[method, "domain_holdout_target_n"] != 63):
            raise AssertionError(f"Domain classifier counts suggest leaked validation cut: {folder}")
    preds = pd.read_csv(root / "candidate_predictions_before_target_labels.csv")
    final = pd.read_csv(root / "predictions_per_cut.csv")
    if (len(preds) != 6 * 315 or preds.groupby(["method", "cut"]).size().ne(1).any() or
            final.cut.tolist() != dev.CUTS.tolist() or final.method.nunique() != 1 or
            final.method.iloc[0] != selected.iloc[0].selected_method):
        raise AssertionError(f"Missing or duplicated final target cut: {root}")
    return {"run": str(root), "checks_passed": True, "source_train": 252,
            "source_val": 63, "candidate_count": 6, "final_target_cuts": 315,
            "selected_method": selected.iloc[0].selected_method}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/dev_model_selection_20260926"))
    a = p.parse_args()
    runs = sorted(path for path in (a.root / "runs").glob("*_to_*/seed_*")
                  if (path / "predictions_per_cut.csv").exists())
    if not runs:
        raise SystemExit("No completed runs")
    results = [audit_one(root) for root in runs]
    dev.write_json(a.root / "output_audit.json", {"completed_runs": len(results), "results": results})
    print(f"Audited {len(results)} complete runs")


if __name__ == "__main__":
    main()
