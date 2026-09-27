"""Read-only pre-score audit; never opens target wear labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_nonnegative_increment import DEFAULT_OUT, GROUPS, checkpoint_path, pairs, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    if not (out / "prediction_lock.json").exists():
        raise RuntimeError("All checkpoints and predictions must be locked first")
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    for path, digest in lock["files"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Lock hash mismatch: {path}")
    records = []
    for source, target, seed in pairs():
        parent_path = checkpoint_path(source, target, seed)
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)["model"]
        orders = []
        for group in GROUPS:
            folder = out / f"{source}_to_{target}" / f"seed_{seed}" / group
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            current = torch.load(folder / "final.pth", map_location="cpu", weights_only=True)
            prediction = pd.read_csv(folder / "target_predictions_unscored.csv")
            losses = pd.read_csv(folder / "epoch_losses.csv")
            if (cfg["source"], cfg["target"], cfg["seed"], cfg["group"], cfg["epochs"],
                    cfg["target_labels_training_reads"], cfg["parent_checkpoint_sha256"]) != (
                    source, target, seed, group, 5, 0, sha(parent_path)):
                raise ValueError(f"Configuration/provenance mismatch: {folder}")
            if current["parent_checkpoint_sha256"] != sha(parent_path) or current["epoch"] != 5:
                raise ValueError(f"Checkpoint parent mismatch: {folder}")
            bn_keys = [name for name in parent if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
            if not bn_keys or any(not torch.equal(parent[name], current["model"][name]) for name in bn_keys):
                raise ValueError(f"BatchNorm buffers changed: {folder}")
            if prediction.cut_index.tolist() != list(range(1, 316)) or len(losses) != 5:
                raise ValueError(f"Cut/epoch count mismatch: {folder}")
            p = prediction.p_vb.to_numpy(float)
            if not np.isfinite(p).all():
                raise ValueError(f"Nonfinite direct prediction: {folder}")
            has_increment = group in ("F2", "F3")
            if has_increment:
                d = prediction.d_vb.to_numpy(float)[1:]
                c = prediction.c_vb.to_numpy(float)
                if (not np.isnan(prediction.d_vb.iloc[0]) or not np.isfinite(d).all()
                        or not np.isfinite(c).all() or (d < 0).any()
                        or not np.allclose(c, np.r_[p[0], p[0] + np.cumsum(d)], atol=1e-4, rtol=0)):
                    raise ValueError(f"Increment/cumulative invariant failed: {folder}")
            elif not prediction[["c_vb", "d_vb"]].isna().all().all():
                raise ValueError(f"Non-applicable increment curve present: {folder}")
            orders.append(losses[["source_order_sha256", "target_order_sha256"]].copy())
            records.append(dict(source=source, target=target, seed=seed, group=group,
                                checkpoint_sha256=sha(folder / "final.pth"),
                                bn_buffer_count=len(bn_keys), bn_unchanged=True,
                                direct_finite=True, increment_nonnegative=has_increment,
                                primary_curve=cfg["primary_curve"]))
        if any(not orders[0].equals(other) for other in orders[1:]):
            raise ValueError(f"Unequal source/target shuffle across groups: {source}->{target} seed {seed}")
    if len(records) != 120:
        raise ValueError("Expected 120 complete runs")
    write_json(out / "pre_score_audit.json", {"passed": True, "n_runs": len(records),
                                              "target_wear_opened": False, "records": records})
    print("Pre-score audit passed: 120/120 checkpoints; BN, ordering and increment invariants verified")


if __name__ == "__main__":
    main()
