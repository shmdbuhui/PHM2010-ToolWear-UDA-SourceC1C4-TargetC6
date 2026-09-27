"""Check whether train/eval prediction gap is caused by BN batch statistics or buffer updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_single_source_pairs as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    parser.add_argument("--mode-root", type=Path, default=Path("artifacts/representation_probe_audit_20260926"))
    parser.add_argument("--train-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/representation_probe_audit_20260926/bn_mechanism_c6.csv"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists")
    provenance = pd.read_csv(args.prior_root / "provenance.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for source in ("c1", "c4"):
        raw = np.load(args.train_root / "feature_cache" / f"{source}_stft.npy", mmap_mode="r")
        mean = raw.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        x = ((raw - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
        for seed in range(42, 47):
            row = provenance[(provenance.source == source) & (provenance.target == "c6") &
                             (provenance.seed == seed) & (provenance.method == "daregram")].iloc[0]
            config = json.loads(Path(row.training_config).read_text(encoding="utf-8"))
            if not np.array_equal(mean, np.asarray(config["source_normalization_mean"], dtype=np.float32)) or not np.array_equal(std, np.asarray(config["source_normalization_std"], dtype=np.float32)):
                raise ValueError("Normalization mismatch")
            model, _ = base.build_model(seed, device)
            state = torch.load(row.checkpoint, map_location="cpu", weights_only=True)["model"]
            model.load_state_dict(state)
            model.train()
            bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
            before = [(m.running_mean.clone(), m.running_var.clone()) for m in bn]
            for m in bn:
                m.momentum = 0.0
            predictions = []
            with torch.no_grad():
                for start in range(0, 315, 63):
                    xb = torch.from_numpy(x[start:start + 63]).to(device)
                    predictions.extend(model.regressor(model.feature_extractor(xb)).cpu().numpy().ravel().tolist())
            original_train = pd.read_csv(args.mode_root / "mode_predictions" /
                                         f"{source}_to_c6_seed_{seed}_daregram_train.csv").pred_vb.to_numpy()
            rows.append({"source": source, "target": "c6", "seed": seed, "method": "daregram",
                         "batchnorm_layer_count": len(bn),
                         "bn_running_buffers_unchanged_with_momentum_zero": all(
                             torch.equal(a, m.running_mean) and torch.equal(b, m.running_var)
                             for (a, b), m in zip(before, bn)),
                         "max_abs_train_prediction_difference_from_original": float(np.max(np.abs(np.asarray(predictions) - original_train))),
                         "checkpoint_sha256": row.checkpoint_sha256})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} BN batch-statistic controls")


if __name__ == "__main__":
    main()
