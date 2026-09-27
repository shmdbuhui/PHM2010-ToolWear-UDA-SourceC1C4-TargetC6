"""Source-only diagnostic of final-checkpoint train-mode sensitivity to batch cut composition."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import run_single_source_pairs as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    parser.add_argument("--train-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--mode-root", type=Path, default=Path("artifacts/representation_probe_audit_20260926"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/representation_probe_audit_20260926/bn_batch_order_c6.csv"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists")
    provenance = pd.read_csv(args.prior_root / "provenance.csv")
    mode = pd.read_csv(args.mode_root / "mode_metrics.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for source in ("c1", "c4"):
        raw = np.load(args.train_root / "feature_cache" / f"{source}_stft.npy", mmap_mode="r")
        mean = raw.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        x = ((raw - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
        y = base.wear_labels(args.raw_root, source).astype(np.float64)
        for seed in range(42, 47):
            generator = torch.Generator().manual_seed(seed)
            order_loader = DataLoader(TensorDataset(torch.arange(1, 316)), batch_size=63, shuffle=True,
                                      drop_last=False, generator=generator)
            for epoch in range(1, 51):
                batches = [cut.ravel().tolist() for (cut,) in order_loader]
            order = [cut for batch in batches for cut in batch]
            digest = hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest()
            for method in ("source_only", "daregram"):
                rec = provenance[(provenance.source == source) & (provenance.target == "c6") &
                                 (provenance.seed == seed) & (provenance.method == method)].iloc[0]
                cfg = json.loads(Path(rec.training_config).read_text(encoding="utf-8"))
                if not np.array_equal(mean, np.asarray(cfg["source_normalization_mean"], dtype=np.float32)):
                    raise ValueError("Normalization mismatch")
                log = (Path(rec.checkpoint).parent / "run.log").read_text(encoding="utf-8")
                hit = re.search(rf"^{method} epoch=50/50 source_MSE=([\d.eE+-]+) gram=[\d.eE+-]+ source_order_sha256=([a-f0-9]{{64}})", log, re.MULTILINE)
                if not hit or hit[2] != digest:
                    raise ValueError("Historical epoch-50 batch order mismatch")
                model, _ = base.build_model(seed, device)
                model.load_state_dict(torch.load(rec.checkpoint, map_location="cpu", weights_only=True)["model"])
                model.train()
                pred = np.full(315, np.nan)
                with torch.no_grad():
                    for batch in batches:
                        ix = np.asarray(batch) - 1
                        xb = torch.from_numpy(x[ix]).to(device)
                        pred[ix] = model.regressor(model.feature_extractor(xb)).cpu().numpy().ravel()
                old = mode[(mode.source == source) & (mode.target == "c6") &
                           (mode.seed == seed) & (mode.method == method)].iloc[0]
                rows.append({"source": source, "target": "c6", "seed": seed, "method": method,
                             "last_epoch_source_order_sha256": digest,
                             "last_epoch_batch_order_match": True,
                             "final_checkpoint_train_mode_shuffled_MAE": float(np.mean(abs(pred - y))),
                             "final_checkpoint_train_mode_shuffled_MSE": float(np.mean((pred - y) ** 2)),
                             "final_checkpoint_train_mode_cut_order_MAE": old.MAE_train,
                             "final_checkpoint_train_mode_cut_order_MSE": old.MSE_train,
                             "final_checkpoint_eval_mode_MAE": old.MAE_eval,
                             "historical_epoch50_training_batchmean_MSE": float(hit[1]),
                             "checkpoint_sha256": rec.checkpoint_sha256})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} historical-batch-order controls")


if __name__ == "__main__":
    main()
