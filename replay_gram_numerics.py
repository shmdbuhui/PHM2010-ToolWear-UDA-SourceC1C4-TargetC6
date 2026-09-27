"""Minimal label-safe replay of one DARE-GRAM training run for Gram diagnostics.

The target STFT cache has no labels. Diagnostics observe the first source/target
batch of each epoch; the training objective and update order are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import run_single_source_pairs as base
from models.DAREGRAM import Trainset as DAREGRAM


def spectrum(feature):
    # Nonzero singular values of [1,F]^T[1,F] equal eigenvalues of [1,F][1,F]^T.
    augmented = torch.cat([torch.ones((len(feature), 1), device=feature.device), feature.detach()], 1)
    eigen = torch.linalg.eigvalsh((augmented @ augmented.T).double()).cpu().numpy()
    eigen = np.maximum(eigen, 0)
    positive = eigen[eigen > max(eigen.max() * 1e-10, 1e-12)]
    probability = positive / positive.sum()
    return {"gram_effective_rank": float(np.exp(-np.sum(probability * np.log(probability)))),
            "gram_numerical_rank": len(positive),
            "gram_nonzero_condition": float(positive.max() / positive.min()),
            "gram_structural_zero_count_min": 513 - len(positive),
            "feature_l2_mean": float(feature.detach().norm(dim=1).mean().item()),
            "feature_nonfinite_count": int((~torch.isfinite(feature)).sum().item()),
            "gram_nonfinite_count": int((~torch.isfinite(augmented @ augmented.T)).sum().item())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=base.TOOLS, default="c1")
    p.add_argument("--target", choices=base.TOOLS, default="c4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--train-root", type=Path, default=Path("artifacts/five_seed_paired"))
    p.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    p.add_argument("--out", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926/replay_gram_numerics.csv"))
    args = p.parse_args()
    if args.source == args.target or args.epochs < 1 or args.epochs > 50 or args.out.exists():
        p.error("Distinct tools, 1..50 epochs, and new output path required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_source = np.load(args.train_root / "feature_cache" / f"{args.source}_stft.npy", mmap_mode="r")
    raw_target = np.load(args.train_root / "feature_cache" / f"{args.target}_stft.npy", mmap_mode="r")
    mean = raw_source.mean(axis=(0, 2, 3)).astype(np.float32)
    std = (raw_source.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    xs = ((raw_source - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    xt = ((raw_target - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    ys = base.wear_labels(args.raw_root, args.source)
    model, initial_hash = base.build_model(args.seed, device)
    src = DataLoader(TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys), torch.arange(1, 316)),
                     batch_size=63, shuffle=True, drop_last=False,
                     generator=torch.Generator().manual_seed(args.seed))
    tgt = DataLoader(TensorDataset(torch.from_numpy(xt), torch.arange(1, 316)),
                     batch_size=63, shuffle=True, drop_last=False,
                     generator=torch.Generator().manual_seed(args.seed + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    old = args.train_root / f"{args.source}_to_{args.target}" / f"seed_{args.seed}" / "daregram"
    pattern = __import__("re").compile(r"^daregram epoch=(\d+)/50 source_MSE=([\d.eE+-]+) gram=([\d.eE+-]+) source_order_sha256=([a-f0-9]{64}) target_unique=315$")
    historical = [pattern.match(line) for line in (old / "run.log").read_text(encoding="utf-8").splitlines()]
    historical = [m for m in historical if m]
    if len(historical) != 50:
        raise ValueError("Missing historical epoch log")
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        it = iter(tgt)
        order, mse_total, gram_total = [], 0.0, 0.0
        weight = 2 / (1 + math.exp(-10 * (epoch - 1) / 49)) - 1
        for step, (xb, yb, source_cuts) in enumerate(src):
            order.extend(source_cuts.tolist())
            tb, target_cuts = next(it)
            xb, yb, tb = xb.to(device), yb.to(device).unsqueeze(1), tb.to(device)
            optimizer.zero_grad()
            fs = model.feature_extractor(xb)
            ft = model.feature_extractor(tb)
            mse = F.mse_loss(model.regressor(fs), yb)
            gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), fs, ft)
            if step == 0:
                with torch.no_grad():
                    source_stats, target_stats = spectrum(fs), spectrum(ft)
                    rows.append({"source": args.source, "target": args.target, "seed": args.seed,
                                 "epoch": epoch, "batch": 1, "source_cuts": ",".join(map(str, source_cuts.tolist())),
                                 "target_cuts": ",".join(map(str, target_cuts.tolist())),
                                 "source_mse_first_batch": float(mse.item()),
                                 "alignment_loss_first_batch": float(gram.item()),
                                 "alignment_weight": weight,
                                 **{f"source_{k}": v for k, v in source_stats.items()},
                                 **{f"target_{k}": v for k, v in target_stats.items()},
                                 "initial_model_sha256": initial_hash})
            (mse + weight * gram).backward()
            optimizer.step()
            mse_total += mse.item()
            gram_total += gram.item()
        digest = hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest()
        old_row = historical[epoch - 1]
        matched = (digest == old_row[4] and
                   abs(mse_total / 5 - float(old_row[2])) < 1e-3 and
                   abs(gram_total / 5 - float(old_row[3])) < 1e-3)
        rows[-1]["historical_epoch_log_match"] = matched
        rows[-1]["epoch_source_mse"] = mse_total / 5
        rows[-1]["epoch_alignment_loss"] = gram_total / 5
        print(f"epoch {epoch}/{args.epochs}: historical_log_match={matched}", flush=True)
        if not matched:
            raise ValueError(f"Replay diverged from historical training at epoch {epoch}")
    if args.epochs == 50:
        checkpoint = torch.load(old / "final.pth", map_location="cpu", weights_only=True)["model"]
        rows[-1]["final_parameters_match_checkpoint"] = all(
            torch.equal(value.cpu(), checkpoint[key]) for key, value in model.state_dict().items())
        if not rows[-1]["final_parameters_match_checkpoint"]:
            raise ValueError("Replay final weights differ from historical checkpoint")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
