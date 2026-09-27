"""Paired DANN/MMD ablations on the frozen 315-cut Z-score protocol.

The historical source-only and DARE-GRAM checkpoints are read only. New methods
start from the same seeded ResNet18 + Linear(512, 1) initialization and source
batch order. Target wear is opened only after a final checkpoint and every
prediction have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import run_single_source_pairs as base
import utils
from models.DANN import Discriminator
from models.DAREGRAM import Trainset as DAREGRAM


SEEDS = (42, 43, 44, 45, 46)
PAIRS = tuple((s, t) for s in base.TOOLS for t in base.TOOLS if s != t)
METHODS = ("dann", "mmd", "daregram_mmd", "daregram_dann", "daregram_mmd_dann")
BASELINES = ("source_only", "daregram")
EPOCHS, BATCH_SIZE, LR = 50, 63, 0.001
KERNEL_ALPHAS = tuple(2.0 ** k for k in range(-3, 2))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def config_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=Path("artifacts/dann_mmd_joint_20260926"))
    p.add_argument("--baseline-root", type=Path, default=Path("artifacts/five_seed_paired"))
    p.add_argument("--full-baseline-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    p.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:4]))
    p.add_argument("--pairs", nargs="+", choices=[f"{s}_to_{t}" for s, t in PAIRS])
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument("--lambda-gram", type=float, default=1.0,
                   help="Preserves the historical DARE-GRAM epoch tradeoff in joint runs")
    p.add_argument("--lambda-mmd", type=float, default=0.1)
    p.add_argument("--lambda-dann", type=float, default=0.1)
    p.add_argument("--ramp", choices=("exp", "constant"), default="exp")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="One paired batch, gradient and label-isolation check")
    a = p.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (a.lambda_gram, a.lambda_mmd, a.lambda_dann)):
        p.error("Loss weights must be finite and nonnegative")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    if a.out_root.resolve() in (a.baseline_root.resolve(), a.full_baseline_root.resolve()):
        p.error("Output must be separate from frozen baselines")
    return a


def feature_cache(a, tool, manifest):
    path = a.baseline_root / "feature_cache" / f"{tool}_stft.npy"
    if base.file_hash(path) != manifest["tools"][tool]["feature_sha256"]:
        raise ValueError(f"STFT cache changed: {path}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (315, 6, 128, 128) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"Invalid STFT cache: {path}")
    return x


def prepare(a, source, target, manifest):
    raw_s = feature_cache(a, source, manifest)
    mean = raw_s.mean(axis=(0, 2, 3)).astype(np.float32)
    std = (raw_s.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    x_s = ((raw_s - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    raw_t = feature_cache(a, target, manifest)
    x_t = ((raw_t - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    y_s = base.wear_labels(a.raw_root, source)
    return x_s, y_s, x_t, mean, std


def mmd_loss(f_s, f_t):
    """Original repository's five Gaussian kernels and biased index matrix.

    sigma^2_k = alpha_k * mean(||z_i-z_j||^2), detached per batch. The
    index matrix uses 1/B in same-domain blocks and -1/B across domains;
    the kernel sum is finally divided by B-1, exactly as utils.py does.
    """
    if f_s.shape != f_t.shape or f_s.ndim != 2 or f_s.shape[0] < 2:
        raise ValueError("MMD requires equal B x 512 source/target cut batches")
    z = torch.cat((f_s, f_t), 0)
    dist2 = torch.cdist(z, z).square()
    sigma_base = dist2.detach().mean().clamp_min(torch.finfo(z.dtype).eps)
    kernel = sum(torch.exp(-dist2 / (2 * alpha * sigma_base)) for alpha in KERNEL_ALPHAS)
    b = f_s.shape[0]
    same = kernel[:b, :b].sum() + kernel[b:, b:].sum()
    cross = kernel[:b, b:].sum() + kernel[b:, :b].sum()
    return (same - cross) / (b * (b - 1))


def ramp(epoch, mode):
    return 1.0 if mode == "constant" else 2 / (1 + math.exp(-10 * (epoch - 1) / (EPOCHS - 1))) - 1


def train(a, method, source, target, seed, x_s, y_s, x_t, folder):
    device = torch.device(a.device)
    model, init_hash = base.build_model(seed, device)
    use_gram = method.startswith("daregram")
    use_mmd = "mmd" in method
    use_dann = "dann" in method
    discriminator = Discriminator().to(device) if use_dann else None
    domain_adv = (utils.DomainAdversarialLoss(discriminator, max_iters=EPOCHS * (315 // BATCH_SIZE))
                  if use_dann else None)
    params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator is not None else [])
    opt = torch.optim.Adam(params, lr=LR)
    source_loader = DataLoader(TensorDataset(torch.from_numpy(x_s), torch.from_numpy(y_s), torch.arange(1, 316)),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                               generator=torch.Generator().manual_seed(seed))
    target_loader = DataLoader(TensorDataset(torch.from_numpy(x_t), torch.arange(1, 316)),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                               generator=torch.Generator().manual_seed(seed + 1))
    rows, order_hashes = [], []
    discriminator_before = (torch.cat([p.detach().flatten().cpu() for p in discriminator.parameters()])
                            if discriminator is not None else None)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        source_order, target_order = [], []
        totals = {k: 0.0 for k in ("wear", "gram", "mmd", "dann", "weighted_gram", "weighted_mmd",
                                   "weighted_dann", "total", "domain_accuracy")}
        factor = ramp(2 if a.smoke else epoch, a.ramp)
        for (xb, yb, sc), (xt, tc) in zip(source_loader, target_loader, strict=True):
            source_order.extend(sc.tolist())
            target_order.extend(tc.tolist())
            xb, yb, xt = xb.to(device), yb.to(device).unsqueeze(1), xt.to(device)
            opt.zero_grad()
            f_s, f_t = model.feature_extractor(xb), model.feature_extractor(xt)
            if f_s.shape != (BATCH_SIZE, 512) or f_t.shape != (BATCH_SIZE, 512):
                raise ValueError("Expected one 512-vector per cut")
            wear = F.mse_loss(model.regressor(f_s), yb)
            gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), f_s, f_t) if use_gram else wear.new_zeros(())
            mmd = mmd_loss(f_s, f_t) if use_mmd else wear.new_zeros(())
            dann, acc = domain_adv(f_s, f_t) if use_dann else (wear.new_zeros(()), wear.new_zeros(()))
            wg = a.lambda_gram * factor * gram
            wm = a.lambda_mmd * factor * mmd
            wd = a.lambda_dann * factor * dann
            loss = wear + wg + wm + wd
            if not all(torch.isfinite(v).item() for v in (wear, gram, mmd, dann, loss)):
                raise FloatingPointError(f"Nonfinite loss {source}->{target} {seed} {method} epoch {epoch}")
            loss.backward()
            opt.step()
            for key, val in (("wear", wear), ("gram", gram), ("mmd", mmd), ("dann", dann),
                             ("weighted_gram", wg), ("weighted_mmd", wm), ("weighted_dann", wd),
                             ("total", loss), ("domain_accuracy", acc)):
                totals[key] += float(val.detach().item())
            if a.smoke and len(source_order) >= 2 * BATCH_SIZE:
                break
        if a.smoke:
            if any(totals[key] <= 0 for key, enabled in (("gram", use_gram), ("mmd", use_mmd),
                                                         ("dann", use_dann)) if enabled):
                raise AssertionError("Active adaptation loss was zero")
            if discriminator is not None:
                after = torch.cat([p.detach().flatten().cpu() for p in discriminator.parameters()])
                if torch.equal(discriminator_before, after):
                    raise AssertionError("Domain discriminator did not update")
            return {"source": source, "target": target, "seed": seed, "method": method,
                    "losses": totals, "discriminator_updated": discriminator is not None,
                    "target_label_reads": 0, "feature_shape": list(f_s.shape),
                    "grl_coefficient_first_step": 0.0,
                    "grl_steps": domain_adv.grl.iter_num if use_dann else 0}
        if sorted(source_order) != base.ALL_CUTS or sorted(target_order) != base.ALL_CUTS:
            raise ValueError("Each epoch must use each source and target cut exactly once")
        order_hash = hashlib.sha256(np.asarray(source_order, dtype=np.int32).tobytes()).hexdigest()
        order_hashes.append(order_hash)
        row = {"epoch": epoch, "source_order_sha256": order_hash, "source_cuts": 315,
               "unlabeled_target_cuts": 315, "target_label_reads": 0,
               "ramp_factor": factor, "lambda_gram": a.lambda_gram if use_gram else 0.0,
               "lambda_mmd": a.lambda_mmd if use_mmd else 0.0,
               "lambda_dann": a.lambda_dann if use_dann else 0.0,
               "grl_steps": domain_adv.grl.iter_num if use_dann else 0}
        row.update({key: val / len(source_loader) for key, val in totals.items()})
        rows.append(row)
        logging.info("%s %s->%s seed=%d epoch=%d wear=%.6g gram=%.6g mmd=%.6g dann=%.6g total=%.6g",
                     method, source, target, seed, epoch, *(row[k] for k in ("wear", "gram", "mmd", "dann", "total")))
    if discriminator is not None:
        after = torch.cat([p.detach().flatten().cpu() for p in discriminator.parameters()])
        if torch.equal(discriminator_before, after):
            raise AssertionError("Domain discriminator did not update")
    ckpt = folder / "final.pth"
    torch.save({"model": model.state_dict(), "discriminator": discriminator.state_dict() if use_dann else None,
                "epoch": EPOCHS}, ckpt)
    pd.DataFrame(rows).to_csv(folder / "epoch_losses.csv", index=False)
    return {"initial_model_sha256": init_hash, "source_order_sha256_by_epoch": order_hashes,
            "checkpoint_sha256": base.file_hash(ckpt), "grl_steps": domain_adv.grl.iter_num if use_dann else 0}


def predict(a, seed, x_t, folder):
    model, _ = base.build_model(seed, torch.device(a.device))
    ckpt = torch.load(folder / "final.pth", map_location=a.device, weights_only=True)
    if ckpt["epoch"] != EPOCHS:
        raise ValueError("Prediction requires the final epoch")
    model.load_state_dict(ckpt["model"])
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, 315, BATCH_SIZE):
            xb = torch.from_numpy(x_t[offset:offset + BATCH_SIZE]).to(a.device)
            values.extend(model.regressor(model.feature_extractor(xb)).cpu().numpy().ravel().tolist())
    result = np.asarray(values, np.float64)
    if result.shape != (315,) or not np.isfinite(result).all():
        raise ValueError("Invalid 315-cut prediction")
    np.save(folder / "predictions_before_target_labels.npy", result)
    return result


def run_one(a, manifest, source, target, seed, method):
    folder = a.out_root / "new_runs" / f"{source}_to_{target}" / f"seed_{seed}" / method
    if (folder / "complete.json").exists():
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if cfg["checkpoint_sha256"] != base.file_hash(folder / "final.pth") or cfg["method"] != method:
            raise ValueError(f"Existing completed run changed: {folder}")
        return
    if folder.exists():
        raise FileExistsError(f"Incomplete run requires inspection: {folder}")
    folder.mkdir(parents=True)
    x_s, y_s, x_t, mean, std = prepare(a, source, target, manifest)
    trace = train(a, method, source, target, seed, x_s, y_s, x_t, folder)
    baseline_cfg = json.loads((a.baseline_root / f"{source}_to_{target}" / f"seed_{seed}" /
                               "source_only" / "config.json").read_text(encoding="utf-8"))
    if trace["initial_model_sha256"] != baseline_cfg["initial_model_sha256"] or trace["source_order_sha256_by_epoch"] != baseline_cfg["source_order_sha256_by_epoch"]:
        raise AssertionError("Initialization or source batch order differs from frozen baseline")
    pred = predict(a, seed, x_t, folder)
    # The sole target-wear read occurs after model selection (fixed final epoch),
    # checkpoint serialization and prediction serialization.
    y_t = base.wear_labels(a.raw_root, target).astype(np.float64)
    frame = pd.DataFrame({"cut_index": base.ALL_CUTS, "true_vb": y_t, "pred_vb": pred})
    frame.to_csv(folder / "predictions.csv", index=False, float_format="%.17g")
    metric = base.metrics(y_t, pred)
    write_json(folder / "metrics.json", metric)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(base.ALL_CUTS, y_t, label="True VB")
    ax.plot(base.ALL_CUTS, pred, label="Predicted VB")
    ax.set(xlabel=f"{target.upper()} cut", ylabel="VB", title=f"{source.upper()}->{target.upper()} seed {seed} {method}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "prediction.png", dpi=200)
    plt.close(fig)
    use = {"gram": method.startswith("daregram"), "mmd": "mmd" in method, "dann": "dann" in method}
    cfg = {"source": source, "target": target, "seed": seed, "method": method,
           "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "optimizer": "Adam, fixed LR",
           "raw_root": str(a.raw_root.resolve()),
           "source_wear_sha256": base.file_hash(a.raw_root / f"{source}_wear.csv"),
           "target_wear_sha256_for_evaluation_only": base.file_hash(a.raw_root / f"{target}_wear.csv"),
           "stft_code_sha256": manifest["stft_code_sha256"],
           "input": "one center 4096-point window per cut; six-channel full-band STFT log1p; 128x128; source Z-score",
           "source_normalization_mean": mean.tolist(), "source_normalization_std": std.tolist(),
           "backbone": "ResNet18", "regressor": "Linear(512,1)", "adaptation_feature_layer": "ResNet18 pooled 512-vector before regressor",
           "adaptation_feature_batch_shape": [63, 512], "mmd_kernel": "Gaussian sum, source utils.py biased index matrix",
           "mmd_alpha": KERNEL_ALPHAS, "mmd_sigma_squared": "alpha * detached batch mean all-pair squared L2 distance",
           "dann_discriminator": "original 512-256-1 sigmoid", "domain_labels": {"source": 1, "target": 0},
           "grl": "original warm start alpha=1, lo=0, hi=1, 250 steps; reversal internal only",
           "ramp": a.ramp, "lambda_gram": a.lambda_gram if use["gram"] else 0.0,
           "lambda_mmd": a.lambda_mmd if use["mmd"] else 0.0,
           "lambda_dann": a.lambda_dann if use["dann"] else 0.0,
           "checkpoint_selection": "final epoch, fixed before target labels",
           "evaluation_cuts": base.ALL_CUTS, "source_labeled_cuts": base.ALL_CUTS,
           "target_unlabeled_cuts": base.ALL_CUTS, "target_label_reads_during_training": 0,
           "initial_model_sha256": trace["initial_model_sha256"],
           "source_order_sha256_by_epoch": trace["source_order_sha256_by_epoch"],
           "checkpoint_sha256": trace["checkpoint_sha256"], "grl_steps": trace["grl_steps"],
           "source_stft_sha256": manifest["tools"][source]["feature_sha256"],
           "target_stft_sha256": manifest["tools"][target]["feature_sha256"]}
    write_json(folder / "config.json", cfg)
    write_json(folder / "complete.json", {"checkpoint_sha256": trace["checkpoint_sha256"], "metrics": metric})


def summarize(a):
    baseline = pd.read_csv(a.full_baseline_root / "per_seed_metrics_full_1_315.csv")
    baseline = baseline[baseline.method.isin(BASELINES)]
    provenance = pd.read_csv(a.full_baseline_root / "checkpoint_audit_full_1_315.csv")
    baseline = baseline.merge(provenance[["source", "target", "seed", "method", "checkpoint",
                                          "checkpoint_sha256"]],
                              on=["source", "target", "seed", "method"], validate="one_to_one")
    rows = []
    for source, target in PAIRS:
        for seed in SEEDS:
            for method in METHODS:
                folder = a.out_root / "new_runs" / f"{source}_to_{target}" / f"seed_{seed}" / method
                if not (folder / "complete.json").exists():
                    continue
                m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                rows.append({"source": source, "target": target, "seed": seed, "method": method,
                             "R2": m["R2"], "MAE": m["MAE"], "RMSE": m["RMSE"],
                             "checkpoint": str((folder / "final.pth").resolve()),
                             "checkpoint_sha256": base.file_hash(folder / "final.pth"),
                             "predictions": str((folder / "predictions.csv").resolve())})
    new = pd.DataFrame(rows)
    cols = ["source", "target", "seed", "method", "R2", "MAE", "RMSE", "checkpoint", "checkpoint_sha256"]
    table = pd.concat((baseline[cols], new[cols] if len(new) else pd.DataFrame(columns=cols)), ignore_index=True)
    table.to_csv(a.out_root / "per_seed_metrics.csv", index=False)
    grouped = table.groupby(["source", "target", "method"], sort=True)
    summary = grouped.agg(n_seeds=("seed", "nunique"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                          MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                          RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std")).reset_index()
    summary.to_csv(a.out_root / "five_seed_summary.csv", index=False)
    pivot = table.set_index(["source", "target", "seed", "method"])
    deltas = []
    for row in table.itertuples():
        for reference in BASELINES:
            ref = pivot.loc[(row.source, row.target, row.seed, reference)]
            deltas.append({"source": row.source, "target": row.target, "seed": row.seed,
                           "method": row.method, "reference": reference,
                           **{f"delta_{m}": getattr(row, m) - ref[m] for m in ("R2", "MAE", "RMSE")}})
    pd.DataFrame(deltas).to_csv(a.out_root / "paired_deltas.csv", index=False)


def main():
    a = config_args()
    manifest_path = a.baseline_root / "feature_cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["stft_code_sha256"] != base.file_hash(Path(base.sampling.__file__)):
        raise ValueError("STFT code changed since frozen baseline")
    if a.smoke:
        source, target = PAIRS[0]
        x_s, y_s, x_t, _, _ = prepare(a, source, target, manifest)
        results = [train(a, method, source, target, SEEDS[0], x_s, y_s, x_t, None) for method in METHODS]
        a.out_root.mkdir(parents=True, exist_ok=True)
        write_json(a.out_root / "smoke.json", {"results": results, "target_wear_opened": False})
        print("Smoke passed:", a.out_root / "smoke.json")
        return
    a.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(a.out_root / "run.log", encoding="utf-8"), logging.StreamHandler()])
    pairs = [pair for pair in PAIRS if a.pairs is None or f"{pair[0]}_to_{pair[1]}" in a.pairs]
    for method in a.methods:
        for source, target in pairs:
            for seed in a.seeds:
                run_one(a, manifest, source, target, seed, method)
                summarize(a)
    summarize(a)


if __name__ == "__main__":
    main()
