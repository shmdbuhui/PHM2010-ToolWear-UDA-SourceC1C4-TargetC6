"""Locked DARE-GRAM continuation with a source-supervised nonnegative wear increment.

Stages are deliberately separate: ``gate`` replays the original head, ``train``
never opens target wear, and ``score`` requires a prediction lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import run_single_source_pairs as base
from models.DAREGRAM import Trainset as DAREGRAM


TOOLS = ("c1", "c4", "c6")
SEEDS = (42, 43, 44, 45, 46)
CUTS = list(range(1, 316))
GROUPS = ("F0", "F1", "F2", "F3")
BASELINE = Path("artifacts/full_1_315_baseline_zscore_20260925")
TRAIN_ROOT = Path("artifacts/five_seed_paired")
DEFAULT_OUT = Path("artifacts/nonnegative_increment_20260926")
EPOCHS = 5
BATCH = 63
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
INCREMENT_LR = 1e-3
MONOTONE_WEIGHT = 0.1
DELTA_WEIGHT = 1.0
CUMULATIVE_WEIGHT = 0.1
DARE_WEIGHT = 1.0
GRAM = DAREGRAM.DARE_GRAM_LOSS


def sha(path: Path) -> str:
    return base.file_hash(path)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pairs():
    return [(source, target, seed) for source in TOOLS for target in TOOLS
            if source != target for seed in SEEDS]


def checkpoint_path(source, target, seed):
    return TRAIN_ROOT / f"{source}_to_{target}" / f"seed_{seed}" / "daregram" / "final.pth"


def baseline_path(source, target, seed):
    return BASELINE / "per_seed" / f"{source}_to_{target}_seed_{seed}_daregram_full_1_315.csv"


def inputs(source, target):
    cache = TRAIN_ROOT / "feature_cache"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["cuts"] != CUTS:
        raise ValueError("STFT cache has no verified cut_index 1..315 mapping")
    raw_s = np.load(cache / f"{source}_stft.npy", mmap_mode="r", allow_pickle=False)
    raw_t = np.load(cache / f"{target}_stft.npy", mmap_mode="r", allow_pickle=False)
    if raw_s.shape != (315, 6, 128, 128) or raw_t.shape != raw_s.shape:
        raise ValueError("Unexpected STFT cache shape")
    mean = raw_s.mean(axis=(0, 2, 3)).astype(np.float32)
    std = (raw_s.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    x_s = ((raw_s - mean[None, :, None, None]) /
           (std[None, :, None, None] + 1e-8)).astype(np.float32)
    x_t = ((raw_t - mean[None, :, None, None]) /
           (std[None, :, None, None] + 1e-8)).astype(np.float32)
    return x_s, x_t, mean, std


def chronological_positions(cut_indices):
    """Resolve true adjacent cuts by explicit cut_index, never by shuffled row adjacency."""
    mapping = {int(cut): position for position, cut in enumerate(cut_indices)}
    if len(mapping) != 315 or sorted(mapping) != CUTS:
        raise ValueError("Need exactly one sample for every cut_index 1..315")
    return torch.tensor([mapping[cut] for cut in CUTS], dtype=torch.long)


def load_original(source, target, seed, device):
    path = checkpoint_path(source, target, seed)
    cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    if (cfg["source"], cfg["target"], cfg["seed"], cfg["method"],
            cfg["regressor"], cfg["source_cuts"], cfg["target_unlabeled_cuts"]) != (
            source, target, seed, "daregram", "Linear(512, 1)", CUTS, CUTS):
        raise ValueError(f"Original protocol differs: {path}")
    parent_audit = json.loads((path.parent.parent / "audit.json").read_text(encoding="utf-8"))
    if sha(path) != parent_audit["checkpoint_sha256"]["daregram"]:
        raise ValueError(f"Checkpoint hash differs: {path}")
    model, _ = base.build_model(seed, device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["epoch"] != 50:
        raise ValueError("Expected original final epoch 50")
    model.load_state_dict(state["model"], strict=True)
    return model, cfg, sha(path)


def forward_ordered(model, x, device, increment=None):
    model.eval()
    if increment is not None:
        increment.eval()
    p, z = [], []
    with torch.no_grad():
        for start in range(0, 315, BATCH):
            xb = torch.from_numpy(x[start:start + BATCH]).to(device)
            feature = model.feature_extractor(xb)
            z.append(feature.cpu())
            p.append(model.regressor(feature).flatten().cpu())
        p = torch.cat(p)
        z = torch.cat(z)
        if increment is None:
            return p.numpy().astype(np.float64), None, None
        d = torch.cat([increment(z[i:min(i + BATCH, 314)].to(device),
                                 z[i + 1:i + BATCH + 1].to(device)).cpu()
                       for i in range(0, 314, BATCH)]).flatten()
        if len(d) != 314 or bool((d < 0).any()):
            raise AssertionError("Nonnegative adjacent increment invariant failed")
        c = torch.cat((p[:1], p[:1] + torch.cumsum(d, 0)))
        return p.numpy().astype(np.float64), c.numpy().astype(np.float64), d.numpy().astype(np.float64)


class IncrementHead(nn.Module):
    def __init__(self, source_labels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Linear(64, 1))
        median = float(np.median(np.maximum(np.diff(source_labels), 0)))
        median = max(median, 1e-3)
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            self.net[-1].bias.fill_(float(np.log(np.expm1(median))))

    def forward(self, previous, current):
        if previous.shape != current.shape or previous.shape[-1] != 512:
            raise ValueError("Increment needs matching 512-dimensional features")
        return F.softplus(self.net(torch.cat((previous, current, current - previous), dim=-1))).flatten()


def gate(out: Path, device):
    if (out / "gate.json").exists():
        existing = json.loads((out / "gate.json").read_text(encoding="utf-8"))
        if existing.get("passed") and len(existing.get("rows", [])) == 30:
            return
        raise RuntimeError("Existing gate is incomplete or failed")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, target, seed in pairs():
        model, cfg, digest = load_original(source, target, seed, device)
        _, x_target, mean, std = inputs(source, target)
        if (not np.array_equal(mean, np.asarray(cfg["source_normalization_mean"], dtype=np.float32))
                or not np.array_equal(std, np.asarray(cfg["source_normalization_std"], dtype=np.float32))):
            raise ValueError(f"Normalization differs: {source}->{target}/{seed}")
        replay, _, _ = forward_ordered(model, x_target, device)
        historical = pd.read_csv(baseline_path(source, target, seed), usecols=["cut_index", "pred_vb"])
        if historical.cut_index.tolist() != CUTS:
            raise ValueError("Historical baseline has wrong cut order")
        deviation = np.abs(replay - historical.pred_vb.to_numpy(np.float64))
        row = dict(source=source, target=target, seed=seed, checkpoint_sha256=digest,
                   baseline_prediction_sha256=sha(baseline_path(source, target, seed)),
                   max_abs_vb=float(deviation.max()), mean_abs_vb=float(deviation.mean()),
                   passed=bool(deviation.max() <= 1e-4))
        rows.append(row)
        print(f"gate {source}->{target} seed {seed}: max |Δ|={deviation.max():.8g} VB", flush=True)
        del model
    result = {"passed": all(row["passed"] for row in rows), "tolerance_vb": 1e-4,
              "mode": "model.eval(); frozen checkpoint; source-normalized cached STFT; ordered batches of 63",
              "rows": rows}
    write_json(out / "gate.json", result)
    if not result["passed"]:
        raise RuntimeError("F0 failed to replay original DARE head; new-method scoring is stopped")


def train_one(out, source, target, seed, group, device, raw_root):
    folder = out / f"{source}_to_{target}" / f"seed_{seed}" / group
    if folder.exists():
        if (folder / "complete.json").exists():
            return
        raise RuntimeError(f"Incomplete existing run; inspect before retry: {folder}")
    folder.mkdir(parents=True)
    base.seed_everything(seed)
    model, cfg, parent_sha = load_original(source, target, seed, device)
    x_s, x_t, mean, std = inputs(source, target)
    labels = base.wear_labels(raw_root, source)
    if sha(raw_root / f"{source}_wear.csv") != cfg["source_wear_sha256"]:
        raise ValueError("Source wear differs from original training")
    has_increment = group in ("F2", "F3")
    has_monotone = group in ("F1", "F3")
    increment = IncrementHead(labels).to(device) if has_increment else None
    # Fixed BN buffers are the audit-confirmed statistics used by eval inference.
    optimizer = torch.optim.Adam([
        {"params": model.feature_extractor.parameters(), "lr": BACKBONE_LR},
        {"params": model.regressor.parameters(), "lr": HEAD_LR},
        *([{"params": increment.parameters(), "lr": INCREMENT_LR}] if increment else []),
    ])
    y = torch.from_numpy(labels).to(device)
    source_tensor = torch.from_numpy(x_s)
    target_tensor = torch.from_numpy(x_t)
    source_chronology = chronological_positions(cfg["source_cuts"])
    target_chronology = chronological_positions(cfg["target_unlabeled_cuts"])
    source_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 1)
    epoch_rows = []
    gram_device = type("Device", (), {"device": device})()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if increment is not None:
            increment.train()
        optimizer.zero_grad(set_to_none=True)
        totals = dict(direct=0., gram=0., monotone=0., delta=0., cumulative=0.)
        source_order = torch.randperm(315, generator=source_generator)
        target_order = torch.randperm(315, generator=target_generator)
        for start in range(0, 315, BATCH):
            source_ids = source_order[start:start + BATCH]
            target_ids = target_order[start:start + BATCH]
            fs = model.feature_extractor(source_tensor[source_ids].to(device))
            ft = model.feature_extractor(target_tensor[target_ids].to(device))
            p = model.regressor(fs).flatten()
            direct = F.mse_loss(p, y[source_ids.to(device)])
            gram = GRAM(gram_device, fs, ft)
            loss = direct + DARE_WEIGHT * gram
            totals["direct"] += float(direct.detach()) / 5
            totals["gram"] += float(gram.detach()) / 5
            (loss / 5).backward()
        if has_monotone:
            previous_t = None
            for start in range(0, 315, BATCH):
                ft = model.feature_extractor(target_tensor[target_chronology[start:start + BATCH]].to(device))
                pt = model.regressor(ft).flatten()
                # Only true cut i -> i+1 pairs, including the block boundary.
                ordered_pt = torch.cat((previous_t, pt)) if previous_t is not None else pt
                monotone = F.relu(ordered_pt[:-1] - ordered_pt[1:]).square().mean()
                (MONOTONE_WEIGHT * monotone / 5).backward()
                totals["monotone"] += float(monotone.detach()) / 5
                previous_t = pt[-1:].detach()
        if has_increment:
            previous_z = previous_c = None
            for start in range(0, 315, BATCH):
                end = start + BATCH
                fs = model.feature_extractor(source_tensor[source_chronology[start:end]].to(device))
                p = model.regressor(fs).flatten()
                ordered_fs = torch.cat((previous_z, fs)) if previous_z is not None else fs
                d = increment(ordered_fs[:-1], ordered_fs[1:])
                expected = torch.clamp(y[start + (0 if start else 1):end] -
                                       y[start - 1 if start else 0:end - 1], min=0)
                if len(d) != len(expected):
                    raise AssertionError("Source increments are misaligned")
                delta = F.mse_loss(d, expected)
                c = (p[:1] if start == 0 else previous_c) + torch.cumsum(
                    torch.cat((torch.zeros(1, device=device), d)) if start == 0 else d, 0)
                if len(c) != len(p):
                    raise AssertionError("Cumulative trajectory has wrong length")
                cumulative = F.mse_loss(c, y[start:end])
                ((DELTA_WEIGHT * delta + CUMULATIVE_WEIGHT * cumulative) / 5).backward()
                totals["delta"] += float(delta.detach()) / 5
                totals["cumulative"] += float(cumulative.detach()) / 5
                previous_c = c[-1:].detach()
                previous_z = fs[-1:].detach()
        optimizer.step()
        epoch_rows.append({"epoch": epoch, **totals,
                           "source_order_sha256": hashlib.sha256(source_order.numpy().astype(np.int32).tobytes()).hexdigest(),
                           "target_order_sha256": hashlib.sha256(target_order.numpy().astype(np.int32).tobytes()).hexdigest()})
        print(f"{source}->{target} seed {seed} {group} epoch {epoch}/{EPOCHS}: {totals}", flush=True)
    model.eval()
    if increment is not None:
        increment.eval()
    checkpoint = folder / "final.pth"
    torch.save({"model": model.state_dict(), "increment": increment.state_dict() if increment else None,
                "epoch": EPOCHS, "parent_checkpoint_sha256": parent_sha}, checkpoint)
    p, c, d = forward_ordered(model, x_t, device, increment)
    pd.DataFrame({"cut_index": CUTS, "p_vb": p,
                  "c_vb": c if c is not None else [np.nan] * 315,
                  "d_vb": np.r_[np.nan, d] if d is not None else [np.nan] * 315}).to_csv(
                      folder / "target_predictions_unscored.csv", index=False, float_format="%.17g")
    pd.DataFrame(epoch_rows).to_csv(folder / "epoch_losses.csv", index=False)
    write_json(folder / "config.json", {
        "source": source, "target": target, "seed": seed, "group": group,
        "parent_checkpoint_sha256": parent_sha, "epochs": EPOCHS, "batch_size": BATCH,
        "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR, "increment_lr": INCREMENT_LR,
        "dare_weight": DARE_WEIGHT, "monotone_weight": MONOTONE_WEIGHT if has_monotone else 0,
        "delta_weight": DELTA_WEIGHT if has_increment else 0,
        "cumulative_weight": CUMULATIVE_WEIGHT if has_increment else 0,
        "source_normalization_mean": mean.tolist(), "source_normalization_std": std.tolist(),
        "bn": "original checkpoint running statistics frozen during continuation and used in model.eval inference",
        "batching": "independent source/target shuffle for direct+DARE; ordered extra passes for temporal losses; one optimizer update per epoch",
        "primary_curve": "c_vb" if has_increment else "p_vb",
        "target_labels_training_reads": 0,
        "target_cut_order": CUTS,
        "source_delta_target": "max(y_t-y_(t-1),0); original y used for direct and cumulative supervision",
        "source_anchor": "p_1", "target_anchor": "p_1"})
    write_json(folder / "complete.json", {
        "checkpoint_sha256": sha(checkpoint),
        "predictions_sha256": sha(folder / "target_predictions_unscored.csv"),
        "config_sha256": sha(folder / "config.json")})


def train(out, device, raw_root):
    gate_file = out / "gate.json"
    if not gate_file.exists() or not json.loads(gate_file.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("F0 replay gate must pass before training")
    if (out / "prediction_lock.json").exists():
        raise RuntimeError("Predictions already locked")
    for source, target, seed in pairs():
        for group in GROUPS:
            train_one(out, source, target, seed, group, device, raw_root)
    files = {}
    for source, target, seed in pairs():
        for group in GROUPS:
            folder = out / f"{source}_to_{target}" / f"seed_{seed}" / group
            files[str((folder / "final.pth").resolve())] = sha(folder / "final.pth")
            files[str((folder / "target_predictions_unscored.csv").resolve())] = sha(folder / "target_predictions_unscored.csv")
            files[str((folder / "config.json").resolve())] = sha(folder / "config.json")
    write_json(out / "prediction_lock.json", {"gate_sha256": sha(gate_file),
                                               "primary_curve": {g: "c_vb" if g in ("F2", "F3") else "p_vb" for g in GROUPS},
                                               "files": files})


def measures(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"R2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan,
            "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "signed_bias": float(np.mean(e))}


def score(out, raw_root):
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    if sha(out / "gate.json") != lock["gate_sha256"]:
        raise RuntimeError("F0 gate changed after prediction lock")
    for path, digest in lock["files"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Locked file changed: {path}")
    if (out / "per_seed_metrics.csv").exists():
        raise RuntimeError("Scoring output already exists")
    truth = {tool: base.wear_labels(raw_root, tool).astype(np.float64) for tool in TOOLS}
    metrics, increments, drifts, curves = [], [], [], []
    for source, target, seed in pairs():
        for group in GROUPS:
            folder = out / f"{source}_to_{target}" / f"seed_{seed}" / group
            pred = pd.read_csv(folder / "target_predictions_unscored.csv")
            if pred.cut_index.tolist() != CUTS:
                raise ValueError("Prediction cut order invalid")
            for curve in ("p_vb", "c_vb"):
                if pred[curve].isna().all():
                    continue
                p = pred[curve].to_numpy(np.float64)
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite prediction")
                y = truth[target]
                for stage, sl in (("full", slice(None)), ("early", slice(0, 105)),
                                  ("middle", slice(105, 210)), ("late", slice(210, 315))):
                    metrics.append(dict(source=source, target=target, seed=seed, group=group,
                                        curve=curve, primary=(curve == lock["primary_curve"][group]),
                                        stage=stage, n=len(y[sl]), **measures(y[sl], p[sl])))
                error = p - y
                drifts.append(dict(source=source, target=target, seed=seed, group=group, curve=curve,
                                   early_abs_error=float(np.mean(abs(error[:105]))),
                                   late_abs_error=float(np.mean(abs(error[210:]))),
                                   late_minus_early_abs_error=float(np.mean(abs(error[210:])) - np.mean(abs(error[:105]))),
                                   cut_error_slope=float(np.polyfit(CUTS, error, 1)[0]),
                                   cut_abs_error_slope=float(np.polyfit(CUTS, abs(error), 1)[0])))
                curves.extend(dict(source=source, target=target, seed=seed, group=group, curve=curve,
                                   cut_index=i, true_vb=float(yt), pred_vb=float(pt), signed_error=float(pt - yt))
                              for i, yt, pt in zip(CUTS, y, p))
            if group in ("F2", "F3"):
                d = pred.d_vb.to_numpy(np.float64)[1:]
                if not np.isfinite(d).all() or (d < 0).any():
                    raise ValueError("Invalid nonnegative increments")
                increments.append(dict(source=source, target=target, seed=seed, group=group,
                                       min=float(d.min()), q05=float(np.quantile(d, .05)),
                                       median=float(np.median(d)), q95=float(np.quantile(d, .95)),
                                       max=float(d.max()), mean=float(d.mean()),
                                       near_zero_count=int((d < .01).sum())))
    pd.DataFrame(metrics).to_csv(out / "per_seed_metrics.csv", index=False)
    pd.DataFrame(increments).to_csv(out / "increment_distribution.csv", index=False)
    pd.DataFrame(drifts).to_csv(out / "cumulative_error_drift.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "target_predictions_scored.csv", index=False, float_format="%.17g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("gate", "train", "score"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.stage == "gate":
        gate(args.out, device)
    elif args.stage == "train":
        train(args.out, device, args.raw_root)
    else:
        score(args.out, args.raw_root)


if __name__ == "__main__":
    main()
