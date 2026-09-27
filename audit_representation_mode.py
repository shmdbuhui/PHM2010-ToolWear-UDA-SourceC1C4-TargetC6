"""Frozen-checkpoint source forward audit in eval and train modes (no updates to files)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_single_source_pairs as base

TOOLS = ("c1", "c4", "c6")
SEEDS = range(42, 47)
METHODS = ("source_only", "daregram")
CUTS = list(range(1, 316))
STAGES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")
F_COLS = [f"feature_{i:03d}" for i in range(512)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def metrics(y, p):
    e = p - y
    return {"MSE": float(np.mean(e * e)), "MAE": float(np.mean(abs(e))),
            "RMSE": float(np.sqrt(np.mean(e * e))),
            "R2": float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2))}


def bn_rows(model, mode, moment, identity):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        mean = module.running_mean.detach().cpu().numpy()
        var = module.running_var.detach().cpu().numpy()
        rows.append(dict(identity, mode=mode, moment=moment, layer=name,
                         running_mean_mean=float(mean.mean()), running_mean_std=float(mean.std()),
                         running_mean_min=float(mean.min()), running_mean_max=float(mean.max()),
                         running_var_mean=float(var.mean()), running_var_std=float(var.std()),
                         running_var_min=float(var.min()), running_var_max=float(var.max()),
                         running_mean_l2=float(np.linalg.norm(mean)),
                         running_var_l2=float(np.linalg.norm(var)),
                         num_batches_tracked=int(module.num_batches_tracked.item())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    parser.add_argument("--train-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/representation_probe_audit_20260926"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.out_root.exists() or args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Fresh output directory and available device required")
    old = pd.read_csv(args.prior_root / "provenance.csv")
    index = pd.read_csv(args.prior_root / "feature_index.csv")
    if len(old) != 60 or len(index) != 60:
        raise ValueError("Expected 60 prior provenance entries")
    join = old.merge(index, on=["source", "target", "seed", "method", "checkpoint_sha256"], validate="one_to_one")
    if len(join) != 60:
        raise ValueError("Prior provenance/feature index mismatch")
    cache = json.loads((args.train_root / "feature_cache" / "manifest.json").read_text(encoding="utf-8"))
    if cache["cuts"] != CUTS or cache["feature_shape"] != [315, 6, 128, 128]:
        raise ValueError("Wrong feature cache cut mapping")
    for row in join.itertuples():
        checkpoint = Path(row.checkpoint)
        config = json.loads(Path(row.training_config).read_text(encoding="utf-8"))
        audit = json.loads(Path(row.feature_audit).read_text(encoding="utf-8"))
        if (sha(checkpoint) != row.checkpoint_sha256 or audit["checkpoint_sha256"] != row.checkpoint_sha256
                or (config["source"], config["target"], config["seed"], config["method"]) !=
                (row.source, row.target, row.seed, row.method)
                or config["source_cuts"] != CUTS or config["target_unlabeled_cuts"] != CUTS
                or config["input"] != "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128"
                or config["regressor"] != "Linear(512, 1)"
                or config["source_feature_sha256"] != cache["tools"][row.source]["feature_sha256"]
                or config["target_feature_sha256"] != cache["tools"][row.target]["feature_sha256"]):
            raise ValueError(f"Checkpoint/protocol mismatch: {checkpoint}")
        for role, path in (("source", row.source_features), ("target", row.target_features)):
            frame = pd.read_csv(path, usecols=["tool_id", "cut_index"])
            if (sha(Path(path)) != audit["feature_stats"][role]["file_sha256"]
                    or frame.tool_id.tolist() != [getattr(row, role)] * 315 or frame.cut_index.tolist() != CUTS):
                raise ValueError(f"Feature/cut mismatch: {path}")
    source_data = {}
    for source in TOOLS:
        raw = np.load(args.train_root / "feature_cache" / f"{source}_stft.npy", mmap_mode="r", allow_pickle=False)
        if raw.shape != (315, 6, 128, 128) or sha(args.train_root / "feature_cache" / f"{source}_stft.npy") != cache["tools"][source]["feature_sha256"]:
            raise ValueError(f"Source cache mismatch: {source}")
        mean = raw.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        x = ((raw - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
        y = base.wear_labels(args.raw_root, source).astype(np.float64)
        source_data[source] = x, y, mean, std
    out = args.out_root
    out.mkdir(parents=True)
    (out / "mode_features").mkdir()
    (out / "mode_predictions").mkdir()
    mode_rows, pred_rows, batch_rows, bn_all, gap_rows, run_rows = [], [], [], [], [], []
    device = torch.device(args.device)
    for row in join.itertuples():
        identity = {k: getattr(row, k) for k in ("source", "target", "seed", "method")}
        config = json.loads(Path(row.training_config).read_text(encoding="utf-8"))
        x, y, mean, std = source_data[row.source]
        if not np.array_equal(mean, np.asarray(config["source_normalization_mean"], dtype=np.float32)) or not np.array_equal(std, np.asarray(config["source_normalization_std"], dtype=np.float32)):
            raise ValueError(f"Source normalization mismatch: {identity}")
        checkpoint_hash_before = sha(Path(row.checkpoint))
        state = torch.load(row.checkpoint, map_location="cpu", weights_only=True)["model"]
        models = {}
        snapshots = {}
        activations = {}
        for mode in ("eval", "train"):
            model, _ = base.build_model(row.seed, device)
            model.load_state_dict(state, strict=True)
            model.eval() if mode == "eval" else model.train()
            models[mode] = model
            activations[mode] = {}
            for name in STAGES:
                module = getattr(model.feature_extractor.backbone, name)
                def capture(_module, _inputs, output, mode=mode, name=name):
                    t = output.detach()
                    if t.ndim == 4:
                        t = t.mean(dim=(-1, -2))
                    activations[mode][name] = t.float().cpu().numpy()
                module.register_forward_hook(capture)
            bn_all.extend(bn_rows(model, mode, "before", identity))
            snapshots[mode] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        features = {"eval": [], "train": []}
        predictions = {"eval": [], "train": []}
        base.seed_everything(row.seed)
        with torch.no_grad():
            for batch_number, start in enumerate(range(0, 315, 63), 1):
                positions = list(range(start + 1, min(start + 64, 316)))
                xb = torch.from_numpy(x[start:start + 63]).to(device)
                batch_rows.append(dict(identity, batch=batch_number, first_cut=positions[0], last_cut=positions[-1],
                                       cut_indices=",".join(map(str, positions)), input_shape=str(tuple(xb.shape)),
                                       checkpoint_sha256=row.checkpoint_sha256))
                for mode in ("eval", "train"):
                    feat = models[mode].feature_extractor(xb)
                    pred = models[mode].regressor(feat)
                    features[mode].append(feat.cpu().numpy())
                    predictions[mode].append(pred.cpu().numpy().reshape(-1))
                for name in STAGES:
                    a, b = activations["eval"][name], activations["train"][name]
                    diff = a.astype(np.float64) - b.astype(np.float64)
                    gap_rows.append(dict(identity, batch=batch_number, layer=name,
                                         channel_mean_rmse=float(np.sqrt(np.mean(diff ** 2))),
                                         relative_channel_mean_rmse=float(np.sqrt(np.mean(diff ** 2)) /
                                                                          max(np.sqrt(np.mean(a.astype(np.float64) ** 2)), 1e-12))))
        for mode in ("eval", "train"):
            bn_all.extend(bn_rows(models[mode], mode, "after", identity))
            if mode == "eval" and any(not torch.equal(value, models[mode].state_dict()[key].cpu()) for key, value in snapshots[mode].items()):
                raise ValueError(f"Eval state changed: {identity}")
            f = np.concatenate(features[mode])
            p = np.concatenate(predictions[mode])
            if f.shape != (315, 512) or p.shape != (315,) or not np.isfinite(f).all() or not np.isfinite(p).all():
                raise ValueError(f"Invalid mode output: {identity}, {mode}")
            name = f"{row.source}_to_{row.target}_seed_{row.seed}_{row.method}_{mode}.csv"
            pd.DataFrame({"tool_id": [row.source] * 315, "cut_index": CUTS, **{key: f[:, i] for i, key in enumerate(F_COLS)}}).to_csv(
                out / "mode_features" / name, index=False, float_format="%.9g")
            pd.DataFrame({"cut_index": CUTS, "true_vb": y, "pred_vb": p}).to_csv(
                out / "mode_predictions" / name, index=False, float_format="%.17g")
            for cut in CUTS:
                pred_rows.append(dict(identity, mode=mode, cut_index=cut, true_vb=y[cut - 1], pred_vb=p[cut - 1],
                                      checkpoint_sha256=row.checkpoint_sha256))
            mode_rows.append(dict(identity, mode=mode, checkpoint_sha256=row.checkpoint_sha256,
                                  **metrics(y, p)))
        if sha(Path(row.checkpoint)) != checkpoint_hash_before:
            raise ValueError(f"Checkpoint file changed: {row.checkpoint}")
        log = Path(row.checkpoint).parent / "run.log"
        match = re.search(rf"^{row.method} epoch=50/50 source_MSE=([\d.eE+-]+) gram=([\d.eE+-]+)", log.read_text(encoding="utf-8"), re.MULTILINE)
        if not match:
            raise ValueError(f"Historical epoch-50 log absent: {log}")
        run_rows.append(dict(identity, checkpoint_sha256=row.checkpoint_sha256,
                             checkpoint=row.checkpoint, training_config=row.training_config,
                             historical_epoch50_batchmean_supervised_mse=float(match[1]),
                             historical_epoch50_batchmean_alignment_loss=float(match[2]),
                             original_checkpoint_unchanged=True,
                             label_definition="mean of three flute VB; no label standardization or inverse transform"))
        print(f"mode audit {row.source}->{row.target} seed={row.seed} {row.method}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics_frame = pd.DataFrame(mode_rows)
    paired = metrics_frame.pivot(index=["source", "target", "seed", "method", "checkpoint_sha256"], columns="mode", values=["MSE", "MAE", "RMSE", "R2"])
    paired.columns = [f"{name}_{mode}" for name, mode in paired.columns]
    paired = paired.reset_index()
    for name in ("MSE", "MAE", "RMSE", "R2"):
        paired[f"train_minus_eval_{name}"] = paired[f"{name}_train"] - paired[f"{name}_eval"]
    pred_frame = pd.DataFrame(pred_rows)
    pred_pair = pred_frame.pivot(index=["source", "target", "seed", "method", "cut_index"], columns="mode", values="pred_vb").reset_index()
    pred_pair["train_minus_eval_pred_vb"] = pred_pair["train"] - pred_pair["eval"]
    pred_pair.to_csv(out / "train_eval_per_cut_predictions.csv", index=False)
    difference = pred_pair.groupby(["source", "target", "seed", "method"]).train_minus_eval_pred_vb.agg(
        prediction_mean_abs_difference=lambda x: float(np.mean(abs(x))),
        prediction_max_abs_difference=lambda x: float(np.max(abs(x)))).reset_index()
    paired = paired.merge(difference, on=["source", "target", "seed", "method"], validate="one_to_one")
    paired = paired.merge(pd.DataFrame(run_rows), on=["source", "target", "seed", "method", "checkpoint_sha256"], validate="one_to_one")
    paired.to_csv(out / "mode_metrics.csv", index=False)
    seed_comparison = []
    for (source, target, seed), block in paired.groupby(["source", "target", "seed"]):
        if set(block.method) != set(METHODS):
            raise ValueError("Missing paired mode audit")
        so = block[block.method == "source_only"].iloc[0]
        dare = block[block.method == "daregram"].iloc[0]
        record = {"source": source, "target": target, "seed": seed}
        for metric in ("MSE", "MAE", "RMSE", "R2"):
            for mode in ("eval", "train"):
                key = f"{metric}_{mode}"
                record[f"source_only_{key}"] = so[key]
                record[f"daregram_{key}"] = dare[key]
                record[f"dare_minus_source_only_{key}"] = dare[key] - so[key]
        seed_comparison.append(record)
    seed_frame = pd.DataFrame(seed_comparison)
    seed_frame.to_csv(out / "mode_paired_seed.csv", index=False)
    direction_rows = []
    for (source, target), block in seed_frame.groupby(["source", "target"]):
        record = {"source": source, "target": target, "n_seeds": 5}
        for key in block.columns:
            if key in ("source", "target", "seed"):
                continue
            record[f"{key}_mean"] = float(block[key].mean())
            record[f"{key}_sd"] = float(block[key].std(ddof=1))
        direction_rows.append(record)
    pd.DataFrame(direction_rows).to_csv(out / "mode_direction_summary.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(out / "batch_cut_trace.csv", index=False)
    pd.DataFrame(bn_all).to_csv(out / "batchnorm_buffers.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(out / "layer_mode_gaps.csv", index=False)
    print(f"Completed {len(paired)} checkpoint mode comparisons")


if __name__ == "__main__":
    main()
