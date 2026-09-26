"""Paired PHM2010 COD add-on; existing baseline checkpoints remain untouched."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
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
from models.COD import cod_metric
from models.DAREGRAM import Trainset as DAREGRAM


SEEDS = (42, 43, 44, 45, 46)
PAIRS = tuple((s, t) for s in base.TOOLS for t in base.TOOLS if s != t)
EPOCHS, BATCH, LR = 50, 63, 1e-3
COD_WEIGHT, COD_EPSILON = 1e-3, 5e-2  # official MPI3D defaults
# Official MPI3D warmup is 3000/10000 steps; preserve that fraction in 250 steps.
COD_START_STEP = 75
METHODS = ("source_only", "daregram", "daregram_cod")


def sha(path):
    return base.file_hash(path)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_inputs(cache, raw_root, source, target):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["cuts"] != base.ALL_CUTS or manifest["feature_shape"] != [315, 6, 128, 128]:
        raise ValueError("Feature cache protocol mismatch")
    if manifest["stft_code_sha256"] != sha(Path(base.sampling.__file__)):
        raise ValueError("STFT code changed since baseline")
    arrays = {}
    for tool in (source, target):
        path = cache / f"{tool}_stft.npy"
        if sha(path) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"Feature cache hash mismatch: {path}")
        arrays[tool] = np.load(path, mmap_mode="r", allow_pickle=False)
        if arrays[tool].shape != (315, 6, 128, 128) or arrays[tool].dtype != np.float32:
            raise ValueError(f"Feature cache shape/dtype mismatch: {tool}")
    mean = arrays[source].mean(axis=(0, 2, 3)).astype(np.float32)
    std = (arrays[source].std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    xs = ((arrays[source] - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    xt = ((arrays[target] - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    ys = base.wear_labels(raw_root, source)
    return xs, ys, xt, mean, std, manifest


def train(method, xs, ys, xt, seed, device, epochs=EPOCHS, cod_weight=COD_WEIGHT,
          cod_start=COD_START_STEP, output=None):
    """No target labels/path enter this function; final epoch is selected."""
    model, init_hash = base.build_model(seed, device)
    source = DataLoader(TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys),
                                      torch.arange(1, 316)), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    target = DataLoader(TensorDataset(torch.from_numpy(xt), torch.arange(1, 316)),
                        batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    records, source_orders, target_orders = [], [], []
    used_target = set()
    grad_norm = None
    for epoch in range(1, epochs + 1):
        model.train()
        source_order, target_order = [], []
        sums = dict(source_mse=0.0, daregram=0.0, cod=0.0,
                    weighted_daregram=0.0, weighted_cod=0.0, total=0.0)
        tradeoff = 2 / (1 + math.exp(-10 * (epoch - 1) / (EPOCHS - 1))) - 1
        for batch_index, ((xb, yb, sc), (tb, tc)) in enumerate(zip(source, target)):
            step = (epoch - 1) * len(source) + batch_index + 1
            source_order.extend(sc.tolist())
            target_order.extend(tc.tolist())
            used_target.update(tc.tolist())
            optimizer.zero_grad()
            fs = model.feature_extractor(xb.to(device))
            ft = model.feature_extractor(tb.to(device))
            ps = model.regressor(fs)
            regression = F.mse_loss(ps, yb.to(device).unsqueeze(1))
            gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), fs, ft)
            weighted_gram = tradeoff * gram
            active = method == "daregram_cod" and cod_weight > 0 and step >= cod_start
            cod = fs.new_zeros(())
            weighted_cod = fs.new_zeros(())
            if active:
                pt = model.regressor(ft)
                # Official random support columns; target batch size comes from signals,
                # never from a target wear label tensor.
                source_condition = torch.cat((yb.to(device).unsqueeze(1),
                    1e-3 * torch.rand(len(yb), 4, device=device)), dim=1)
                target_condition = torch.cat((pt,
                    1e-3 * torch.rand(len(tb), 4, device=device)), dim=1).detach()
                cod = cod_metric(fs, ft, source_condition, target_condition,
                                 epsilon=COD_EPSILON)
                weighted_cod = cod_weight * cod
                if grad_norm is None:
                    gs, gt = torch.autograd.grad(weighted_cod, (fs, ft), retain_graph=True)
                    grad_norm = [float(gs.norm()), float(gt.norm())]
                    if not np.isfinite(grad_norm).all() or min(grad_norm) <= 0:
                        raise FloatingPointError(f"COD feature gradients invalid: {grad_norm}")
            loss = regression + weighted_gram + weighted_cod
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch} step={step}")
            loss.backward()
            optimizer.step()
            for key, val in (("source_mse", regression), ("daregram", gram),
                             ("cod", cod), ("weighted_daregram", weighted_gram),
                             ("weighted_cod", weighted_cod), ("total", loss)):
                sums[key] += float(val.detach())
        if sorted(source_order) != base.ALL_CUTS or sorted(target_order) != base.ALL_CUTS:
            raise ValueError("Batch order did not cover CUT 1..315")
        source_orders.append(hashlib.sha256(np.asarray(source_order, dtype=np.int32).tobytes()).hexdigest())
        target_orders.append(hashlib.sha256(np.asarray(target_order, dtype=np.int32).tobytes()).hexdigest())
        record = {"epoch": epoch, "cod_start_step": cod_start, "cod_weight": cod_weight,
                  "daregram_weight": tradeoff,
                  **{key: val / len(source) for key, val in sums.items()}}
        records.append(record)
        logging.info("%s seed=%d epoch=%d source=%.6f gram=%.6f cod=%.6f total=%.6f",
                     method, seed, epoch, record["source_mse"], record["daregram"],
                     record["cod"], record["total"])
    trace = {"initial_model_sha256": init_hash, "source_order_sha256_by_epoch": source_orders,
             "target_order_sha256_by_epoch": target_orders,
             "actual_unlabeled_target_cuts": sorted(used_target),
             "cod_gradient_norm_first_active_step": grad_norm,
             "epoch_losses": records}
    if output is not None:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        torch.save({"model": model.state_dict(), "epoch": epochs}, output / "final.pth")
        pd.DataFrame(records).to_csv(output / "epoch_losses.csv", index=False)
        write_json(output / "training_trace.json", trace)
    return model, trace


def predict(model, xt, device):
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, 315, BATCH):
            xb = torch.from_numpy(xt[offset:offset+BATCH]).to(device)
            values.extend(model.regressor(model.feature_extractor(xb)).cpu().numpy().ravel().tolist())
    return np.asarray(values, dtype=np.float64)


def smoke(args):
    source, target, seed = "c1", "c4", 42
    xs, ys, xt, _, _, _ = load_inputs(args.cache, args.raw_root, source, target)
    # The training API accepts no target labels. Run with the target wear file
    # unavailable to any training function, then repeat from the same seed.
    original = base.wear_labels
    def guarded(root, tool, evaluation_dir=None):
        if tool == target:
            raise AssertionError("Target labels accessed during training")
        return original(root, tool, evaluation_dir)
    model_cod, cod_trace = train("daregram_cod", xs, ys, xt, seed, args.device,
                                 epochs=1, cod_start=1)
    cod_hash = base.model_hash(model_cod)
    base.wear_labels = guarded
    try:
        model_repeat, _ = train("daregram_cod", xs, ys, xt, seed, args.device,
                                epochs=1, cod_start=1)
        repeat_hash = base.model_hash(model_repeat)
        model_zero, zero_trace = train("daregram_cod", xs, ys, xt, seed, args.device,
                                       epochs=1, cod_weight=0, cod_start=1)
        zero_hash = base.model_hash(model_zero)
        model_gram, gram_trace = train("daregram", xs, ys, xt, seed, args.device,
                                       epochs=1, cod_start=1)
        gram_hash = base.model_hash(model_gram)
    finally:
        base.wear_labels = original
    checks = {"forward_backward_finite": bool(np.isfinite(cod_trace["epoch_losses"][0]["total"])),
              "cod_feature_gradient_nonzero": min(cod_trace["cod_gradient_norm_first_active_step"]) > 0,
              "lambda_zero_equals_daregram": zero_hash == gram_hash and all(
                  zero_trace["epoch_losses"][0][key] == gram_trace["epoch_losses"][0][key]
                  for key in ("source_mse", "daregram", "cod", "weighted_daregram",
                              "weighted_cod", "total")),
              "target_labels_absent_and_deterministic": cod_hash == repeat_hash}
    result = {"direction": "C1->C4", "purpose": "program smoke test only; no metrics read",
              "checks": checks, "cod_gradient_norm": cod_trace["cod_gradient_norm_first_active_step"],
              "cod_model_sha256": cod_hash, "cod_repeat_sha256": repeat_hash,
              "lambda_zero_sha256": zero_hash, "daregram_sha256": gram_hash}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "smoke.json", result)
    if not all(checks.values()):
        raise AssertionError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def baseline_check(folder, source, target, seed, mean, std, manifest):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    checks = [(config["source"], config["target"], config["seed"]) == (source, target, seed),
              (config["epochs"], config["batch_size"], config["lr"]) == (EPOCHS, BATCH, LR),
              config["source_feature_sha256"] == manifest["tools"][source]["feature_sha256"],
              config["target_feature_sha256"] == manifest["tools"][target]["feature_sha256"],
              np.array_equal(np.asarray(config["source_normalization_mean"], np.float32), mean),
              np.array_equal(np.asarray(config["source_normalization_std"], np.float32), std),
              audit["target_label_reads_during_training"] == 0,
              audit["target_unlabeled_cuts"] == base.ALL_CUTS,
              audit["paired_initialization_verified"], audit["paired_source_order_verified"]]
    if not all(checks):
        raise ValueError(f"Baseline audit failed: {folder}")
    for method in ("source_only", "daregram"):
        method_config = json.loads((folder / method / "config.json").read_text(encoding="utf-8"))
        if method_config["initial_model_sha256"] != audit["same_initial_model_sha256"]:
            raise ValueError(f"Baseline initial model mismatch: {folder}")
        if sha(folder / method / "final.pth") != audit["checkpoint_sha256"][method]:
            raise ValueError(f"Baseline checkpoint mismatch: {folder}")
    return config, audit


def run_one(args, source, target, seed):
    folder = args.out / f"{source}_to_{target}" / f"seed_{seed}"
    if folder.exists():
        if (folder / "complete.json").exists():
            logging.info("Verified completed %s", folder)
            return
        raise FileExistsError(f"Incomplete output requires manual audit: {folder}")
    xs, ys, xt, mean, std, manifest = load_inputs(args.cache, args.raw_root, source, target)
    baseline = args.baseline / f"{source}_to_{target}" / f"seed_{seed}"
    config, audit = baseline_check(baseline, source, target, seed, mean, std, manifest)
    output = folder / "daregram_cod"
    model, trace = train("daregram_cod", xs, ys, xt, seed, args.device, output=output)
    baseline_dare = json.loads((baseline / "daregram" / "config.json").read_text(encoding="utf-8"))
    if trace["initial_model_sha256"] != audit["same_initial_model_sha256"] or \
       trace["source_order_sha256_by_epoch"] != baseline_dare["source_order_sha256_by_epoch"]:
        raise ValueError("COD initialization/source order differ from paired baselines")
    if trace["actual_unlabeled_target_cuts"] != base.ALL_CUTS:
        raise ValueError("COD did not use all unlabeled target cuts")
    cod_pred = predict(model, xt, args.device)
    del model
    # All training checkpoints and the COD prediction are fixed before this read.
    if not (output / "final.pth").exists():
        raise RuntimeError("COD checkpoint missing before evaluation")
    truth = base.wear_labels(args.raw_root, target).astype(np.float64)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for method in METHODS:
        destination = folder / method
        destination.mkdir(exist_ok=True)
        if method == "daregram_cod":
            pred = cod_pred
        else:
            checkpoint = torch.load(baseline / method / "final.pth", map_location=args.device,
                                    weights_only=True)
            loaded, _ = base.build_model(seed, args.device)
            loaded.load_state_dict(checkpoint["model"])
            pred = predict(loaded, xt, args.device)
            del loaded
        csv_path = destination / "predictions.csv"
        pd.DataFrame({"cut_index": base.ALL_CUTS, "true_vb": truth,
                      "pred_vb": pred}).to_csv(csv_path, index=False, float_format="%.17g")
        result = base.metrics(truth, pred)
        write_json(destination / "metrics.json", result)
        records.append({"source": source, "target": target, "seed": seed,
                        "method": method, "n": 315, **result})
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(base.ALL_CUTS, truth, label="True VB")
        ax.plot(base.ALL_CUTS, pred, label="Predicted VB")
        ax.set(xlim=(1, 315), xlabel=f"{target.upper()} cut index", ylabel="VB",
               title=f"{source.upper()}->{target.upper()} seed {seed}: {method}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / "prediction.png", dpi=200)
        plt.close(fig)
    write_json(folder / "config.json", {"source": source, "target": target, "seed": seed,
        "epochs": EPOCHS, "batch_size": BATCH, "lr": LR, "checkpoint_selection": "final epoch",
        "source_normalization_mean": mean.tolist(), "source_normalization_std": std.tolist(),
        "cod_weight": COD_WEIGHT, "cod_epsilon": COD_EPSILON,
        "cod_start_step": COD_START_STEP, "target_labels_training_reads": 0,
        "baseline_path": str(baseline.resolve()),
        "baseline_source_checkpoint_sha256": sha(baseline / "source_only" / "final.pth"),
        "baseline_daregram_checkpoint_sha256": sha(baseline / "daregram" / "final.pth"),
        "cod_checkpoint_sha256": sha(output / "final.pth")})
    pd.DataFrame(records).to_csv(folder / "metrics.csv", index=False)
    write_json(folder / "complete.json", {"methods": list(METHODS), "evaluation_cuts": base.ALL_CUTS})
    logging.info("Completed %s->%s seed=%d", source, target, seed)


def summarize(args):
    rows = []
    for source, target in PAIRS:
        for seed in SEEDS:
            path = args.out / f"{source}_to_{target}" / f"seed_{seed}" / "metrics.csv"
            if path.exists():
                rows.append(pd.read_csv(path))
    if not rows:
        return
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(args.out / "per_seed_metrics.csv", index=False)
    summary = all_rows.groupby(["source", "target", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), R2=("R2", "mean"), MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"), MAPE_percent=("MAPE_percent", "mean"))
    summary.to_csv(args.out / "six_direction_three_method_metrics.csv", index=False)
    paired = all_rows.pivot(index=["source", "target", "seed"], columns="method",
                            values=["R2", "MAE", "RMSE", "MAPE_percent"])
    delta = pd.DataFrame({metric: paired[(metric, "daregram_cod")] - paired[(metric, "daregram")]
                          for metric in ("R2", "MAE", "RMSE", "MAPE_percent")}).reset_index()
    delta.drop(columns="seed").groupby(["source", "target"], as_index=False).mean(numeric_only=True).to_csv(
        args.out / "cod_minus_daregram_by_direction.csv", index=False)
    if len(all_rows) == len(PAIRS) * len(SEEDS) * len(METHODS):
        curves = []
        for source, target in PAIRS:
            for seed in SEEDS:
                for method in METHODS:
                    curves.append(args.out / f"{source}_to_{target}" / f"seed_{seed}" /
                                  method / "predictions.csv")
        limits = np.concatenate([pd.read_csv(path)[["true_vb", "pred_vb"]].to_numpy().ravel()
                                 for path in curves])
        lower, upper = float(limits.min()), float(limits.max())
        margin = (upper - lower) * 0.03
        for path in curves:
            data = pd.read_csv(path)
            parts = path.parts
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(data.cut_index, data.true_vb, label="True VB")
            ax.plot(data.cut_index, data.pred_vb, label="Predicted VB")
            ax.set(xlim=(1, 315), ylim=(lower - margin, upper + margin),
                   xticks=np.arange(1, 316, 50), xlabel="Target cut index", ylabel="VB",
                   title=f"{parts[-4]} {parts[-3]} {parts[-2]}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path.with_name("prediction.png"), dpi=200)
            plt.close(fig)
        write_json(args.out / "plot_coordinates.json", {"xlim": [1, 315],
            "ylim": [lower - margin, upper + margin],
            "xticks": np.arange(1, 316, 50).tolist()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "run", "summarize"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/five_seed_paired/feature_cache"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cod_comparison_20260926"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.out.resolve() == args.baseline.resolve() or args.baseline.resolve() in args.out.resolve().parents:
        parser.error("Output cannot overwrite or nest inside baseline artifacts")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "run":
        smoke_file = args.out / "smoke.json"
        if not smoke_file.exists() or not all(json.loads(smoke_file.read_text(encoding="utf-8"))["checks"].values()):
            raise RuntimeError("Passing smoke test required before formal training")
        for source, target in PAIRS:
            for seed in SEEDS:
                run_one(args, source, target, seed)
                summarize(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
