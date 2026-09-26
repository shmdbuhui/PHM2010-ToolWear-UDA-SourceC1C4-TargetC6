"""Run 5 paired seeds for every directed PHM2010 single-tool transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

import run_single_source_pairs as base
from trend_physics import trend_loss
from models.DAREGRAM import Trainset as DAREGRAM


SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 50
BATCH_SIZE = 63
LR = 1e-3
METHODS = ("source_only", "daregram")
PAIRS = tuple((source, target) for source in base.TOOLS for target in base.TOOLS if source != target)
OLD_ROOT = Path("artifacts/single_source_pairs_seed42")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--split-record", type=Path, default=Path(r"E:\QLP\test-9.3\outputs\predictions_comparison.csv"))
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.out_root.resolve() == args.old_root.resolve():
        parser.error("New output root must differ from the completed single-run root")
    return args


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint_raw(root, tool):
    files = base.signal_files(root, tool)
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def prepare_cache(args):
    cache = args.out_root / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    code_hash = base.file_hash(Path(base.sampling.__file__))
    manifest_path = cache / "manifest.json"
    manifest = {"stft_code_sha256": code_hash, "raw_root": str(args.raw_root.resolve()),
                "cuts": base.ALL_CUTS, "feature_shape": [315, 6, 128, 128], "tools": {}}
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old_manifest is not None and (old_manifest["stft_code_sha256"] != code_hash or
                                     old_manifest["raw_root"] != manifest["raw_root"]):
        raise ValueError("Existing feature cache comes from a different raw root or STFT implementation")
    for tool in base.TOOLS:
        raw_fingerprint = fingerprint_raw(args.raw_root, tool)
        path = cache / f"{tool}_stft.npy"
        if old_manifest is not None and tool in old_manifest["tools"]:
            old = old_manifest["tools"][tool]
            if old["raw_fingerprint"] != raw_fingerprint or old["feature_sha256"] != base.file_hash(path):
                raise ValueError(f"Cached {tool} features do not match recorded raw files or checksum")
            logging.info("Verified feature-only STFT cache: %s", tool)
            manifest["tools"][tool] = old
        else:
            if path.exists():
                raise FileExistsError(f"Unrecorded cache file: {path}")
            features = base.stft_features(args.raw_root, tool)
            np.save(path, features)
            del features
            manifest["tools"][tool] = {"raw_fingerprint": raw_fingerprint,
                                       "feature_sha256": base.file_hash(path)}
            json_write(manifest_path, manifest)
            logging.info("Saved feature-only STFT cache: %s", path)
    json_write(manifest_path, manifest)
    return manifest


def cache_features(args, tool):
    features = np.load(args.out_root / "feature_cache" / f"{tool}_stft.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (315, 6, 128, 128) or features.dtype != np.float32:
        raise ValueError(f"Invalid cached features for {tool}")
    return features


def metrics_from_csv(path, expected_cuts):
    frame = pd.read_csv(path)
    if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"]:
        raise ValueError(f"Unexpected prediction CSV schema: {path}")
    if frame.cut_index.astype(int).tolist() != expected_cuts or len(frame) != len(expected_cuts):
        raise ValueError(f"Prediction cuts disagree with evaluation cuts: {path}")
    if not np.isfinite(frame[["true_vb", "pred_vb"]].to_numpy()).all():
        raise ValueError(f"Nonfinite predictions: {path}")
    return base.metrics(frame.true_vb.to_numpy(), frame.pred_vb.to_numpy())


def audit_prior_seed42(args, manifest):
    """Inspect old results without modifying them; reject reuse unless fully provable."""
    findings = []
    preserved = {}
    for source, target in PAIRS:
        pair = args.old_root / f"{source}_to_{target}"
        config = json.loads((pair / "config.json").read_text(encoding="utf-8"))
        audit = json.loads((pair / "audit.json").read_text(encoding="utf-8"))
        expected_cuts, _ = base.evaluation_cuts(target, args.split_record)
        raw_source = cache_features(args, source)
        mean = raw_source.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw_source.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        _, expected_init = base.build_model(42, torch.device("cpu"))
        checks = {
            "source_target_seed": (config["source"], config["target"], config["seed"]) == (source, target, 42),
            "training_hyperparameters": (config["epochs"], config["batch_size"], config["lr"]) ==
                                        (EPOCHS, BATCH_SIZE, LR),
            "raw_root": config["raw_root"] == str(args.raw_root.resolve()),
            "split_record": config["split_record"] == str(args.split_record.resolve()),
            "source_normalization": np.array_equal(mean, np.asarray(config["normalization_mean"], dtype=np.float32))
                                    and np.array_equal(std, np.asarray(config["normalization_std"], dtype=np.float32)),
            "source_count": config["source_sample_count"] == audit["source_sample_count"] == 315,
            "unlabeled_target_cuts": config["target_unlabeled_cuts"] == audit["target_unlabeled_cuts"] == base.ALL_CUTS,
            "evaluation_cuts": config["evaluation_cuts"] == audit["evaluation_cuts"] == expected_cuts,
            "initial_model_hash": config["initial_model_sha256"] == audit["same_initial_model_sha256"] == expected_init,
            "final_epoch_rule": config["checkpoint_selection"] == "final epoch, no target labels or target metrics",
            "source_only_no_training_target": config["source_only_actual_training_target_cuts"] == [],
        }
        for method in METHODS:
            recorded = json.loads((pair / method / "metrics.json").read_text(encoding="utf-8"))
            recalculated = metrics_from_csv(pair / method / "predictions.csv", expected_cuts)
            checks[f"{method}_metrics"] = all(np.isclose(recorded[key], recalculated[key], rtol=1e-12, atol=1e-12)
                                                   for key in base.METRICS)
            checkpoint = torch.load(pair / method / "final.pth", map_location="cpu", weights_only=True)
            current, _ = base.build_model(42, torch.device("cpu"))
            checks[f"{method}_checkpoint_structure"] = checkpoint["epoch"] == EPOCHS and all(
                key in checkpoint["model"] and checkpoint["model"][key].shape == value.shape
                for key, value in current.state_dict().items())
            for name in ("final.pth", "metrics.json", "predictions.csv", "prediction.png", "config.json", "run.log"):
                path = pair / method / name
                preserved[str(path.resolve())] = base.file_hash(path)
        # The previous run did not persist source batch permutations or raw-signal content hashes.
        checks["historical_source_batch_order_recorded"] = False
        checks["historical_raw_content_hash_recorded"] = False
        findings.append({"direction": f"{source}_to_{target}", "checks": checks,
                         "all_observable_checks_pass": all(v for k, v in checks.items() if not k.startswith("historical_")),
                         "fully_provable_for_reuse": all(checks.values())})
    decision = {"decision": "rerun_seed42_in_new_root_preserve_old",
                "reason": "Old results lack recorded source batch order and raw signal content hashes; full equivalence cannot be established before reuse.",
                "old_root": str(args.old_root.resolve()), "findings": findings,
                "old_artifact_sha256": preserved,
                "feature_cache_manifest_sha256": base.file_hash(args.out_root / "feature_cache" / "manifest.json")}
    json_write(args.out_root / "prior_seed42_audit.json", decision)
    if not all(item["all_observable_checks_pass"] for item in findings):
        logging.warning("Some observable old seed=42 checks failed; all seed=42 runs will be independent reruns")
    else:
        logging.info("Old seed=42 observable checks pass, but full equivalence unprovable; rerunning in new root")
    return preserved


def ensure_preserved(preserved):
    changed = [path for path, digest in preserved.items() if base.file_hash(Path(path)) != digest]
    if changed:
        raise RuntimeError(f"Previous single-run artifacts changed: {changed}")


def train(method, x_source, y_source, x_target, source, target, seed, pair_dir, device,
          trend_lambda=0.0, trend_delta=0.0):
    if trend_lambda < 0 or trend_delta < 0 or not np.isfinite([trend_lambda, trend_delta]).all():
        raise ValueError("Trend weight and delta must be finite and nonnegative")
    model, init_hash = base.build_model(seed, device)
    source_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_source), torch.from_numpy(y_source), torch.arange(1, 316)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed))
    target_loader = None
    if method == "daregram":
        if x_target is None:
            raise ValueError("DARE-GRAM requires unlabeled target features")
        target_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_target), torch.arange(1, 316)),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
            generator=torch.Generator().manual_seed(seed + 1))
    elif x_target is not None:
        raise ValueError("Source-only must not receive target training features")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    method_dir = pair_dir / method
    handler = logging.FileHandler(method_dir / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    orders = []
    epoch_records = []
    target_usage = set()
    try:
        logging.info("Start %s %s->%s seed=%d init_sha256=%s", method, source, target, seed, init_hash)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            target_iter = iter(target_loader) if target_loader is not None else None
            source_order = []
            epoch_target = set()
            mse_sum = gram_sum = trend_source_sum = trend_target_sum = total_sum = 0.0
            weighted_trend_sum = trend_grad_sum = 0.0
            trend_grad_active_steps = 0
            source_pairs = target_pairs = 0
            tradeoff = 2 / (1 + math.exp(-10 * (epoch - 1) / (EPOCHS - 1))) - 1
            for xb, yb, source_cuts in source_loader:
                source_order.extend(source_cuts.tolist())
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                optimizer.zero_grad()
                feat_s = model.feature_extractor(xb)
                pred_s = model.regressor(feat_s)
                source_mse = F.mse_loss(pred_s, yb)
                loss = source_mse
                if target_iter is not None:
                    xt, target_cuts = next(target_iter)
                    epoch_target.update(target_cuts.tolist())
                    feat_t = model.feature_extractor(xt.to(device))
                    gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), feat_s, feat_t)
                    loss = loss + tradeoff * gram
                    gram_sum += gram.item()
                if trend_lambda > 0:
                    source_trend, n_source = trend_loss(pred_s, [source] * len(source_cuts),
                                                        source_cuts.tolist(), trend_delta)
                    source_pairs += n_source
                    parts = [source_trend] if n_source else []
                    trend_source_sum += source_trend.item()
                    if target_iter is not None:
                        pred_t = model.regressor(feat_t)
                        target_trend, n_target = trend_loss(pred_t,
                                                           [target] * len(target_cuts),
                                                           target_cuts.tolist(), trend_delta)
                        target_pairs += n_target
                        trend_target_sum += target_trend.item()
                        if n_target:
                            parts.append(target_trend)
                    if parts:
                        weighted_trend = trend_lambda * torch.stack(parts).mean()
                        trend_grad = torch.autograd.grad(weighted_trend, model.regressor[-1].weight,
                                                         retain_graph=True, allow_unused=True)[0]
                        grad_norm = 0.0 if trend_grad is None else float(trend_grad.norm().item())
                        trend_grad_sum += grad_norm
                        trend_grad_active_steps += int(grad_norm > 0)
                        weighted_trend_sum += weighted_trend.item()
                        loss = loss + weighted_trend
                total_sum += loss.item()
                loss.backward()
                optimizer.step()
                mse_sum += source_mse.item()
            if sorted(source_order) != base.ALL_CUTS:
                raise ValueError(f"Source order did not cover exactly 315 cuts: {source}->{target}, {seed}, epoch {epoch}")
            if target_iter is not None and epoch_target != set(base.ALL_CUTS):
                raise ValueError(f"Unlabeled target order did not cover 1..315: {source}->{target}, {seed}, epoch {epoch}")
            target_usage.update(epoch_target)
            order_hash = hashlib.sha256(np.asarray(source_order, dtype=np.int32).tobytes()).hexdigest()
            orders.append(order_hash)
            epoch_records.append({"epoch": epoch, "supervised_mse": mse_sum / len(source_loader),
                                  "domain_gram": gram_sum / len(source_loader),
                                  "domain_weight": tradeoff if target_iter is not None else 0.0,
                                  "source_trend_loss": trend_source_sum / len(source_loader),
                                  "target_trend_loss": trend_target_sum / len(source_loader),
                                  "source_valid_pairs": source_pairs, "target_valid_pairs": target_pairs,
                                  "trend_weight": trend_lambda,
                                  "weighted_trend_loss": weighted_trend_sum / len(source_loader),
                                  "trend_grad_norm_regressor_weight": trend_grad_sum / len(source_loader),
                                  "trend_grad_active_steps": trend_grad_active_steps,
                                  "total_loss": total_sum / len(source_loader)})
            logging.info("%s epoch=%d/%d source_MSE=%.6f gram=%.6f source_order_sha256=%s target_unique=%d",
                         method, epoch, EPOCHS, mse_sum / len(source_loader), gram_sum / len(source_loader),
                         order_hash, len(epoch_target))
            logging.info("%s trend epoch=%d source=%.6f pairs=%d target=%.6f pairs=%d weight=%.6f total=%.6f",
                         method, epoch, trend_source_sum / len(source_loader), source_pairs,
                         trend_target_sum / len(source_loader), target_pairs, trend_lambda,
                         total_sum / len(source_loader))
        ckpt = method_dir / "final.pth"
        torch.save({"model": model.state_dict(), "epoch": EPOCHS}, ckpt)
        pd.DataFrame(epoch_records).to_csv(method_dir / "epoch_losses.csv", index=False)
        logging.info("Saved final epoch checkpoint: %s", ckpt)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"initial_model_sha256": init_hash, "source_order_sha256_by_epoch": orders,
            "source_order_sha256_all_epochs": hashlib.sha256("".join(orders).encode()).hexdigest(),
            "actual_unlabeled_target_cuts": sorted(target_usage), "checkpoint_sha256": base.file_hash(ckpt),
            "epoch_losses": epoch_records}


def predict(method, x_target, cuts, seed, pair_dir, device):
    model, _ = base.build_model(seed, device)
    checkpoint = torch.load(pair_dir / method / "final.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, len(cuts), BATCH_SIZE):
            positions = np.asarray(cuts[offset:offset + BATCH_SIZE]) - 1
            xb = torch.from_numpy(x_target[positions]).to(device)
            values.extend(model.regressor(model.feature_extractor(xb)).cpu().numpy().reshape(-1).tolist())
    del model
    return np.asarray(values, dtype=np.float64)


def run_pair(args, source, target, seed, manifest):
    if source == target or source not in base.TOOLS or target not in base.TOOLS or seed not in SEEDS:
        raise ValueError("Invalid source, target, or seed")
    direction = args.out_root / f"{source}_to_{target}"
    direction.mkdir(parents=True, exist_ok=True)
    final_dir = direction / f"seed_{seed}"
    if final_dir.exists():
        verify_completed_pair(final_dir, source, target, seed, args, manifest)
        logging.info("Verified and reused completed %s->%s seed=%d", source, target, seed)
        return
    attempt = 1
    while (direction / f".seed_{seed}_work_{attempt:03d}").exists():
        attempt += 1
    pair_dir = direction / f".seed_{seed}_work_{attempt:03d}"
    pair_dir.mkdir()
    for method in METHODS:
        (pair_dir / method).mkdir()
    handler = logging.FileHandler(pair_dir / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        device = torch.device(args.device)
        cuts, cut_definition = base.evaluation_cuts(target, args.split_record)
        raw_source = cache_features(args, source)
        source_labels = base.wear_labels(args.raw_root, source)
        if len(source_labels) != 315:
            raise ValueError("Single-source labeled sample count must be 315")
        mean = raw_source.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw_source.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        x_source = ((raw_source - mean[None, :, None, None]) /
                    (std[None, :, None, None] + 1e-8)).astype(np.float32)
        del raw_source
        common = {"source": source, "target": target, "seed": seed, "epochs": EPOCHS,
                  "batch_size": BATCH_SIZE, "lr": LR, "device": str(device),
                  "raw_root": str(args.raw_root.resolve()), "split_record": str(args.split_record.resolve()),
                  "stft_code_sha256": manifest["stft_code_sha256"],
                  "source_feature_sha256": manifest["tools"][source]["feature_sha256"],
                  "target_feature_sha256": manifest["tools"][target]["feature_sha256"],
                  "source_wear_sha256": base.file_hash(args.raw_root / f"{source}_wear.csv"),
                  "source_count": 315, "source_cuts": base.ALL_CUTS,
                  "target_unlabeled_cuts": base.ALL_CUTS, "evaluation_cuts": cuts,
                  "evaluation_cut_definition": cut_definition,
                  "source_normalization_mean": mean.tolist(), "source_normalization_std": std.tolist(),
                  "input": "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128",
                  "backbone": "ResNet18", "regressor": "Linear(512, 1)", "optimizer": "Adam",
                  "scheduler": "fixed", "source_loss": "MSE",
                  "daregram_loss": "original inverse Gram loss with exp epoch tradeoff",
                  "checkpoint_selection": "final epoch", "protocol": "transductive UDA for DARE-GRAM",
                  "target_labels_training_reads": 0}
        if target == "c6":
            common["split_record_sha256"] = base.file_hash(args.split_record)
        logging.info("Start %s->%s seed=%d; evaluation cuts=%d (%s)", source, target, seed, len(cuts), cut_definition)
        source_trace = train("source_only", x_source, source_labels, None, source, target, seed, pair_dir, device)
        # Source-only training has finished before the target feature array is loaded.
        raw_target = cache_features(args, target)
        x_target = ((raw_target - mean[None, :, None, None]) /
                    (std[None, :, None, None] + 1e-8)).astype(np.float32)
        del raw_target
        dare_trace = train("daregram", x_source, source_labels, x_target, source, target, seed, pair_dir, device)
        if source_trace["initial_model_sha256"] != dare_trace["initial_model_sha256"]:
            raise ValueError("Paired methods did not start from identical model weights")
        if source_trace["source_order_sha256_by_epoch"] != dare_trace["source_order_sha256_by_epoch"]:
            raise ValueError("Paired methods did not use the same source batch sequence")
        if source_trace["actual_unlabeled_target_cuts"] != [] or dare_trace["actual_unlabeled_target_cuts"] != base.ALL_CUTS:
            raise ValueError("Incorrect target information use")
        predictions = {method: predict(method, x_target, cuts, seed, pair_dir, device) for method in METHODS}
        # Target wear is opened only after both final checkpoints and predictions are fixed.
        target_labels = base.wear_labels(args.raw_root, target, evaluation_dir=pair_dir)
        y_true = target_labels[np.asarray(cuts) - 1].astype(np.float64)
        results = {}
        for method in METHODS:
            folder = pair_dir / method
            pd.DataFrame({"cut_index": cuts, "true_vb": y_true, "pred_vb": predictions[method]}).to_csv(
                folder / "predictions.csv", index=False, float_format="%.17g")
            results[method] = metrics_from_csv(folder / "predictions.csv", cuts)
            json_write(folder / "metrics.json", results[method])
            fig, ax = base.plt.subplots(figsize=(10, 4.5))
            ax.plot(cuts, y_true, label="True VB")
            ax.plot(cuts, predictions[method], label="Predicted VB")
            ax.set(xlabel=f"{target.upper()} cut index", ylabel="VB", title=f"{source.upper()}->{target.upper()} seed {seed}: {method}")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(folder / "prediction.png", dpi=200)
            base.plt.close(fig)
            trace = source_trace if method == "source_only" else dare_trace
            json_write(folder / "config.json", dict(common, method=method,
                        initial_model_sha256=trace["initial_model_sha256"],
                        source_order_sha256_by_epoch=trace["source_order_sha256_by_epoch"],
                        source_order_sha256_all_epochs=trace["source_order_sha256_all_epochs"],
                        actual_unlabeled_target_cuts=trace["actual_unlabeled_target_cuts"],
                        target_training_information="none" if method == "source_only" else "all 315 unlabeled STFT inputs"))
        delta = {metric: results["daregram"][metric] - results["source_only"][metric] for metric in base.METRICS}
        audit = {"source_target_distinct": source != target, "source_count": 315,
                 "source_labeled_cuts": base.ALL_CUTS, "target_unlabeled_cuts": dare_trace["actual_unlabeled_target_cuts"],
                 "source_only_training_target_cuts": source_trace["actual_unlabeled_target_cuts"],
                 "evaluation_cuts": cuts, "target_label_reads_during_training": 0,
                 "target_label_first_read": "after both final checkpoints and predictions",
                 "same_initial_model_sha256": source_trace["initial_model_sha256"],
                 "same_source_order_sha256_all_epochs": source_trace["source_order_sha256_all_epochs"],
                 "paired_initialization_verified": True, "paired_source_order_verified": True,
                 "checkpoint_sha256": {"source_only": source_trace["checkpoint_sha256"],
                                       "daregram": dare_trace["checkpoint_sha256"]}}
        json_write(pair_dir / "config.json", common)
        json_write(pair_dir / "audit.json", audit)
        json_write(pair_dir / "comparison.json", {"source_only": results["source_only"],
                  "daregram": results["daregram"], "daregram_minus_source_only": delta})
        logging.info("Completed %s->%s seed=%d: source_only=%s daregram=%s", source, target, seed,
                     results["source_only"], results["daregram"])
        handler.flush()
        logging.getLogger().removeHandler(handler)
        handler.close()
        # The rename is confined to one direction directory and never replaces a prior result.
        if final_dir.exists() or pair_dir.resolve().parent != final_dir.resolve().parent:
            raise FileExistsError(f"Cannot finalize over an existing directory: {final_dir}")
        pair_dir.rename(final_dir)
        json_write(final_dir / "complete.json", {"source": source, "target": target, "seed": seed,
                   "source_only_MAE": results["source_only"]["MAE"], "daregram_MAE": results["daregram"]["MAE"]})
    finally:
        if handler in logging.getLogger().handlers:
            logging.getLogger().removeHandler(handler)
            handler.close()


def verify_completed_pair(folder, source, target, seed, args, manifest):
    if not (folder / "complete.json").is_file():
        raise ValueError(f"Existing result is incomplete; refusing overwrite: {folder}")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    cuts, _ = base.evaluation_cuts(target, args.split_record)
    expected = {"source": source, "target": target, "seed": seed, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "lr": LR, "device": args.device,
                "raw_root": str(args.raw_root.resolve()), "split_record": str(args.split_record.resolve()),
                "source_count": 315, "evaluation_cuts": cuts,
                "stft_code_sha256": manifest["stft_code_sha256"],
                "source_feature_sha256": manifest["tools"][source]["feature_sha256"],
                "target_feature_sha256": manifest["tools"][target]["feature_sha256"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed pair config differs from requested protocol: {folder}")
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    if not (audit["paired_initialization_verified"] and audit["paired_source_order_verified"] and
            audit["target_unlabeled_cuts"] == base.ALL_CUTS and audit["source_only_training_target_cuts"] == [] and
            audit["target_label_reads_during_training"] == 0):
        raise ValueError(f"Completed pair audit failed: {folder}")
    for method in METHODS:
        current = metrics_from_csv(folder / method / "predictions.csv", cuts)
        recorded = json.loads((folder / method / "metrics.json").read_text(encoding="utf-8"))
        if not all(np.isclose(current[key], recorded[key], rtol=1e-12, atol=1e-12) for key in base.METRICS):
            raise ValueError(f"Completed pair metrics disagree with predictions: {folder}/{method}")
        for name in ("config.json", "run.log", "final.pth", "prediction.png"):
            if not (folder / method / name).is_file():
                raise FileNotFoundError(folder / method / name)


def summarize(args, manifest):
    seed_rows = []
    incomplete = []
    for source, target in PAIRS:
        for seed in SEEDS:
            folder = args.out_root / f"{source}_to_{target}" / f"seed_{seed}"
            if not folder.exists():
                incomplete.append(f"{source}->{target} seed={seed}")
                continue
            verify_completed_pair(folder, source, target, seed, args, manifest)
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
            row = {"source": source, "target": target, "seed": seed,
                   "evaluation_count": len(config["evaluation_cuts"]),
                   "evaluation_cuts": ",".join(map(str, config["evaluation_cuts"]))}
            for method, prefix in (("source_only", "source_only"), ("daregram", "daregram"),
                                   ("daregram_minus_source_only", "delta")):
                row.update({f"{prefix}_{metric}": comparison[method][metric] for metric in base.METRICS})
            seed_rows.append(row)
    pd.DataFrame(seed_rows).to_csv(args.out_root / "seed_metrics.csv", index=False)
    summary_rows = []
    for source, target in PAIRS:
        rows = [row for row in seed_rows if row["source"] == source and row["target"] == target]
        if len(rows) != len(SEEDS):
            continue
        row = {"source": source, "target": target, "n": 5,
               "evaluation_count": rows[0]["evaluation_count"],
               "evaluation_cuts": rows[0]["evaluation_cuts"],
               "daregram_improved_MAE_count": sum(x["delta_MAE"] < 0 for x in rows),
               "daregram_improved_R2_count": sum(x["delta_R2"] > 0 for x in rows)}
        for metric in base.METRICS:
            for prefix in ("source_only", "daregram", "delta"):
                values = np.asarray([x[f"{prefix}_{metric}"] for x in rows], dtype=np.float64)
                row[f"{prefix}_{metric}_mean"] = float(values.mean())
                row[f"{prefix}_{metric}_std"] = float(values.std(ddof=1))
                if prefix == "delta":
                    margin = float(student_t.ppf(0.975, df=4) * values.std(ddof=1) / np.sqrt(5))
                    row[f"delta_{metric}_ci95_low"] = float(values.mean() - margin)
                    row[f"delta_{metric}_ci95_high"] = float(values.mean() + margin)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.out_root / "summary.csv", index=False)
    json_write(args.out_root / "completion.json", {"completed_pair_seed_count": len(seed_rows),
               "completed_training_count": len(seed_rows) * 2, "required_training_count": 60,
               "incomplete": incomplete})
    report = ["# PHM2010 five-seed paired single-source experiments", "",
              "Seeds: 42, 43, 44, 45, 46. Each seed independently initializes ResNet18 and Linear(512,1).",
              "Within each direction and seed, both methods have identical initial weights and source batch order (verified by SHA256).",
              "Only the source tool's 315 labels train the models. DARE-GRAM aligns with all 315 unlabeled target inputs, including evaluation inputs: **transductive UDA**.",
              "Target wear labels are first read after both final-epoch checkpoints and predictions are fixed.",
              "C6 evaluates cuts 95-315; C1/C4 evaluate cuts 1-315. Cross-target metrics use different test sets.", "",
              f"Completed {len(seed_rows)}/30 direction-seed pairs ({2*len(seed_rows)}/60 trainings).", "",
              "Prior seed=42 results were preserved and rerun in this directory because the old run did not save source order or raw content fingerprints; see prior_seed42_audit.json.", "",
              "## Method metrics and paired differences", "",
              "Each standard deviation uses ddof=1. The 95% interval is mean paired difference ± t(0.975,4)×SD/√5. MAPE differences are percentage points.", "",
              "| Direction | Metric | Source-only mean ± SD | DARE-GRAM mean ± SD | Paired Δ mean ± SD | Paired Δ 95% t interval |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        for metric in base.METRICS:
            report.append(f"| {row['source'].upper()}→{row['target'].upper()} | {metric} | "
                          f"{row[f'source_only_{metric}_mean']:.4f} ± {row[f'source_only_{metric}_std']:.4f} | "
                          f"{row[f'daregram_{metric}_mean']:.4f} ± {row[f'daregram_{metric}_std']:.4f} | "
                          f"{row[f'delta_{metric}_mean']:+.4f} ± {row[f'delta_{metric}_std']:.4f} | "
                          f"[{row[f'delta_{metric}_ci95_low']:+.4f}, {row[f'delta_{metric}_ci95_high']:+.4f}] |")
    report += ["", "## Improvement counts and stability", "",
               "| Direction | MAE improved | R² improved | Interpretation |", "|---|---:|---:|---|"]
    for row in summary_rows:
        mae_count = row["daregram_improved_MAE_count"]
        r2_count = row["daregram_improved_R2_count"]
        if mae_count == 5 and r2_count == 5:
            verdict = "Both metrics improved for all five seeds"
        elif mae_count == 0 and r2_count == 0:
            verdict = "Neither metric improved for any seed"
        else:
            verdict = "Mixed across seeds or metrics"
        report.append(f"| {row['source'].upper()}→{row['target'].upper()} | {mae_count}/5 | {r2_count}/5 | {verdict} |")
    report += ["", "## Four directions of interest", ""]
    focus = {(row["source"], row["target"]): row for row in summary_rows}
    for source, target in (("c1", "c6"), ("c4", "c1"), ("c6", "c1"), ("c6", "c4")):
        if (source, target) not in focus:
            continue
        row = focus[(source, target)]
        mae_count = row["daregram_improved_MAE_count"]
        r2_count = row["daregram_improved_R2_count"]
        stable_mae = mae_count == 5 and row["delta_MAE_ci95_high"] < 0
        stable_r2 = r2_count == 5 and row["delta_R2_ci95_low"] > 0
        report.append(
            f"- **{source.upper()}→{target.upper()}**: MAE improved in {mae_count}/5 seeds; "
            f"paired ΔMAE={row['delta_MAE_mean']:+.3f}, 95% CI "
            f"[{row['delta_MAE_ci95_low']:+.3f}, {row['delta_MAE_ci95_high']:+.3f}]—"
            f"{'consistent MAE improvement' if stable_mae else 'MAE improvement is not established across all seeds'}. "
            f"R² improved in {r2_count}/5 seeds; paired ΔR²={row['delta_R2_mean']:+.3f}, "
            f"95% CI [{row['delta_R2_ci95_low']:+.3f}, {row['delta_R2_ci95_high']:+.3f}]—"
            f"{'consistent R² improvement' if stable_r2 else 'R² improvement is not established across all seeds'}."
        )
    report += ["", "These intervals describe the five fixed seeds under the transductive protocol; they do not make metrics on different target evaluation sets directly comparable.", ""]
    if incomplete:
        report += ["## Incomplete tasks", "", *[f"- {item}" for item in incomplete], ""]
    (args.out_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return seed_rows, incomplete


def main():
    args = arguments()
    args.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_root / "batch.log", encoding="utf-8")])
    manifest = prepare_cache(args)
    prior_audit = args.out_root / "prior_seed42_audit.json"
    if prior_audit.exists():
        prior = json.loads(prior_audit.read_text(encoding="utf-8"))
        preserved = prior["old_artifact_sha256"]
        ensure_preserved(preserved)
    else:
        preserved = audit_prior_seed42(args, manifest)
    summarize(args, manifest)
    try:
        for source, target in PAIRS:
            for seed in SEEDS:
                run_pair(args, source, target, seed, manifest)
                summarize(args, manifest)
    finally:
        summarize(args, manifest)
        ensure_preserved(preserved)
    logging.info("All 60 trainings completed")


if __name__ == "__main__":
    main()
