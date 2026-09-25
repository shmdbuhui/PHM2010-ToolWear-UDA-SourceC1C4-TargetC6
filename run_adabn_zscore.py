"""Read-only source-only checkpoint AdaBN evaluation for the five-seed Z-score main run.

Only BN running buffers are recalibrated. No gradient, parameter update, training,
input normalization refit, or target label access occurs during calibration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from networks.resnet import ResNet18


TOOLS = ("c1", "c4", "c6")
SEEDS = range(42, 47)
METRICS = ("R2", "MAE", "RMSE", "MAPE_percent")
BATCH_SIZE = 63


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = y - pred
    return {"R2": float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2)),
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE_percent": float(100 * np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-8)))}


def prior_evaluation(folder: Path, expected_cuts: list[int]) -> tuple[np.ndarray, dict]:
    with (folder / "predictions.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [int(row["cut_index"]) for row in rows] != expected_cuts:
        raise ValueError(f"Existing evaluation cuts disagree: {folder}")
    y = np.asarray([float(row["true_vb"]) for row in rows])
    pred = np.asarray([float(row["pred_vb"]) for row in rows])
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError(f"Nonfinite existing predictions: {folder}")
    derived, recorded = metrics(y, pred), read_json(folder / "metrics.json")
    if any(not np.isclose(derived[k], recorded[k], rtol=0, atol=1e-10) for k in METRICS):
        raise ValueError(f"Existing metrics disagree with predictions: {folder}")
    return y, recorded


def build_model(state: dict, device: torch.device) -> nn.Module:
    model = nn.Module()
    model.feature_extractor = ResNet18()
    model.regressor = nn.Sequential(nn.Linear(512, 1))
    model.load_state_dict(state, strict=True)
    return model.to(device)


def predict(model: nn.Module, x: np.ndarray, device: torch.device,
            capture_conv1: bool = False) -> tuple[np.ndarray, torch.Tensor | None]:
    captured = []
    hook = None
    if capture_conv1:
        def capture(_module, inputs):
            if not captured:
                captured.append(inputs[0].detach().cpu().clone())
        hook = model.feature_extractor.backbone.conv1.register_forward_pre_hook(capture)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, 315, BATCH_SIZE):
            xb = torch.from_numpy(x[start:start + BATCH_SIZE]).to(device)
            output.append(model.regressor(model.feature_extractor(xb)).cpu().numpy().reshape(-1))
    if hook is not None:
        hook.remove()
    result = np.concatenate(output).astype(np.float64)
    if result.shape != (315,) or not np.isfinite(result).all():
        raise ValueError("Invalid full-lifecycle prediction")
    return result, captured[0] if captured else None


def run_one(root: Path, out: Path, cache: dict[str, np.ndarray], source: str,
            target: str, seed: int, device: torch.device) -> list[dict]:
    pair = root / f"{source}_to_{target}" / f"seed_{seed}"
    source_folder, dare_folder = pair / "source_only", pair / "daregram"
    config = read_json(source_folder / "config.json")
    dare_config = read_json(dare_folder / "config.json")
    expected_cuts = list(range(95, 316)) if target == "c6" else list(range(1, 316))
    if (config["source"], config["target"], config["seed"], config["method"],
        config["evaluation_cuts"], config["checkpoint_selection"]) != (
            source, target, seed, "source_only", expected_cuts, "final epoch"):
        raise ValueError(f"Wrong source-only experiment: {source_folder}")
    if (dare_config["source"], dare_config["target"], dare_config["seed"],
        dare_config["method"], dare_config["evaluation_cuts"]) != (
            source, target, seed, "daregram", expected_cuts):
        raise ValueError(f"Wrong DARE-GRAM comparison: {dare_folder}")
    if (config["source_normalization_mean"] != dare_config["source_normalization_mean"] or
        config["source_normalization_std"] != dare_config["source_normalization_std"]):
        raise ValueError("Paired normalization differs")
    if config["source_feature_sha256"] != sha(root / "feature_cache" / f"{source}_stft.npy"):
        raise ValueError("Source cache hash changed")
    if config["target_feature_sha256"] != sha(root / "feature_cache" / f"{target}_stft.npy"):
        raise ValueError("Target cache hash changed")
    mean = np.asarray(config["source_normalization_mean"], dtype=np.float32)
    std = np.asarray(config["source_normalization_std"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,) or not (std > 0).all():
        raise ValueError("Invalid source normalization")
    x = ((cache[target] - mean[None, :, None, None]) /
         (std[None, :, None, None] + 1e-8)).astype(np.float32)
    if x.shape != (315, 6, 128, 128) or not np.isfinite(x).all():
        raise ValueError("Invalid target input")
    checkpoint_path = source_folder / "final.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != config["epochs"] or config["epochs"] != 50:
        raise ValueError("Checkpoint is not final epoch 50")
    original = build_model(checkpoint["model"], device)
    adapted = copy.deepcopy(original)
    del checkpoint
    original.eval()
    original_predictions, original_conv_input = predict(original, x, device, capture_conv1=True)
    before_state = {k: v.detach().cpu().clone() for k, v in original.state_dict().items()}
    for parameter in adapted.parameters():
        parameter.requires_grad_(False)
    adapted.eval()
    bn_layers = [(name, module) for name, module in adapted.named_modules()
                 if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers or any(not module.track_running_stats for _, module in bn_layers):
        raise ValueError("Expected running-stat BatchNorm layers")
    bn_buffer_names = {f"{name}.{buffer}" for name, _ in bn_layers
                       for buffer in ("running_mean", "running_var", "num_batches_tracked")}
    for _, module in bn_layers:
        module.reset_running_stats()
        module.momentum = None  # cumulative average over five equal-sized batches
        module.train()
    if any(module.training for module in adapted.modules()
           if module is not adapted and not isinstance(module, nn.modules.batchnorm._BatchNorm)):
        raise ValueError("A non-BN layer entered training mode")
    batches = 0
    with torch.no_grad():
        for start in range(0, 315, BATCH_SIZE):
            xb = torch.from_numpy(x[start:start + BATCH_SIZE]).to(device)
            adapted.feature_extractor(xb)
            batches += 1
    adapted.eval()
    after_state = {k: v.detach().cpu() for k, v in adapted.state_dict().items()}
    non_bn_equal = all(torch.equal(before_state[k], after_state[k])
                       for k in before_state if k not in bn_buffer_names)
    if not non_bn_equal:
        raise ValueError("Non-BN parameter or buffer changed")
    changed_layers = []
    for name, module in bn_layers:
        mean_changed = not torch.equal(before_state[f"{name}.running_mean"], after_state[f"{name}.running_mean"])
        var_changed = not torch.equal(before_state[f"{name}.running_var"], after_state[f"{name}.running_var"])
        tracked = int(after_state[f"{name}.num_batches_tracked"])
        if tracked != batches or not (mean_changed or var_changed):
            raise ValueError(f"BN statistics did not update as required: {name}")
        changed_layers.append({"name": name, "running_mean_changed": mean_changed,
                               "running_var_changed": var_changed, "num_batches_tracked": tracked})
    adapted_predictions, adapted_conv_input = predict(adapted, x, device, capture_conv1=True)
    conv_input_equal = torch.equal(original_conv_input, adapted_conv_input)
    if not conv_input_equal:
        raise ValueError("Conv1 input changed between original and AdaBN")
    # Evaluation labels are read only after BN calibration and predictions are fixed.
    y_source, source_metrics = prior_evaluation(source_folder, expected_cuts)
    y_dare, dare_metrics = prior_evaluation(dare_folder, expected_cuts)
    if not np.array_equal(y_source, y_dare):
        raise ValueError("Existing methods used different target labels")
    indices = np.asarray(expected_cuts) - 1
    baseline_reproduced = metrics(y_source, original_predictions[indices])
    if abs(baseline_reproduced["RMSE"] - source_metrics["RMSE"]) > 0.02:
        raise ValueError("Existing source-only checkpoint does not reproduce its RMSE")
    adapted_metrics = metrics(y_source, adapted_predictions[indices])
    if not all(np.isfinite(list(adapted_metrics.values()))):
        raise ValueError("Nonfinite AdaBN metric")
    folder = out / f"{source}_to_{target}" / f"seed_{seed}"
    folder.mkdir(parents=True)
    with (folder / "predictions_all_cuts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cut_index", "pred_vb", "in_existing_evaluation"])
        writer.writerows((cut, format(float(pred), ".17g"), int(cut in expected_cuts))
                         for cut, pred in enumerate(adapted_predictions, 1))
    torch.save({k: after_state[k].clone() for k in sorted(bn_buffer_names)}, folder / "adabn_bn_buffers.pth")
    audit = {"source": source, "target": target, "seed": seed,
             "source_checkpoint": str(checkpoint_path.resolve()),
             "source_checkpoint_sha256": sha(checkpoint_path),
             "daregram_checkpoint": str((dare_folder / "final.pth").resolve()),
             "daregram_checkpoint_sha256": sha(dare_folder / "final.pth"),
             "input_cache": str((root / "feature_cache" / f"{target}_stft.npy").resolve()),
             "normalization_source": source, "normalization_mean": mean.tolist(),
             "normalization_std": std.tolist(), "input_shape": list(x.shape),
             "bn_scheme": {"model_initial_state": "eval", "bn_only_state": "train",
                           "reset_running_stats": True, "momentum": None,
                           "momentum_meaning": "cumulative average across batches",
                           "batch_size": BATCH_SIZE, "batch_order": "cut 1..315 ascending",
                           "passes": 1, "drop_last": False,
                           "last_batch_size": 315 % BATCH_SIZE or BATCH_SIZE,
                           "num_batches": batches, "gradients": False,
                           "post_calibration_state": "eval"},
             "bn_layers": changed_layers, "all_non_bn_state_equal": non_bn_equal,
             "all_parameters_frozen_during_calibration": True,
             "conv1_input_equal_before_after": conv_input_equal,
             "original_reproduced_rmse": baseline_reproduced["RMSE"],
             "existing_source_only_rmse": source_metrics["RMSE"],
             "evaluation_cuts": expected_cuts,
             "target_labels_first_read": "after BN calibration and all-cut predictions",
             "predictions_finite": True, "metrics_finite": True}
    save_json(folder / "audit.json", audit)
    save_json(folder / "metrics.json", adapted_metrics)
    rows = []
    for method, values in (("source_only", source_metrics), ("source_only_adabn", adapted_metrics),
                           ("daregram", dare_metrics)):
        rows.append({"source": source, "target": target, "seed": seed,
                     "method": method, "evaluation_count": len(expected_cuts), **values,
                     "delta_rmse_adabn_minus_source_only": adapted_metrics["RMSE"] - source_metrics["RMSE"]})
    del original, adapted, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/adabn_zscore_20260925"))
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root, out = args.experiment_root.resolve(), args.out_root.resolve()
    if out.exists() or out == root or root in out.parents:
        parser.error("Output must be a new directory outside the input experiment root")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    cache = {}
    for tool in TOOLS:
        cache[tool] = np.load(root / "feature_cache" / f"{tool}_stft.npy",
                              mmap_mode="r", allow_pickle=False)
        if cache[tool].shape != (315, 6, 128, 128) or cache[tool].dtype != np.float32:
            raise ValueError(f"Invalid target cache: {tool}")
    out.mkdir(parents=True)
    rows = []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                rows.extend(run_one(root, out, cache, source, target, seed, device))
                print(f"AdaBN {source}->{target} seed {seed} complete", flush=True)
    with (out / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for method in ("source_only", "source_only_adabn", "daregram"):
                sample = [r for r in rows if r["source"] == source and r["target"] == target and r["method"] == method]
                row = {"source": source, "target": target, "method": method, "n_seeds": len(sample),
                       "evaluation_count": sample[0]["evaluation_count"]}
                for metric in METRICS:
                    values = np.asarray([r[metric] for r in sample])
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_sd"] = float(values.std(ddof=1))
                deltas = np.asarray([r["delta_rmse_adabn_minus_source_only"] for r in sample])
                row["delta_rmse_adabn_minus_source_only_mean"] = float(deltas.mean())
                row["delta_rmse_adabn_minus_source_only_sd"] = float(deltas.std(ddof=1))
                summary.append(row)
    with (out / "direction_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Completed {len(rows) // 3} AdaBN checkpoints: {out}")


if __name__ == "__main__":
    main()
