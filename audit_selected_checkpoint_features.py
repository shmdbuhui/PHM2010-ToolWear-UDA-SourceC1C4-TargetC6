"""Read-only inference audit for one multi-window checkpoint (no labels or training)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import run_multi_window_ablation as experiment
import run_single_source_pairs as base


def sha256(path: Path) -> str:
    return base.file_hash(path)


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=experiment.DEFAULT_OUT)
    parser.add_argument("--direction", default="c1_to_c6")
    parser.add_argument("--method", default="daregram")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/selected_checkpoint_feature_audit_20260926"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=63)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    source, target = args.direction.split("_to_")
    folder = args.experiment_root / args.direction / f"seed_{args.seed}"
    config_path = folder / "config.json"
    checkpoint_path = folder / args.method / "final.pth"
    cache_dir = args.experiment_root / "feature_cache"
    manifest_path = cache_dir / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (config["source"], config["target"], config["seed"]) == (source, target, args.seed)
    assert manifest["stft_code_sha256"] == sha256(Path(experiment.sampling.__file__))
    assert manifest["shape"] == [315, 3, 6, 128, 128]
    assert sha256(checkpoint_path) == config["traces"][args.method]["checkpoint_sha256"]
    caches = {}
    for domain in (source, target):
        path = cache_dir / f"{domain}_stft.npy"
        assert sha256(path) == manifest["tools"][domain]["feature_sha256"]
        caches[domain] = experiment.cache_view(args.experiment_root, domain)
    mean = np.array(config["source_zscore_mean"], dtype=np.float32)
    std = np.array(config["source_zscore_std"], dtype=np.float32)
    fitted_mean, fitted_std = experiment.source_stats(caches[source])
    assert np.array_equal(mean, fitted_mean) and np.array_equal(std, fitted_std)
    device = torch.device(args.device)
    model, _ = base.build_model(args.seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    assert checkpoint["seed"] == args.seed and checkpoint["epoch"] == config["epochs"]
    model.load_state_dict(checkpoint["model"], strict=True)
    assert model.feature_extractor.backbone.conv1.in_channels == 6
    assert model.regressor[0].in_features == 512 and model.regressor[0].out_features == 1
    model.eval()
    stage_shapes = {}
    hooks = []
    def record_shape(stage):
        def hook(_module, _inputs, output):
            stage_shapes[stage] = list(output.shape)
        return hook
    for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"):
        module = getattr(model.feature_extractor.backbone, name)
        hooks.append(module.register_forward_hook(record_shape(name)))
    with torch.inference_mode():
        sample = torch.zeros((1, 6, 128, 128), device=device)
        assert list(model.feature_extractor(sample).shape) == [1, 512]
    for hook in hooks:
        hook.remove()
    args.out.mkdir(parents=True)
    id_cols = ["direction", "model_type", "seed", "domain", "tool_id", "cut_index", "window_index", "center_fraction"]
    feature_cols = [f"dim_{i:03d}" for i in range(512)]
    input_rows, feature_rows, head_rows, prediction_rows = [], [], [], []
    vectors: dict[str, list[np.ndarray]] = {source: [], target: []}
    reference_predictions = None
    if source != target and (folder / args.method / "predictions.csv").is_file():
        with (folder / args.method / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            reference_predictions = list(csv.DictReader(stream))
    max_reference_error = 0.0
    with torch.inference_mode():
        for domain in (source, target):
            data = caches[domain]
            for window in range(3):
                for start in range(0, 315, args.batch_size):
                    raw = np.asarray(data[start:start + args.batch_size, window])
                    normalized = np.asarray((raw - mean[None, :, None, None]) /
                                            std[None, :, None, None], dtype=np.float32)
                    features = model.feature_extractor(torch.from_numpy(normalized).to(device))
                    head_input = features  # the exact tensor passed to the regressor
                    predicted = model.regressor(head_input)
                    features_np = features.cpu().numpy()
                    head_np = head_input.cpu().numpy()
                    predicted_np = predicted.cpu().numpy().reshape(-1)
                    assert features_np.shape == (len(raw), 512)
                    assert np.array_equal(features_np, head_np)
                    assert np.isfinite(normalized).all() and np.isfinite(features_np).all()
                    vectors[domain].append(features_np)
                    for row in range(len(raw)):
                        cut = start + row + 1
                        identity = [args.direction, args.method, args.seed, "source" if domain == source else "target",
                                    domain, cut, window, config["window_fractions"][window]]
                        image = normalized[row].astype(np.float64)
                        for channel, name in enumerate(experiment.sampling.EXPECTED_INPUT_COLS):
                            values = image[channel]
                            input_rows.append([*identity, channel, name, float(values.mean()), float(values.std()),
                                               float(values.min()), float(values.max()),
                                               float(np.mean(np.abs(values) <= 1e-6))])
                        feature_rows.append([*identity, *map(float, features_np[row])])
                        head_rows.append([*identity, *map(float, head_np[row])])
                        prediction_rows.append([*identity, float(predicted_np[row])])
                        if domain == target and reference_predictions is not None:
                            assert int(reference_predictions[cut - 1]["cut_index"]) == cut
                            column = ("pred_w25", "pred_center", "pred_w75")[window]
                            max_reference_error = max(max_reference_error,
                                abs(float(predicted_np[row]) - float(reference_predictions[cut - 1][column])))
    assert max_reference_error < 1e-4, max_reference_error
    order = {source: 0, target: 1}
    input_rows.sort(key=lambda r: (order[r[4]], r[5], r[6], r[8]))
    feature_rows.sort(key=lambda r: (order[r[4]], r[5], r[6]))
    head_rows.sort(key=lambda r: (order[r[4]], r[5], r[6]))
    prediction_rows.sort(key=lambda r: (order[r[4]], r[5], r[6]))
    write_csv(args.out / "normalized_input_stats.csv", id_cols +
              ["channel_index", "channel", "mean", "std_population", "min", "max", "near_zero_ratio_abs_le_1e-6"], input_rows)
    per_cut_input_rows = []
    for domain in (source, target):
        for cut in range(1, 316):
            for channel in range(6):
                rows = [r for r in input_rows if r[4] == domain and r[5] == cut and r[8] == channel]
                assert [r[6] for r in rows] == [0, 1, 2]
                pooled_mean = float(np.mean([r[10] for r in rows]))
                pooled_std = float(np.sqrt(max(np.mean([r[11]**2 + r[10]**2 for r in rows]) - pooled_mean**2, 0)))
                per_cut_input_rows.append([args.direction, args.method, args.seed,
                                           "source" if domain == source else "target", domain, cut,
                                           channel, rows[0][9], pooled_mean, pooled_std,
                                           min(r[12] for r in rows), max(r[13] for r in rows),
                                           float(np.mean([r[14] for r in rows]))])
    write_csv(args.out / "normalized_input_stats_per_cut.csv", id_cols[:6] +
              ["channel_index", "channel", "mean", "std_population", "min", "max", "near_zero_ratio_abs_le_1e-6"],
              per_cut_input_rows)
    write_csv(args.out / "resnet18_512.csv", id_cols + feature_cols, feature_rows)
    write_csv(args.out / "regressor_input_512.csv", id_cols + feature_cols, head_rows)
    write_csv(args.out / "predictions_per_window.csv", id_cols + ["prediction_vb"], prediction_rows)
    cut_rows = []
    for domain in (source, target):
        for cut in range(1, 316):
            rows = [r for r in prediction_rows if r[4] == domain and r[5] == cut]
            assert [r[6] for r in rows] == [0, 1, 2]
            cut_rows.append([args.direction, args.method, args.seed,
                             "source" if domain == source else "target", domain, cut,
                             *[r[-1] for r in rows], float(np.mean([r[-1] for r in rows]))])
    write_csv(args.out / "predictions_per_cut.csv",
              id_cols[:6] + ["pred_w25", "pred_center", "pred_w75", "pred_mean"], cut_rows)
    stats_rows = []
    near_zero_counts = {}
    for domain in (source, target):
        matrix = np.concatenate(vectors[domain], axis=0).astype(np.float64)
        assert matrix.shape == (945, 512)
        sd = matrix.std(axis=0)
        near_zero_counts[domain] = int(np.sum(sd <= 1e-6))
        for index in range(512):
            column = matrix[:, index]
            stats_rows.append([args.direction, args.method, args.seed,
                               "source" if domain == source else "target", domain, index,
                               float(column.mean()), float(sd[index]),
                               float(np.mean(np.abs(column) <= 1e-6)),
                               float(column.min()), float(column.max()), len(column)])
    write_csv(args.out / "dimension_domain_stats.csv",
              id_cols[:5] + ["dimension", "mean", "std_population", "near_zero_ratio_abs_le_1e-6",
                             "min", "max", "n_cut_windows"], stats_rows)
    report = {
        "direction": args.direction, "model_type": args.method, "seed": args.seed,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
        "configuration": str(config_path.resolve()), "manifest": str(manifest_path.resolve()),
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "cache_sha256": {d: manifest["tools"][d]["feature_sha256"] for d in (source, target)},
        "counts": {"domains": 2, "cuts_per_domain": 315, "windows_per_cut": 3,
                   "input_channel_rows": len(input_rows), "feature_rows": len(feature_rows)},
        "stage_output_shapes_for_batch_1": stage_shapes,
        "near_zero_definition": {"magnitude": "abs(value) <= 1e-6",
                                 "near_constant_dimension": "population std across 945 windows <= 1e-6"},
        "near_constant_dimension_counts": near_zero_counts,
        "resnet_output_equals_regressor_input_bitwise": True,
        "max_abs_prediction_difference_from_original_target_csv": max_reference_error,
        "frequency_range_hz": [0, 25000], "stft_raw_shape_per_channel": [129, 121],
        "no_training_or_checkpoint_write": True,
    }
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
