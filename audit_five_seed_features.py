"""Read-only feature audit for the six-direction, five-seed paired experiment.

The training code uses feature_cache/*.npy.  The c1/c4 NPZ samples are checked
against that cache; c6 has no NPZ.  NPZ labels are never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from networks.resnet import ResNet18


TOOLS = ("c1", "c4", "c6")
METHODS = ("source_only", "daregram")
SEEDS = range(42, 47)
CUTS = list(range(1, 316))
FEATURE_COLUMNS = [f"feature_{i:03d}" for i in range(512)]
NEAR_ZERO_VARIANCE = 1e-12  # absolute population variance across 315 cuts
PREPROCESS = {
    "raw_window": "center 4096 samples, [floor(T/2)-2048, floor(T/2)+2048)",
    "channels": ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz"],
    "stft": {"fs_hz": 50000.0, "window": "hann", "nperseg": 256,
             "noverlap": 224, "hop": 32, "detrend": False,
             "boundary": None, "padded": False, "raw_bins": [129, 121]},
    "transform": "log1p(abs(STFT)); scipy.ndimage.zoom(order=1) to 128x128",
    "fmax_argument_hz": 10000.0,
    "frequency_crop_executed": False,
    "frequency_crop_reason": "data_sampling._stft_crop_and_resize has its fmax branch commented out",
    "input_shape_per_cut": [6, 128, 128],
    "normalization": "float32((cache - float32(source mean)) / (float32(source std) + 1e-8)); source std = float32(cache.std(axis=(0,2,3)) + 1e-8)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # Exactly the expression used in run_five_seed_pairs.run_pair.
    return ((raw - mean[None, :, None, None]) /
            (std[None, :, None, None] + 1e-8)).astype(np.float32)


def write_features(path: Path, tool: str, vectors: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tool_id", "cut_index", *FEATURE_COLUMNS])
        for cut, vector in enumerate(vectors, 1):
            writer.writerow([tool, cut, *(format(float(v), ".9g") for v in vector)])


def feature_stats(vectors: np.ndarray) -> dict:
    finite = np.isfinite(vectors)
    variance = np.var(vectors.astype(np.float64), axis=0) if finite.all() else None
    return {
        "shape": list(vectors.shape), "nan_count": int(np.isnan(vectors).sum()),
        "inf_count": int(np.isinf(vectors).sum()),
        "variance_definition": "population variance across all 315 ordered cuts, float64 accumulation",
        "variance_by_dimension": variance.tolist() if variance is not None else None,
        "near_zero_variance_threshold": NEAR_ZERO_VARIANCE,
        "near_zero_variance_count": int((variance <= NEAR_ZERO_VARIANCE).sum()) if variance is not None else None,
    }


def vector_summary(vector: np.ndarray) -> dict:
    top = np.argsort(np.abs(vector))[-5:][::-1]
    return {"mean": float(np.mean(vector)), "std": float(np.std(vector)),
            "min": float(np.min(vector)), "max": float(np.max(vector)),
            "l2_norm": float(np.linalg.norm(vector)),
            "top_abs_dimensions": [int(i) for i in top],
            "top_abs_values": [float(vector[i]) for i in top]}


def bn_summary(state: dict) -> dict:
    names = sorted(k for k in state if k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    digest = hashlib.sha256()
    details = {}
    for name in names:
        array = state[name].detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
        details[name] = {"shape": list(array.shape), "sha256": sha256_array(array),
                         "mean": float(np.mean(array)), "min": float(np.min(array)),
                         "max": float(np.max(array))}
    return {"sha256": digest.hexdigest(), "buffer_count": len(names), "buffers": details}


def first_difference(a: np.ndarray, b: np.ndarray) -> dict | None:
    for cut in CUTS:
        if not np.array_equal(a[cut - 1], b[cut - 1]):
            index = np.argwhere(a[cut - 1] != b[cut - 1])[0].tolist()
            return {"cut_index": cut, "input_coordinate_channel_frequency_time": index,
                    "source_only_value": float(a[cut - 1][tuple(index)]),
                    "daregram_value": float(b[cut - 1][tuple(index)])}
    return None


def plot_stft(path: Path, raw_cut: np.ndarray, tool: str, cut: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for channel, ax in enumerate(axes.flat):
        ax.imshow(raw_cut[channel], aspect="auto", origin="lower", cmap="magma",
                  extent=[0, 4096, 0, 25000])
        ax.set(title=PREPROCESS["channels"][channel], xlabel="sample in 4096 window",
               ylabel="frequency (Hz)")
    fig.suptitle(f"{tool.upper()} cut {cut}: cached six-channel log1p STFT, before source normalization")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(output: Path) -> None:
    """Summarize the completed machine-readable audit without reading labels."""
    run = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    with (output / "audit_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    norms = {}
    for row in rows:
        norms.setdefault(row["normalization_source"],
                         (json.loads(row["normalization_mean"]), json.loads(row["normalization_std"])))
    nonfinite = sum(int(row[key]) for row in rows for key in
                    ("source_nan_count", "source_inf_count", "target_nan_count", "target_inf_count"))
    near_zero = sum(int(row[key]) for row in rows for key in
                    ("source_near_zero_variance_count", "target_near_zero_variance_count"))
    lines = ["# PHM2010 five-seed feature audit", "",
             f"Audited {run['experiments_audited']} of {run['expected_experiments']} method experiments "
             f"across {run['paired_comparisons']} direction × seed pairs.",
             f"Paired model inputs identical: {run['all_paired_inputs_identical']}.",
             f"Missing configuration/checkpoint entries: {len(run['missing_experiments'])}.",
             f"Feature NaN/Inf values: {nonfinite}; near-zero variance dimensions across records: {near_zero} "
             f"(population variance ≤ {NEAR_ZERO_VARIANCE:g}).", "",
             "The training script used `feature_cache/*.npy`; c1/c4 NPZ `samples` were checked "
             "elementwise against those caches. No c6 NPZ exists. NPZ `labels` were never opened.",
             "Each target input uses the named **source tool's** six-channel mean and standard deviation. "
             "A target tool can therefore have different standardized inputs in experiments with different sources.",
             "The STFT frequency crop was not executed: its code branch is commented out. "
             "The cached image includes the full 0–25 kHz STFT range, resized to 128×128.", "",
             "## Source normalization (Fx, Fy, Fz, Vx, Vy, Vz)", ""]
    for source, (mean, std) in sorted(norms.items()):
        lines.extend([f"- `{source}` mean: `{json.dumps(mean)}`",
                      f"  standard deviation: `{json.dumps(std)}`"])
    lines.extend(["", "## Files", "",
                  "- `file_index.json`: paths for every direction × seed × method, including both feature CSVs, evaluation rows, audit, configuration, and checkpoint.",
                  "- `audit_summary.csv`: one record per experiment with preprocessing, input/checkpoint/BatchNorm digests, and all 512 source and target variances.",
                  "- `experiments/<source>_to_<target>/seed_<seed>/<method>/source_features.csv` and `target_features.csv`: 315 ordered rows, each with `tool_id`, `cut_index`, and `feature_000`–`feature_511`.",
                  "- `experiments/.../target_evaluation_rows.csv`: evaluation cut and zero-based row in `target_features.csv`; C6 uses 95–315, C1/C4 use 1–315.",
                  "- `experiments/.../audit.json`: full BatchNorm running-buffer hashes and descriptive statistics, plus feature variances.",
                  "- `paired_input_comparison.json`: bitwise input comparison for each direction × seed.",
                  "- `selected_target_vectors.csv`: early/middle/late target cut vector summaries and feature row links. These are unnamed ResNet dimensions, not handcrafted signal features.",
                  "- `stft_examples/*.png`: nine six-channel cached STFT images before source normalization.",
                  "- `missing_experiments.json`: missing configuration/checkpoint list (empty in this run).", "",
                  "Selected target cuts: C1/C4 = 1, 158, 315; C6 evaluation suffix = 95, 205, 315.",
                  "No model training or checkpoint modification occurred. Target labels were not read.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--npz-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/five_seed_feature_audit_20260925"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=63)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("Invalid batch size or CUDA unavailable")
    root = args.experiment_root.resolve()
    output = args.out_root.resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new --out-root: {output}")
    if root in output.parents or output == root:
        parser.error("Output must be outside the original experiment root")
    output.mkdir(parents=True)
    (output / "experiments").mkdir()
    (output / "stft_examples").mkdir()
    manifest_path = root / "feature_cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["stft_code_sha256"] != sha256_file(Path(__file__).parent / "data_sampling.py"):
        raise ValueError("Current STFT implementation differs from training manifest")
    raw = {}
    input_info = {}
    for tool in TOOLS:
        path = root / "feature_cache" / f"{tool}_stft.npy"
        if sha256_file(path) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"Training cache hash mismatch: {tool}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (315, 6, 128, 128) or value.dtype != np.float32:
            raise ValueError(f"Invalid cached input: {tool} {value.shape} {value.dtype}")
        raw[tool] = value
        npz_checks = {}
        for prefix in ("train", "target", "unlabeled"):
            npz = args.npz_root / f"{prefix}_{tool}.npz"
            if npz.is_file():
                with np.load(npz, allow_pickle=False) as archive:
                    # Deliberately never access archive['labels'].
                    npz_checks[prefix] = bool(np.array_equal(archive["samples"], value))
        input_info[tool] = {"training_cache": str(path.resolve()), "cache_sha256": sha256_file(path),
                            "npz_samples_equal_cache": npz_checks,
                            "npz_present": list(npz_checks)}
        print(f"Validated {tool} input cache; NPZ equality: {npz_checks}", flush=True)
    selected_cuts = {"c1": [1, 158, 315], "c4": [1, 158, 315], "c6": [95, 205, 315]}
    for tool, cuts in selected_cuts.items():
        for cut in cuts:
            plot_stft(output / "stft_examples" / f"{tool}_cut_{cut:03d}.png", raw[tool][cut - 1], tool, cut)
    rows, index, missing, comparisons, selected = [], [], [], [], []
    torch.set_grad_enabled(False)
    for source in TOOLS:
        # Compute stats exactly as the training script did, using source cache only.
        fitted_mean = raw[source].mean(axis=(0, 2, 3)).astype(np.float32)
        fitted_std = (raw[source].std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                pair = root / f"{source}_to_{target}" / f"seed_{seed}"
                configurations = {}
                for method in METHODS:
                    config_path = pair / method / "config.json"
                    checkpoint = pair / method / "final.pth"
                    if not config_path.is_file() or not checkpoint.is_file():
                        missing.append({"source": source, "target": target, "seed": seed,
                                        "method": method, "missing": [name for name, path in
                                        (("config", config_path), ("checkpoint", checkpoint)) if not path.is_file()]})
                        continue
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    if (config["source"], config["target"], config["seed"], config["method"]) != (source, target, seed, method):
                        raise ValueError(f"Experiment identity mismatch: {config_path}")
                    for label, tool in (("source", source), ("target", target)):
                        if config[f"{label}_feature_sha256"] != input_info[tool]["cache_sha256"]:
                            raise ValueError(f"Training input hash mismatch: {config_path}, {label}")
                    mean = np.asarray(config["source_normalization_mean"], dtype=np.float32)
                    std = np.asarray(config["source_normalization_std"], dtype=np.float32)
                    if not np.array_equal(mean, fitted_mean) or not np.array_equal(std, fitted_std):
                        raise ValueError(f"Source normalization does not match cache: {config_path}")
                    if config["input"] != "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128":
                        raise ValueError(f"Unexpected preprocessing declaration: {config_path}")
                    if config["stft_code_sha256"] != manifest["stft_code_sha256"]:
                        raise ValueError(f"STFT code hash mismatch: {config_path}")
                    configurations[method] = config
                if len(configurations) != 2:
                    continue
                # Rebuild both full model inputs independently from the method's recorded statistics.
                inputs = {}
                for method in METHODS:
                    config = configurations[method]
                    mean = np.asarray(config["source_normalization_mean"], dtype=np.float32)
                    std = np.asarray(config["source_normalization_std"], dtype=np.float32)
                    inputs[method] = {tool: normalized(raw[tool], mean, std) for tool in (source, target)}
                difference = {tool: first_difference(inputs["source_only"][tool], inputs["daregram"][tool])
                              for tool in (source, target)}
                same = all(v is None for v in difference.values())
                comparisons.append({"source": source, "target": target, "seed": seed,
                                    "inputs_identical": same, "first_difference": difference,
                                    "step_if_different": "source normalization" if not same else None,
                                    "normalization_source": source})
                for method in METHODS:
                    config = configurations[method]
                    checkpoint_path = pair / method / "final.pth"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    if checkpoint.get("epoch") != config["epochs"]:
                        raise ValueError(f"Checkpoint is not final epoch: {checkpoint_path}")
                    model = ResNet18().to(args.device)
                    state = {key.removeprefix("feature_extractor."): value for key, value in checkpoint["model"].items()
                             if key.startswith("feature_extractor.")}
                    model.load_state_dict(state, strict=True)
                    model.eval()
                    experiment_dir = output / "experiments" / f"{source}_to_{target}" / f"seed_{seed}" / method
                    experiment_dir.mkdir(parents=True)
                    audit = {"source": source, "target": target, "seed": seed, "method": method,
                             "config_path": str((pair / method / "config.json").resolve()),
                             "checkpoint_path": str(checkpoint_path.resolve()),
                             "checkpoint_sha256": sha256_file(checkpoint_path),
                             "preprocessing": PREPROCESS, "normalization_source": source,
                             "source_normalization_mean": config["source_normalization_mean"],
                             "source_normalization_std": config["source_normalization_std"],
                             "training_input": input_info, "bn_running_statistics": bn_summary(state),
                             "model_input_identical_between_methods": same,
                             "input_sha256": {tool: sha256_array(inputs[method][tool]) for tool in (source, target)},
                             "input_shape": {tool: list(inputs[method][tool].shape) for tool in (source, target)},
                             "feature_stats": {}}
                    paths = {}
                    for role, tool in (("source", source), ("target", target)):
                        data = inputs[method][tool]
                        chunks = []
                        for offset in range(0, 315, args.batch_size):
                            xb = torch.from_numpy(data[offset:offset + args.batch_size]).to(args.device)
                            with torch.inference_mode():
                                chunks.append(model(xb).cpu().numpy())
                        vectors = np.concatenate(chunks, axis=0)
                        if vectors.shape != (315, 512):
                            raise ValueError(f"Invalid feature shape: {vectors.shape}")
                        path = experiment_dir / f"{role}_features.csv"
                        write_features(path, tool, vectors)
                        paths[role] = str(path.resolve())
                        audit["feature_stats"][role] = feature_stats(vectors)
                        audit["feature_stats"][role]["file_sha256"] = sha256_file(path)
                        if role == "target":
                            for stage, cut in zip(("early", "middle", "late"), selected_cuts[target]):
                                selected.append({"source": source, "target": target, "seed": seed,
                                                 "method": method, "stage": stage, "cut_index": cut,
                                                 "normalization_source": source,
                                                 "stft_image": str((output / "stft_examples" / f"{target}_cut_{cut:03d}.png").resolve()),
                                                 "feature_file": str(path.resolve()), "feature_row_index_zero_based": cut - 1,
                                                 **vector_summary(vectors[cut - 1])})
                    evaluation = config["evaluation_cuts"]
                    if evaluation != (list(range(95, 316)) if target == "c6" else CUTS):
                        raise ValueError(f"Unexpected evaluation cut selection: {pair}")
                    eval_path = experiment_dir / "target_evaluation_rows.csv"
                    with eval_path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["tool_id", "cut_index", "feature_row_index_zero_based"])
                        writer.writerows((target, cut, cut - 1) for cut in evaluation)
                    audit["evaluation_cuts"] = evaluation
                    audit["evaluation_rows_file"] = str(eval_path.resolve())
                    detail_path = experiment_dir / "audit.json"
                    save_json(detail_path, audit)
                    row = {"source": source, "target": target, "seed": seed, "method": method,
                           "normalization_source": source, "raw_window": PREPROCESS["raw_window"],
                           "stft_parameters": json.dumps(PREPROCESS["stft"]),
                           "frequency_crop_executed": False,
                           "input_shape": json.dumps(audit["input_shape"]),
                           "normalization_mean": json.dumps(config["source_normalization_mean"]),
                           "normalization_std": json.dumps(config["source_normalization_std"]),
                           "input_sha256": json.dumps(audit["input_sha256"]),
                           "checkpoint_sha256": audit["checkpoint_sha256"],
                           "bn_running_statistics_sha256": audit["bn_running_statistics"]["sha256"],
                           "source_feature_shape": json.dumps(audit["feature_stats"]["source"]["shape"]),
                           "target_feature_shape": json.dumps(audit["feature_stats"]["target"]["shape"]),
                           "source_nan_count": audit["feature_stats"]["source"]["nan_count"],
                           "source_inf_count": audit["feature_stats"]["source"]["inf_count"],
                           "target_nan_count": audit["feature_stats"]["target"]["nan_count"],
                           "target_inf_count": audit["feature_stats"]["target"]["inf_count"],
                           "source_near_zero_variance_count": audit["feature_stats"]["source"]["near_zero_variance_count"],
                           "target_near_zero_variance_count": audit["feature_stats"]["target"]["near_zero_variance_count"],
                           "source_variance_by_dimension": json.dumps(audit["feature_stats"]["source"]["variance_by_dimension"]),
                           "target_variance_by_dimension": json.dumps(audit["feature_stats"]["target"]["variance_by_dimension"]),
                           "paired_input_identical": same}
                    rows.append(row)
                    index.append({"source": source, "target": target, "seed": seed, "method": method,
                                  "normalization_source": source, "source_features": paths["source"],
                                  "target_features": paths["target"],
                                  "target_evaluation_rows": str(eval_path.resolve()),
                                  "audit": str(detail_path.resolve()), "checkpoint": str(checkpoint_path.resolve()),
                                  "config": str((pair / method / "config.json").resolve())})
                    del model
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                del inputs
                print(f"Audited {source}->{target} seed {seed}: paired inputs identical={same}", flush=True)
    with (output / "audit_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "selected_target_vectors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]) if selected else [])
        writer.writeheader()
        writer.writerows(selected)
    save_json(output / "paired_input_comparison.json", comparisons)
    save_json(output / "missing_experiments.json", missing)
    save_json(output / "file_index.json", index)
    save_json(output / "input_sources.json", input_info)
    save_json(output / "run_summary.json", {"experiments_audited": len(rows), "expected_experiments": 60,
                                            "paired_comparisons": len(comparisons),
                                            "all_paired_inputs_identical": all(x["inputs_identical"] for x in comparisons),
                                            "missing_experiments": missing, "near_zero_variance_threshold": NEAR_ZERO_VARIANCE,
                                            "selected_cuts": selected_cuts,
                                            "target_labels_read": False})
    write_report(output)
    print(f"Completed {len(rows)}/60 experiments; output: {output}", flush=True)


if __name__ == "__main__":
    main()
