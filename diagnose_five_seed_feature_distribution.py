"""Diagnose within-checkpoint source/target feature distances for the Z-score main run.

Uses the already extracted, read-only 512-D features from audit_five_seed_features.py.
Verifies their hashes and provenance against the current final checkpoints before use.
No target labels, prediction values, or model training are read or run here.
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
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


TOOLS = ("c1", "c4", "c6")
SEEDS = range(42, 47)
METHODS = ("source_only", "daregram")
DIM = 512
NEAR_ZERO_VARIANCE = 1e-12
SIGMA_FLOOR = 1e-8
COLORS = {"source": "#3267a8", "target": "#d78128"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_features(path: Path, tool: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        expected = ["tool_id", "cut_index"] + [f"feature_{i:03d}" for i in range(DIM)]
        if next(reader) != expected:
            raise ValueError(f"Feature columns changed: {path}")
        rows = list(reader)
    if len(rows) != 315 or any(len(row) != DIM + 2 or row[0] != tool or
                               int(row[1]) != i for i, row in enumerate(rows, 1)):
        raise ValueError(f"Feature rows/cuts changed: {path}")
    value = np.asarray([row[2:] for row in rows], dtype=np.float64)
    if value.shape != (315, DIM) or not np.isfinite(value).all():
        raise ValueError(f"Nonfinite or incorrect feature array: {path}")
    return value


def distances(source: np.ndarray, target: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    source_var = np.var(source, axis=0, ddof=0)
    target_var = np.var(target, axis=0, ddof=0)
    source_sigma = np.sqrt(source_var)
    if not np.isfinite(source_var).all() or not np.isfinite(target_var).all():
        raise ValueError("Nonfinite feature variances")
    # A zero/very small source sigma is diagnosed and floored only for this analysis.
    scale = np.maximum(source_sigma, SIGMA_FLOOR)
    zs = (source - source.mean(axis=0)) / scale
    zt = (target - source.mean(axis=0)) / scale
    mean_distance = np.linalg.norm(zs.mean(axis=0) - zt.mean(axis=0)) / np.sqrt(DIM)
    covariance_distance = np.linalg.norm(np.cov(zs, rowvar=False, ddof=1) -
                                         np.cov(zt, rowvar=False, ddof=1), ord="fro") / DIM
    stats = {
        "mean_distance": float(mean_distance),
        "covariance_distance": float(covariance_distance),
        "source_nan_count": int(np.isnan(source).sum()),
        "target_nan_count": int(np.isnan(target).sum()),
        "source_inf_count": int(np.isinf(source).sum()),
        "target_inf_count": int(np.isinf(target).sum()),
        "source_constant_dimensions": int(np.count_nonzero(source_var == 0)),
        "target_constant_dimensions": int(np.count_nonzero(target_var == 0)),
        "source_near_zero_variance_dimensions": int(np.count_nonzero(source_var <= NEAR_ZERO_VARIANCE)),
        "target_near_zero_variance_dimensions": int(np.count_nonzero(target_var <= NEAR_ZERO_VARIANCE)),
        "source_sigma_floored_dimensions": int(np.count_nonzero(source_sigma < SIGMA_FLOOR)),
    }
    if not np.isfinite([mean_distance, covariance_distance]).all():
        raise ValueError("Nonfinite distances")
    return stats, zs, zt


def pca_panel(ax, zs: np.ndarray, zt: np.ndarray, title: str) -> float:
    # This PCA is fit separately for every checkpoint. Panel axes are independent.
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    points = pca.fit_transform(np.vstack([zs, zt]))
    for label, part in (("source", points[:315]), ("target", points[315:])):
        ax.scatter(part[:, 0], part[:, 1], s=9, alpha=0.38,
                   color=COLORS[label], label=label, linewidths=0)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    ax.legend(loc="best", frameon=False, markerscale=1.8)
    ax.grid(alpha=0.15)
    return float(pca.explained_variance_ratio_.sum())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--feature-audit-root", type=Path,
                        default=Path("artifacts/five_seed_feature_audit_20260925"))
    parser.add_argument("--out-root", type=Path,
                        default=Path("artifacts/five_seed_feature_distribution_20260925"))
    args = parser.parse_args()
    root, audit_root, out = (p.resolve() for p in
                             (args.experiment_root, args.feature_audit_root, args.out_root))
    if out.exists() or out == root or root in out.parents or out == audit_root or audit_root in out.parents:
        parser.error("Output must be a new directory outside the input experiment and audit roots")
    run = read_json(audit_root / "run_summary.json")
    if (run["experiments_audited"], run["expected_experiments"],
        run["paired_comparisons"], run["all_paired_inputs_identical"],
        run["missing_experiments"], run["target_labels_read"]) != (60, 60, 30, True, [], False):
        raise ValueError("Previous feature extraction audit is incomplete")
    index = {(r["source"], r["target"], r["seed"], r["method"]): r
             for r in read_json(audit_root / "file_index.json")}
    if len(index) != 60:
        raise ValueError("Expected exactly 60 indexed feature extractions")
    out.mkdir(parents=True)
    (out / "pca").mkdir()
    records, pair_rows, provenance = [], [], []
    with threadpool_limits(limits=4):
        for source in TOOLS:
            for target in TOOLS:
                if source == target:
                    continue
                for seed in SEEDS:
                    pair = root / f"{source}_to_{target}" / f"seed_{seed}"
                    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
                    current = {}
                    configs = {}
                    for ax, method in zip(axes, METHODS):
                        folder = pair / method
                        entry = index[source, target, seed, method]
                        config_path = folder / "config.json"
                        checkpoint_path = folder / "final.pth"
                        audit_path = audit_root / "experiments" / f"{source}_to_{target}" / f"seed_{seed}" / method / "audit.json"
                        config, extraction_audit = read_json(config_path), read_json(audit_path)
                        if (config["source"], config["target"], config["seed"], config["method"]) != (source, target, seed, method):
                            raise ValueError(f"Wrong checkpoint/config identity: {config_path}")
                        if config["checkpoint_selection"] != "final epoch" or config["regressor"] != "Linear(512, 1)":
                            raise ValueError(f"Unexpected model protocol: {config_path}")
                        if config["evaluation_cuts"] != (list(range(95, 316)) if target == "c6" else list(range(1, 316))):
                            raise ValueError(f"Wrong evaluation cuts: {config_path}")
                        if entry["normalization_source"] != source or extraction_audit["normalization_source"] != source:
                            raise ValueError(f"Wrong normalization source: {audit_path}")
                        if (config["source_normalization_mean"] != extraction_audit["source_normalization_mean"] or
                            config["source_normalization_std"] != extraction_audit["source_normalization_std"]):
                            raise ValueError(f"Normalization does not match extracted features: {audit_path}")
                        checkpoint_hash = digest(checkpoint_path)
                        if checkpoint_hash != extraction_audit["checkpoint_sha256"]:
                            raise ValueError(f"Checkpoint changed since feature extraction: {checkpoint_path}")
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                        if (checkpoint["epoch"] != config["epochs"] or
                            tuple(checkpoint["model"]["regressor.0.weight"].shape) != (1, DIM) or
                            tuple(checkpoint["model"]["regressor.0.bias"].shape) != (1,)):
                            raise ValueError(f"Wrong final checkpoint structure: {checkpoint_path}")
                        del checkpoint
                        configs[method] = config
                        vectors = {}
                        for role, tool in (("source", source), ("target", target)):
                            path = audit_root / "experiments" / f"{source}_to_{target}" / f"seed_{seed}" / method / f"{role}_features.csv"
                            if digest(path) != extraction_audit["feature_stats"][role]["file_sha256"]:
                                raise ValueError(f"Extracted features changed: {path}")
                            vectors[role] = read_features(path, tool)
                        stats, zs, zt = distances(vectors["source"], vectors["target"])
                        pca_variance = pca_panel(ax, zs, zt, method)
                        ax.set_title(f"{method} | PC1+PC2 {pca_variance:.1%}")
                        rmse = float(read_json(folder / "metrics.json")["RMSE"])
                        if not np.isfinite(rmse):
                            raise ValueError(f"Nonfinite RMSE: {folder}")
                        current[method] = {"rmse": rmse, **stats}
                        records.append({"source": source, "target": target, "seed": seed,
                                        "method": method, "evaluation_count": len(config["evaluation_cuts"]),
                                        "pca_two_component_variance_ratio": pca_variance,
                                        **current[method]})
                        provenance.append({"source": source, "target": target, "seed": seed,
                                           "method": method, "checkpoint_sha256": checkpoint_hash,
                                           "checkpoint": str(checkpoint_path), "config": str(config_path),
                                           "feature_audit": str(audit_path),
                                           "normalization_source": source,
                                           "source_normalization_mean": config["source_normalization_mean"],
                                           "source_normalization_std": config["source_normalization_std"]})
                    if (configs["source_only"]["source_normalization_mean"] != configs["daregram"]["source_normalization_mean"] or
                        configs["source_only"]["source_normalization_std"] != configs["daregram"]["source_normalization_std"]):
                        raise ValueError(f"Paired normalization differs: {pair}")
                    figure_name = f"{source}_to_{target}_seed_{seed}.png"
                    fig.suptitle(f"{source.upper()} → {target.upper()} | seed {seed} | independently fitted PCA")
                    fig.savefig(out / "pca" / figure_name, dpi=150)
                    plt.close(fig)
                    s, d = current["source_only"], current["daregram"]
                    pair_rows.append({"source": source, "target": target, "seed": seed,
                                      "evaluation_count": len(configs["source_only"]["evaluation_cuts"]),
                                      "source_only_mean_distance": s["mean_distance"],
                                      "daregram_mean_distance": d["mean_distance"],
                                      "delta_mean_distance": d["mean_distance"] - s["mean_distance"],
                                      "source_only_covariance_distance": s["covariance_distance"],
                                      "daregram_covariance_distance": d["covariance_distance"],
                                      "delta_covariance_distance": d["covariance_distance"] - s["covariance_distance"],
                                      "source_only_rmse": s["rmse"], "daregram_rmse": d["rmse"],
                                      "delta_rmse": d["rmse"] - s["rmse"],
                                      "pca_figure": f"pca/{figure_name}"})
                    print(f"Diagnosed {source}->{target} seed {seed}", flush=True)
    write_csv(out / "checkpoint_metrics.csv", records)
    write_csv(out / "paired_seed_metrics.csv", pair_rows)
    summary = []
    measure_names = ("source_only_mean_distance", "daregram_mean_distance", "delta_mean_distance",
                     "source_only_covariance_distance", "daregram_covariance_distance",
                     "delta_covariance_distance", "source_only_rmse", "daregram_rmse", "delta_rmse")
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            subset = [r for r in pair_rows if r["source"] == source and r["target"] == target]
            row = {"source": source, "target": target, "n_seeds": len(subset)}
            for name in measure_names:
                values = np.asarray([r[name] for r in subset])
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_sd"] = float(values.std(ddof=1))
            for name in ("mean", "covariance"):
                row[f"seeds_both_{name}_and_rmse_improve"] = sum(
                    r[f"delta_{name}_distance"] < 0 and r["delta_rmse"] < 0 for r in subset)
                row[f"seeds_{name}_improves_rmse_worsens"] = sum(
                    r[f"delta_{name}_distance"] < 0 and r["delta_rmse"] > 0 for r in subset)
                row[f"seeds_{name}_worsens_rmse_improves"] = sum(
                    r[f"delta_{name}_distance"] > 0 and r["delta_rmse"] < 0 for r in subset)
            summary.append(row)
    write_csv(out / "direction_summary.csv", summary)
    (out / "provenance.json").write_text(json.dumps({"feature_position":
        "ResNet18 global-average-pool output, fc=Identity, direct input to Linear(512,1)",
        "feature_extraction_source": str(audit_root), "experiment_source": str(root),
        "standardization": "per checkpoint: (features - own source 315-cut mean) / max(own source population sigma, 1e-8)",
        "near_zero_variance_threshold": NEAR_ZERO_VARIANCE,
        "covariance": "sample covariance, ddof=1, over all 315 source and 315 target cuts",
        "pca": "each checkpoint fitted independently on its 630 standardized source+target features",
        "rmse": "existing metrics.json on original evaluation cuts; C6 targets use 95..315 only",
        "target_labels_read": False, "checkpoints": provenance}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Completed {len(records)} checkpoints, {len(pair_rows)} paired seeds: {out}")


if __name__ == "__main__":
    main()
