"""Evaluate saved five-seed checkpoints on every target cut, without training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

import run_five_seed_pairs as paired
import run_single_source_pairs as base


CUTS = base.ALL_CUTS
EARLY_C6 = list(range(1, 95))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked_csv(path: Path, cuts: list[int], labels: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"]:
        raise ValueError(f"Unexpected CSV columns: {path}")
    if len(frame) != len(cuts) or frame.cut_index.tolist() != cuts or frame.cut_index.duplicated().any():
        raise ValueError(f"Missing, repeated or misordered cut: {path}")
    if not np.isfinite(frame[["true_vb", "pred_vb"]].to_numpy()).all():
        raise ValueError(f"Nonfinite true/predicted value: {path}")
    if labels is not None and not np.allclose(frame.true_vb.to_numpy(), labels[np.array(cuts) - 1],
                                               rtol=0, atol=1e-12):
        raise ValueError(f"True wear is misaligned with cut index: {path}")
    return frame


def preflight(trained_root: Path, raw_root: Path):
    manifest_path = trained_root / "feature_cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["cuts"] != CUTS or manifest["feature_shape"] != [315, 6, 128, 128]:
        raise ValueError("Feature cache does not cover target cuts 1..315")
    if manifest["raw_root"] != str(raw_root.resolve()):
        raise ValueError("Raw data root differs from training cache")
    for tool in base.TOOLS:
        cache = trained_root / "feature_cache" / f"{tool}_stft.npy"
        if base.file_hash(cache) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"Feature cache checksum mismatch: {cache}")
    for source, target in paired.PAIRS:
        assert source != target
        for seed in paired.SEEDS:
            folder = trained_root / f"{source}_to_{target}" / f"seed_{seed}"
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
            if (config["source"], config["target"], config["seed"]) != (source, target, seed):
                raise ValueError(f"Wrong training configuration: {folder}")
            if (config["source_feature_sha256"] != manifest["tools"][source]["feature_sha256"] or
                config["target_feature_sha256"] != manifest["tools"][target]["feature_sha256"] or
                config["epochs"] != 50 or config["checkpoint_selection"] != "final epoch"):
                raise ValueError(f"Checkpoint or cached feature provenance differs: {folder}")
            if (config["source_count"] != 315 or config["source_cuts"] != CUTS or
                config["target_unlabeled_cuts"] != CUTS or
                audit["target_unlabeled_cuts"] != CUTS or
                audit["source_only_training_target_cuts"] != [] or
                audit["target_label_reads_during_training"] != 0 or
                not audit["paired_initialization_verified"] or
                not audit["paired_source_order_verified"]):
                raise ValueError(f"Training protocol audit failed: {folder}")
            for method in paired.METHODS:
                ckpt = folder / method / "final.pth"
                if not ckpt.is_file():
                    raise FileNotFoundError(f"Final checkpoint required; no retraining will be attempted: {ckpt}")
                if base.file_hash(ckpt) != audit["checkpoint_sha256"][method]:
                    raise ValueError(f"Checkpoint checksum differs from training audit: {ckpt}")
    return manifest


def normalized_target(trained_root: Path, target: str, config: dict) -> np.ndarray:
    raw = np.load(trained_root / "feature_cache" / f"{target}_stft.npy", mmap_mode="r")
    if raw.shape != (315, 6, 128, 128) or raw.dtype != np.float32:
        raise ValueError(f"Invalid cached STFT features: {target}")
    mean = np.asarray(config["source_normalization_mean"], dtype=np.float32)
    std = np.asarray(config["source_normalization_std"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0):
        raise ValueError("Invalid source-only normalization statistics")
    return ((raw - mean[None, :, None, None]) /
            (std[None, :, None, None] + 1e-8)).astype(np.float32)


def evaluate_pair(args, source: str, target: str, seed: int):
    old = args.trained_root / f"{source}_to_{target}" / f"seed_{seed}"
    new = args.out_root / f"{source}_to_{target}" / f"seed_{seed}"
    config = json.loads((old / "config.json").read_text(encoding="utf-8"))
    old_cuts = list(range(95, 316)) if target == "c6" else CUTS
    inferred = {}
    if target == "c6":
        features = normalized_target(args.trained_root, target, config)
        for method in paired.METHODS:
            inferred[method] = paired.predict(method, features, EARLY_C6, seed, old, torch.device(args.device))
        del features
    # No target wear labels are opened until all required checkpoint-based predictions are fixed.
    labels = base.wear_labels(args.raw_root, target, evaluation_dir=old).astype(np.float64)
    if labels.shape != (315,):
        raise ValueError(f"Wrong target label count: {target}")
    results = {}
    for method in paired.METHODS:
        old_csv = old / method / "predictions.csv"
        prior = checked_csv(old_csv, old_cuts, labels)
        pred = (np.concatenate((inferred[method], prior.pred_vb.to_numpy(dtype=np.float64)))
                if target == "c6" else prior.pred_vb.to_numpy(dtype=np.float64))
        if len(pred) != 315:
            raise ValueError(f"Expected one prediction for every target cut: {old_csv}")
        folder = new / method
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "predictions.csv"
        pd.DataFrame({"cut_index": CUTS, "true_vb": labels, "pred_vb": pred}).to_csv(
            output, index=False, float_format="%.17g")
        checked = checked_csv(output, CUTS, labels)
        result = base.metrics(checked.true_vb.to_numpy(), checked.pred_vb.to_numpy())
        write_json(folder / "metrics.json", result)
        write_json(folder / "config.json", {
            "source": source, "target": target, "seed": seed, "method": method,
            "evaluation_scope": "全程 1–315 评估", "evaluation_cuts": CUTS,
            "training_checkpoint": str((old / method / "final.pth").resolve()),
            "checkpoint_sha256": base.file_hash(old / method / "final.pth"),
            "training_config": str((old / "config.json").resolve()),
            "source_normalization": "saved source-only training statistics",
            "prediction_provenance": "checkpoint inference for cuts 1..94; verified legacy CSV for 95..315"
                if target == "c6" else "verified legacy full-cut prediction CSV",
            "target_labels_first_read": "after final checkpoints verified and missing predictions inferred",
            "training_performed": False,
        })
        fig, ax = base.plt.subplots(figsize=(10, 4.5))
        ax.plot(CUTS, labels, label="True VB", lw=1.8)
        ax.plot(CUTS, pred, label="Predicted VB", lw=1.5)
        ax.set(xlim=(1, 315), xlabel="Target cut index", ylabel="VB",
               title=f"{source.upper()} to {target.upper()} | seed {seed} | {method} | Full lifecycle 1–315 evaluation")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(folder / "prediction.png", dpi=200)
        base.plt.close(fig)
        results[method] = result
    write_json(new / "comparison.json", {
        "source_only": results["source_only"], "daregram": results["daregram"],
        "daregram_minus_source_only": {metric: results["daregram"][metric] -
                                       results["source_only"][metric] for metric in base.METRICS},
    })
    logging.info("Evaluated %s->%s seed=%d, 315 cuts (%s)", source, target, seed,
                 "C6 cuts 1..94 inferred" if target == "c6" else "existing full CSV verified")
    return results


def summarize(out_root: Path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(out_root / "seed_metrics.csv", index=False)
    summary = []
    for source, target in paired.PAIRS:
        group = [x for x in rows if (x["source"], x["target"]) == (source, target)]
        if len(group) != 5:
            raise ValueError(f"Expected five seeds: {source}->{target}")
        row = {"source": source, "target": target, "n": 5, "evaluation_count": 315,
               "evaluation_cuts": ",".join(map(str, CUTS)),
               "daregram_improved_MAE_count": sum(x["delta_MAE"] < 0 for x in group),
               "daregram_improved_R2_count": sum(x["delta_R2"] > 0 for x in group)}
        for metric in base.METRICS:
            for method in ("source_only", "daregram", "delta"):
                values = np.asarray([x[f"{method}_{metric}"] for x in group])
                row[f"{method}_{metric}_mean"] = float(values.mean())
                row[f"{method}_{metric}_std"] = float(values.std(ddof=1))
                if method == "delta":
                    margin = float(student_t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5))
                    row[f"delta_{metric}_ci95_low"] = float(values.mean() - margin)
                    row[f"delta_{metric}_ci95_high"] = float(values.mean() + margin)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    lines = ["# PHM2010: 全程 1–315 评估 (five paired seeds)", "",
             "All six directions use target cuts 1–315. Each prediction CSV has one row per cut.",
             "Saved final-epoch checkpoints were evaluated; no model was retrained or selected using target labels.",
             "For C6, cuts 1–94 were inferred from the saved checkpoints; verified older predictions were used for 95–315.",
             "DARE-GRAM training used all 315 **unlabeled** target inputs: transductive UDA. Source-only used no target features in training.",
             "Target wear labels were opened only after all needed checkpoints and predictions were fixed.",
             "The old C6 95–315 metrics and plots remain under `../five_seed_paired/` and are the **旧评估口径**.",
             "Method values below are mean ± sample SD (n=5). Paired delta is DARE-GRAM minus source-only within each seed; 95% t interval uses df=4.", "",
             "| Direction | Metric | Source-only | DARE-GRAM | Paired delta | 95% t interval |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        for metric in base.METRICS:
            lines.append(f"| {row['source'].upper()}→{row['target'].upper()} | {metric} | "
                         f"{row[f'source_only_{metric}_mean']:.4f} ± {row[f'source_only_{metric}_std']:.4f} | "
                         f"{row[f'daregram_{metric}_mean']:.4f} ± {row[f'daregram_{metric}_std']:.4f} | "
                         f"{row[f'delta_{metric}_mean']:+.4f} ± {row[f'delta_{metric}_std']:.4f} | "
                         f"[{row[f'delta_{metric}_ci95_low']:+.4f}, {row[f'delta_{metric}_ci95_high']:+.4f}] |")
    lines += ["", "| Direction | MAE improved seeds | R² improved seeds |",
              "|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['source'].upper()}→{row['target'].upper()} | "
                     f"{row['daregram_improved_MAE_count']}/5 | {row['daregram_improved_R2_count']}/5 |")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_root / "completion.json", {"direction_seed_pairs": 30, "evaluated_checkpoints": 60,
               "new_trainings": 0, "evaluation_cuts": CUTS, "old_c6_evaluation": "95..315 (旧评估口径)"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/five_seed_full_lifecycle"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.out_root.resolve() == args.trained_root.resolve() or args.out_root.exists():
        parser.error("Output root must be new and separate from preserved training results")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    preflight(args.trained_root, args.raw_root)
    args.out_root.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out_root / "evaluation.log", encoding="utf-8")])
    rows = []
    for source, target in paired.PAIRS:
        for seed in paired.SEEDS:
            result = evaluate_pair(args, source, target, seed)
            row = {"source": source, "target": target, "seed": seed, "evaluation_count": 315,
                   "evaluation_cuts": ",".join(map(str, CUTS))}
            for method in paired.METHODS:
                row.update({f"{method}_{metric}": result[method][metric] for metric in base.METRICS})
            row.update({f"delta_{metric}": result["daregram"][metric] -
                        result["source_only"][metric] for metric in base.METRICS})
            rows.append(row)
    summarize(args.out_root, rows)
    logging.info("Completed 30 direction-seed evaluations from 60 saved checkpoints; 0 trainings")


if __name__ == "__main__":
    main()
