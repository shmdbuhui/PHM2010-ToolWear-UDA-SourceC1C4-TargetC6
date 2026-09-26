"""Verify completed trend runs and generate baseline curves without altering predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_five_seed_pairs as five
import run_single_source_pairs as base
from run_trend_physics_comparison import plot_curve
from trend_physics import adjacent_positions, physical_audit


def shuffled_pairs(seed, exhaust_iterator=True):
    loader = torch.utils.data.DataLoader(torch.arange(1, 316), batch_size=five.BATCH_SIZE,
                                          shuffle=True, generator=torch.Generator().manual_seed(seed))
    counts, coverage, hashes = [], set(), []
    for _ in range(five.EPOCHS):
        if exhaust_iterator:
            batches = [batch.tolist() for batch in loader]
        else:
            iterator = iter(loader)
            batches = [next(iterator).tolist() for _ in range(315 // five.BATCH_SIZE)]
        flat = np.asarray([cut for batch in batches for cut in batch], dtype=np.int32)
        if sorted(flat.tolist()) != base.ALL_CUTS:
            raise ValueError("Shuffled loader lost cuts")
        import hashlib
        hashes.append(hashlib.sha256(flat.tobytes()).hexdigest())
        pairs = [adjacent_positions(["tool"] * len(batch), batch) for batch in batches]
        counts.append(sum(len(left) for left, _ in pairs))
        for batch, (left, _) in zip(batches, pairs):
            coverage.update(batch[i] for i in left)
    return counts, coverage, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("artifacts/trend_physics_full_1_315_20260925"))
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    parser.add_argument("--mark-declines", action="store_true")
    args = parser.parse_args()
    records, missing = [], []
    for source, target in five.PAIRS:
        for seed in five.SEEDS:
            folder = args.result_root / f"{source}_to_{target}" / f"seed_{seed}"
            if not (folder / "complete.json").exists():
                missing.append(f"{source}_to_{target}/seed_{seed}")
                continue
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if cfg["target_labels_training_reads"] != 0 or cfg["source_only_training_target_cuts"] != [] or cfg["daregram_training_target_cuts"] != base.ALL_CUTS:
                raise ValueError(f"Target isolation mismatch: {folder}")
            source_counts, source_coverage, source_hashes = shuffled_pairs(seed)
            target_counts, target_coverage, _ = shuffled_pairs(seed + 1, exhaust_iterator=False)
            if source_coverage != set(range(1, 315)) or target_coverage != set(range(1, 315)):
                raise ValueError(f"Some adjacent edge never appeared: {folder}")
            for method in ("source_only", "daregram"):
                sub = folder / method
                trace = json.loads((sub / "training_trace.json").read_text(encoding="utf-8"))
                loss = pd.read_csv(sub / "epoch_losses.csv")
                if (len(loss) != five.EPOCHS or trace["source_order_sha256_by_epoch"] != source_hashes or
                    loss.source_valid_pairs.tolist() != source_counts or
                    loss.target_valid_pairs.tolist() != (target_counts if method == "daregram" else [0] * five.EPOCHS)):
                    raise ValueError(f"Pair/order mismatch: {sub}")
                if (not np.isfinite(loss.select_dtypes(include="number").to_numpy()).all() or
                    not (loss.supervised_mse > 0).all()):
                    raise ValueError(f"Invalid training loss: {sub}")
                if method == "source_only" and trace["actual_unlabeled_target_cuts"] != []:
                    raise ValueError(f"Source-only saw target cuts: {sub}")
                if method == "daregram" and trace["actual_unlabeled_target_cuts"] != base.ALL_CUTS:
                    raise ValueError(f"DARE-GRAM did not see entire target tool: {sub}")
                frame = pd.read_csv(sub / "predictions.csv")
                if frame.cut_index.tolist() != base.ALL_CUTS or not np.isfinite(frame[["true_vb", "pred_vb"]].to_numpy()).all():
                    raise ValueError(f"Incomplete/nonfinite predictions: {sub}")
                reported = json.loads((sub / "physical_summary.json").read_text(encoding="utf-8"))
                reported.setdefault("missing_cut_count", len(reported["missing_cuts"]))
                reported.setdefault("duplicate_cut_count", len(reported["duplicate_cuts"]))
                _, audit = physical_audit(frame.cut_index, frame.pred_vb, cfg["delta_vb"], cfg["source_label_max_vb"])
                for key in audit:
                    if isinstance(audit[key], float):
                        if not np.isclose(audit[key], reported[key], rtol=1e-9, atol=1e-9):
                            raise ValueError(f"Physical audit mismatch {key}: {sub}")
                    elif audit[key] != reported[key]:
                        raise ValueError(f"Physical audit mismatch {key}: {sub}")
                baseline = pd.read_csv(args.baseline_root / "per_seed" /
                                       f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv")
                if baseline.cut_index.tolist() != base.ALL_CUTS or not np.allclose(baseline.true_vb, frame.true_vb, atol=1e-6, rtol=0):
                    raise ValueError(f"Baseline mismatch: {sub}")
                base_per_cut, base_audit = physical_audit(baseline.cut_index, baseline.pred_vb,
                                                         cfg["delta_vb"], cfg["source_label_max_vb"])
                plot_curve(sub / "baseline_prediction.png", source, target, seed, method + " baseline",
                           baseline.cut_index.to_numpy(), baseline.true_vb.to_numpy(),
                           baseline.pred_vb.to_numpy(), base_per_cut, args.mark_declines)
                if audit["prediction_variance_vb2"] <= 1e-6:
                    raise ValueError(f"Nearly constant prediction: {sub}")
                records.append({"source": source, "target": target, "seed": seed, "method": method,
                                "source_edge_coverage_50_epochs": len(source_coverage),
                                "target_edge_coverage_50_epochs": len(target_coverage) if method == "daregram" else 0,
                                "supervised_mse_final": float(loss.supervised_mse.iloc[-1]),
                                "domain_gram_final": float(loss.domain_gram.iloc[-1]),
                                "source_trend_positive_epochs": int((loss.source_trend_loss > 0).sum()),
                                "target_trend_positive_epochs": int((loss.target_trend_loss > 0).sum()),
                                "prediction_variance_vb2": audit["prediction_variance_vb2"]})
    pd.DataFrame(records).to_csv(args.result_root / "validation.csv", index=False)
    plot_dir = args.result_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    for source, target in five.PAIRS:
        if any(f"{source}_to_{target}/seed_{seed}" in missing for seed in five.SEEDS):
            continue
        curves = {"cut_index": base.ALL_CUTS}
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = {"source_only": "#2563a6", "source_only_trend": "#37a8c7",
                  "daregram": "#db7026", "daregram_trend": "#a23b7b"}
        truth = None
        for method in ("source_only", "source_only_trend", "daregram", "daregram_trend"):
            values = []
            for seed in five.SEEDS:
                if method.endswith("_trend"):
                    path = args.result_root / f"{source}_to_{target}" / f"seed_{seed}" / method.removesuffix("_trend") / "predictions.csv"
                else:
                    path = args.baseline_root / "per_seed" / f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
                frame = pd.read_csv(path)
                if truth is None:
                    truth = frame.true_vb.to_numpy(float)
                elif not np.allclose(truth, frame.true_vb, atol=1e-6, rtol=0):
                    raise ValueError(f"Plot labels differ: {source}->{target}")
                values.append(frame.pred_vb.to_numpy(float))
            matrix = np.stack(values)
            mean = matrix.mean(axis=0)
            curves[f"{method}_mean_vb"] = mean
            curves[f"{method}_seed_sd_vb"] = matrix.std(axis=0, ddof=1)
            ax.plot(base.ALL_CUTS, mean, color=colors[method], lw=1.5, label=method)
        curves["true_vb"] = truth
        ax.plot(base.ALL_CUTS, truth, color="black", lw=2, label="True VB")
        ax.set(xlabel="Cut", ylabel="VB", title=f"{source.upper()}→{target.upper()}: five-seed raw prediction mean")
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{source}_to_{target}_comparison.png", dpi=190)
        plt.close(fig)
        pd.DataFrame(curves).to_csv(plot_dir / f"{source}_to_{target}_curves.csv", index=False, float_format="%.17g")
    five.json_write(args.result_root / "validation.json", {"complete_pair_seed_count": len(records) // 2,
                   "expected_pair_seed_count": len(five.PAIRS) * len(five.SEEDS),
                   "missing_pair_seeds": missing, "all_completed_checks_pass": True,
                   "pair_coverage_definition": "union of valid same-tool adjacent cut edges over 50 original shuffled batches"})
    print(f"Validated {len(records)//2}/30 direction-seed pairs; missing={len(missing)}")


if __name__ == "__main__":
    main()
