"""Audit and report the completed six-direction COD comparison."""

from pathlib import Path
import argparse
import json
import math
import re

import numpy as np
import pandas as pd
import torch

import run_cod_comparison as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/cod_comparison_20260926"))
    parser.add_argument("--old-full", type=Path,
                        default=Path("artifacts/full_1_315_baseline_zscore_20260925/per_seed"))
    args = parser.parse_args()
    root = args.root
    smoke = json.loads((root / "smoke.json").read_text(encoding="utf-8"))
    if not all(smoke["checks"].values()):
        raise AssertionError("Smoke test failed")
    table = pd.read_csv(root / "per_seed_metrics.csv")
    if len(table) != 90 or set(table.method) != set(run.METHODS):
        raise AssertionError("Expected 90 rows and three methods")
    checks = []
    for source, target in run.PAIRS:
        for seed in run.SEEDS:
            folder = root / f"{source}_to_{target}" / f"seed_{seed}"
            if not (folder / "complete.json").exists():
                raise AssertionError(f"Missing complete marker: {folder}")
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            trace = json.loads((folder / "daregram_cod" / "training_trace.json").read_text(encoding="utf-8"))
            baseline = Path(cfg["baseline_path"])
            baseline_cfg = json.loads((baseline / "daregram" / "config.json").read_text(encoding="utf-8"))
            if cfg["target_labels_training_reads"] != 0 or \
               trace["actual_unlabeled_target_cuts"] != run.base.ALL_CUTS or \
               trace["initial_model_sha256"] != baseline_cfg["initial_model_sha256"] or \
               trace["source_order_sha256_by_epoch"] != baseline_cfg["source_order_sha256_by_epoch"] or \
               len(trace["epoch_losses"]) != run.EPOCHS or \
               min(trace["cod_gradient_norm_first_active_step"]) <= 0:
                raise AssertionError(f"Training protocol mismatch: {folder}")
            checkpoint = torch.load(folder / "daregram_cod" / "final.pth",
                                    map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != run.EPOCHS or run.sha(folder / "daregram_cod" / "final.pth") != cfg["cod_checkpoint_sha256"]:
                raise AssertionError(f"Checkpoint mismatch: {folder}")
            old_gram_log = (baseline / "daregram" / "run.log").read_text(encoding="utf-8")
            warmup_pattern = re.compile(r"^daregram epoch=(\d+)/50 source_MSE=([\d.eE+-]+) gram=([\d.eE+-]+)", re.M)
            warmup_rows = {int(m.group(1)): (float(m.group(2)), float(m.group(3)))
                           for m in warmup_pattern.finditer(old_gram_log)}
            for record in trace["epoch_losses"][:14]:
                old_mse, old_gram = warmup_rows[record["epoch"]]
                if not np.isclose(record["source_mse"], old_mse, atol=1e-3) or \
                   not np.isclose(record["daregram"], old_gram, atol=1e-3) or \
                   record["cod"] != 0:
                    raise AssertionError(f"Pre-COD warmup differs from historical DARE-GRAM: {folder}")
            for method in run.METHODS:
                path = folder / method / "predictions.csv"
                frame = pd.read_csv(path)
                if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"] or \
                   frame.cut_index.tolist() != run.base.ALL_CUTS or \
                   not np.isfinite(frame[["true_vb", "pred_vb"]].to_numpy()).all():
                    raise AssertionError(f"Prediction schema/cuts/finiteness: {path}")
                metric = run.base.metrics(frame.true_vb.to_numpy(), frame.pred_vb.to_numpy())
                row = table[(table.source == source) & (table.target == target) &
                            (table.seed == seed) & (table.method == method)]
                if len(row) != 1 or any(not np.isclose(metric[key], row.iloc[0][key], atol=1e-8)
                                        for key in metric):
                    raise AssertionError(f"Metric mismatch: {path}")
                if method != "daregram_cod":
                    old = pd.read_csv(args.old_full /
                        f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv")
                    diff = float(np.max(np.abs(frame.pred_vb.to_numpy() - old.pred_vb.to_numpy())))
                    checks.append(diff)
                    if diff > 1e-5:
                        raise AssertionError(f"Baseline prediction changed: {path}: {diff}")
                    # The historical baseline persisted six-decimal epoch logs.
                    # Publish a common four-loss view without changing those files.
                    pattern = re.compile(rf"^{method} epoch=(\d+)/50 source_MSE=([\d.eE+-]+) gram=([\d.eE+-]+)")
                    baseline_rows = []
                    for line in (baseline / method / "run.log").read_text(encoding="utf-8").splitlines():
                        matched = pattern.match(line)
                        if matched:
                            epoch = int(matched.group(1))
                            mse = float(matched.group(2))
                            gram = float(matched.group(3))
                            weight = 0.0 if method == "source_only" else \
                                2 / (1 + math.exp(-10 * (epoch - 1) / (run.EPOCHS - 1))) - 1
                            baseline_rows.append({"epoch": epoch, "source_mse": mse,
                                "daregram": gram, "cod": 0.0,
                                "weighted_daregram": weight * gram,
                                "weighted_cod": 0.0, "total": mse + weight * gram,
                                "provenance": "reconstructed_from_historical_6_decimal_epoch_log"})
                    if len(baseline_rows) != run.EPOCHS:
                        raise AssertionError(f"Missing baseline epoch log: {baseline / method}")
                    pd.DataFrame(baseline_rows).to_csv(folder / method / "epoch_losses.csv", index=False)
                else:
                    losses = pd.read_csv(folder / method / "epoch_losses.csv")
                    if len(losses) != run.EPOCHS:
                        raise AssertionError(f"Missing COD epoch losses: {folder}")
    summary = pd.read_csv(root / "six_direction_three_method_metrics.csv")
    delta = pd.read_csv(root / "cod_minus_daregram_by_direction.csv")
    if len(summary) != 18 or len(delta) != 6 or not (summary.seeds == 5).all():
        raise AssertionError("Incomplete six-direction summary")
    if not (root / "plot_coordinates.json").exists():
        raise AssertionError("Unified plot coordinates missing")
    lines = ["# PHM2010 COD six-direction comparison", "",
             "Five fixed seeds (42–46); mean of per-seed CUT 1–315 metrics. "
             "MAPE is in percent. No target labels were used for training, model selection or parameter choice.", "",
             "## Six directions × three methods", "",
             "| Direction | Method | R² | MAE | RMSE | MAPE (%) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | {row.method} | "
                     f"{row.R2:.4f} | {row.MAE:.4f} | {row.RMSE:.4f} | {row.MAPE_percent:.4f} |")
    lines += ["", "## COD minus DARE-GRAM", "",
              "Positive ΔR² is better; negative ΔMAE, ΔRMSE and ΔMAPE are better. "
              "All six directions are shown, including regressions.", "",
              "| Direction | ΔR² | ΔMAE | ΔRMSE | ΔMAPE (pp) |",
              "|---|---:|---:|---:|---:|"]
    for row in delta.itertuples():
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | {row.R2:+.4f} | "
                     f"{row.MAE:+.4f} | {row.RMSE:+.4f} | {row.MAPE_percent:+.4f} |")
    pivot = table.pivot(index=["source", "target", "seed"], columns="method",
                        values=["R2", "MAE", "RMSE", "MAPE_percent"])
    per_seed_delta = pd.DataFrame({metric: pivot[(metric, "daregram_cod")] -
                                   pivot[(metric, "daregram")]
                                   for metric in ("R2", "MAE", "RMSE", "MAPE_percent")}).reset_index()
    per_seed_delta.to_csv(root / "cod_minus_daregram_per_seed.csv", index=False)
    counts = per_seed_delta.groupby(["source", "target"], as_index=False).agg(
        r2_improved=("R2", lambda x: int((x > 0).sum())),
        mae_improved=("MAE", lambda x: int((x < 0).sum())))
    lines += ["", "## Per-seed direction counts", "",
              "| Direction | R² improved | MAE improved |",
              "|---|---:|---:|"]
    for row in counts.itertuples():
        lines.append(f"| {row.source.upper()}→{row.target.upper()} | "
                     f"{row.r2_improved}/5 | {row.mae_improved}/5 |")
    lines += ["", "COD improves all four mean metrics for C1→C6 and C4→C1. "
              "It worsens all four for C6→C1; the other directions are mixed. "
              "Per-seed signs vary, so these fixed-seed means do not establish a universal gain.", ""]
    lines += ["", "## Audit", "",
              f"- Verified 30 COD final checkpoints and 90 ordered 315-row prediction CSVs.",
              f"- Maximum absolute difference between the 60 re-evaluated baseline predictions "
              f"and the frozen full-lifecycle audit: {max(checks):.3g}.",
              "- Initial weights, source batch order, 315 unlabeled target cuts, final epoch "
              "selection, finite metrics and nonzero COD gradient were checked per run.",
              "- Every curve uses the same axes in `plot_coordinates.json`.", ""]
    lines += ["The baseline epoch loss files in this new directory reconstruct total loss "
              "from the historical six-decimal source/Gram log; the COD epoch loss file "
              "contains the unrounded four components recorded during training.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit passed: {len(table)} metric rows; maximum baseline prediction difference {max(checks):.3g}")


if __name__ == "__main__":
    main()
