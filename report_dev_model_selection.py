"""Produce six-direction DEV selection narrative and five-seed raw curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_dev_model_selection as dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/dev_model_selection_20260926"))
    a = parser.parse_args()
    selected = pd.read_csv(a.root / "selected_models.csv")
    scores = pd.read_csv(a.root / "dev_scores.csv")
    compare = pd.read_csv(a.root / "selected_vs_daregram_best_worst.csv")
    pred = pd.read_csv(a.root / "predictions_per_cut.csv")
    if len(selected) != 30 or len(scores) != 180 or len(pred) != 30 * 315:
        raise AssertionError("Six-direction, five-seed grid is incomplete")
    curve_root = a.root / "selected_curves"
    curve_root.mkdir(exist_ok=True)
    lines = ["# DEV selection results", "",
             "All metrics below use the newly retrained checkpoints with the 252/63 source split.",
             "The oracle uses target wear only after model selection and is diagnostic.", ""]
    for source, target in dev.joint.PAIRS:
        sub = selected.loc[(selected.source == source) & (selected.target == target)].sort_values("seed")
        if sub.seed.tolist() != list(dev.joint.SEEDS):
            raise AssertionError(f"Missing seed for {source}->{target}")
        fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
        lines += [f"## {source.upper()} → {target.upper()}", "",
                  "| Seed | Selector | Selected | Score | ESS | Selected MAE | DARE-GRAM MAE | Oracle MAE |",
                  "|---:|---|---|---:|---:|---:|---:|---:|"]
        for ax, row in zip(axes, sub.itertuples()):
            cand = scores.loc[(scores.source == source) & (scores.target == target) &
                              (scores.seed == row.seed)].set_index("method")
            s = cand.loc[row.selected_method]
            ess = s.weight_ess if row.selection_variant == "DEV" else s.dev_guard_weight_ess
            comp = compare.loc[(compare.source == source) & (compare.target == target) &
                               (compare.seed == row.seed)].iloc[0]
            lines.append(f"| {row.seed} | {row.selection_variant} | {row.selected_method} | "
                         f"{row.selected_score:.4f} | {ess:.1f} | {comp.selected_MAE:.3f} | "
                         f"{comp.daregram_MAE:.3f} | {comp.oracle_MAE:.3f} |")
            curve = pred.loc[(pred.source == source) & (pred.target == target) &
                             (pred.seed == row.seed)].sort_values("cut")
            ax.plot(curve.cut, curve.true_vb, label="True VB", linewidth=1.4)
            ax.plot(curve.cut, curve.pred_vb, label=f"{row.selected_method} ({row.selection_variant})", linewidth=1.1)
            ax.set_ylabel(f"Seed {row.seed}\nVB")
            ax.grid(alpha=.22); ax.legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel(f"{target.upper()} cut")
        fig.suptitle(f"{source.upper()} → {target.upper()}: selected raw predictions")
        fig.tight_layout()
        path = curve_root / f"{source}_to_{target}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        counts = sub.selected_method.value_counts().to_dict()
        consistency = "all five seeds agree" if len(counts) == 1 else "seeds differ"
        guard_count = int((sub.selection_variant == "DEV-guard").sum())
        direction_scores = scores.loc[(scores.source == source) & (scores.target == target)]
        low_raw = direction_scores.loc[direction_scores.weight_ess < 10, ["seed", "method"]]
        invalid_raw = direction_scores.loc[direction_scores.status != "ok", ["seed", "method", "anomalies"]]
        low_guard = direction_scores.loc[direction_scores.dev_guard_weight_ess < 10, ["seed", "method"]]
        lines += ["", f"Selected counts: {counts}; {consistency}. DEV-guard used for {guard_count}/5 seeds.",
                  "Every selected model is the lowest finite score within its recorded selector variant; "
                  "the tie break is alphabetical method name.",
                  f"Raw DEV weight ESS < 10/63: {len(low_raw)} of 30 candidate fits; "
                  f"DEV-guard ESS < 10/63: {len(low_guard)} of 30.",
                  f"Invalid raw DEV weights/scores: {len(invalid_raw)} of 30; "
                  + ("; ".join(f"seed {int(r.seed)} {r.method} ({r.anomalies})" for r in invalid_raw.itertuples())
                     if len(invalid_raw) else "none") + ".",
                  f"[Five selected prediction curves](selected_curves/{path.name})", ""]
    lines += ["## Interpretation", "",
              "The selected method uses a single checkpoint over cuts 1–315 for each direction and seed. "
              "Negative DEV scores can occur because the covariance correction is signed. "
              "Low ESS or invalid raw MLP weights indicate unstable importance weighting; "
              "DEV-guard uses a regularized logistic density estimator when any of the six raw scores "
              "for the same direction and seed is invalid. The oracle is a target-label upper bound "
              "and is not part of model selection.", ""]
    (a.root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(a.root / "RESULTS.md")


if __name__ == "__main__":
    main()
