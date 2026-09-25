"""Plot true wear against five-seed source-only and DARE-GRAM predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIRS = (("c1", "c4"), ("c1", "c6"), ("c4", "c1"),
         ("c4", "c6"), ("c6", "c1"), ("c6", "c4"))
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("source_only", "daregram")
COLORS = {"source_only": "#2563a6", "daregram": "#db7026"}
LABELS = {"source_only": "Source-only", "daregram": "DARE-GRAM"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/five_seed_paired/plots"))
    parser.add_argument("--evaluation-scope", choices=("legacy", "full"), default="legacy",
                        help="full requires cuts 1..315 for every target; legacy keeps C6 cuts 95..315")
    return parser.parse_args()


def load_pair(root: Path, source: str, target: str, scope: str):
    expected_cuts = list(range(95, 316)) if scope == "legacy" and target == "c6" else list(range(1, 316))
    predictions = {}
    truth = None
    for method in METHODS:
        series = []
        for seed in SEEDS:
            path = root / f"{source}_to_{target}" / f"seed_{seed}" / method / "predictions.csv"
            frame = pd.read_csv(path)
            if list(frame.columns) != ["cut_index", "true_vb", "pred_vb"]:
                raise ValueError(f"Unexpected CSV schema: {path}")
            if frame.cut_index.tolist() != expected_cuts or len(frame) != len(expected_cuts):
                raise ValueError(f"Unexpected evaluated cut list: {path}")
            y = frame.true_vb.to_numpy(dtype=np.float64)
            pred = frame.pred_vb.to_numpy(dtype=np.float64)
            if not np.isfinite(y).all() or not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite wear or prediction: {path}")
            if truth is None:
                truth = y
            elif not np.array_equal(truth, y):
                raise ValueError(f"True wear differs across methods or seeds: {path}")
            series.append(pred)
        values = np.stack(series)
        predictions[method] = {"mean": values.mean(axis=0), "sd": values.std(axis=0, ddof=1)}
    return np.asarray(expected_cuts), truth, predictions


def plot_one(ax, source, target, cuts, truth, predictions, summary_row, ylim, scope):
    ax.plot(cuts, truth, color="#242424", lw=2.25, label="True VB", zorder=4)
    for method in METHODS:
        mean = predictions[method]["mean"]
        sd = predictions[method]["sd"]
        color = COLORS[method]
        ax.fill_between(cuts, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0,
                        label=f"{LABELS[method]} ±1 seed SD", zorder=1)
        ax.plot(cuts, mean, color=color, lw=1.75, label=f"{LABELS[method]} mean", zorder=3)
    scope_text = "全程 1–315 评估" if scope == "full" else "Legacy evaluation range"
    ax.set(title=f"{source.upper()} → {target.upper()}  |  5-seed mean  |  {scope_text}",
           xlabel="Target cut index", ylabel="Flank wear (VB)")
    ax.set_xlim(cuts[0], cuts[-1])
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.text(0.012, 0.985,
            f"Mean per-seed R²: source-only {summary_row.source_only_R2_mean:.3f}  ·  "
            f"DARE-GRAM {summary_row.daregram_R2_mean:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3})


def main():
    args = arguments()
    if args.evaluation_scope == "full":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.result_root / "summary.csv")
    seed_metrics = pd.read_csv(args.result_root / "seed_metrics.csv")
    if len(summary) != 6:
        raise ValueError("Expected all six completed five-seed directions")
    data = {}
    target_ranges = {}
    for source, target in PAIRS:
        cuts, truth, predictions = load_pair(args.result_root, source, target, args.evaluation_scope)
        data[(source, target)] = (cuts, truth, predictions)
        boundaries = [truth.min(), truth.max()]
        for method in METHODS:
            mean, sd = predictions[method]["mean"], predictions[method]["sd"]
            boundaries.extend(((mean - sd).min(), (mean + sd).max()))
        low, high = target_ranges.get(target, (np.inf, -np.inf))
        target_ranges[target] = (min(low, min(boundaries)), max(high, max(boundaries)))
    for target, (low, high) in target_ranges.items():
        pad = (high - low) * 0.06
        target_ranges[target] = (low - pad, high + pad)

    manifest = {"seeds": list(SEEDS), "evaluation_scope": args.evaluation_scope,
                "evaluation_label": "全程 1–315 评估" if args.evaluation_scope == "full" else "旧评估口径",
                "prediction_line": "arithmetic mean of five per-seed predictions at each cut",
                "shaded_band": "sample standard deviation (ddof=1) of five predictions at each cut",
                "R2_annotation": "mean of five per-seed R2 values; not R2 of the mean prediction",
                "protocol": "transductive UDA for DARE-GRAM; source-only uses no target features in training",
                "directions": {}}
    fig_all, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    best_dir = args.out_dir / "best_r2"
    best_dir.mkdir(exist_ok=True)
    for ax_all, (source, target) in zip(axes.flat, PAIRS):
        cuts, truth, predictions = data[(source, target)]
        row = summary.loc[(summary.source == source) & (summary.target == target)]
        if len(row) != 1:
            raise ValueError(f"Missing summary row: {source}->{target}")
        row = next(row.itertuples(index=False))
        ylim = target_ranges[target]
        fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
        plot_one(ax, source, target, cuts, truth, predictions, row, ylim, args.evaluation_scope)
        ax.legend(loc="lower right", ncol=2, fontsize=8.7, framealpha=0.9)
        stem = f"{source}_to_{target}"
        fig.savefig(args.out_dir / f"{stem}.png", dpi=220)
        fig.savefig(args.out_dir / f"{stem}.pdf")
        plt.close(fig)
        plot_one(ax_all, source, target, cuts, truth, predictions, row, ylim, args.evaluation_scope)
        curve = pd.DataFrame({"cut_index": cuts, "true_vb": truth,
                              "source_only_mean_vb": predictions["source_only"]["mean"],
                              "source_only_seed_sd_vb": predictions["source_only"]["sd"],
                              "daregram_mean_vb": predictions["daregram"]["mean"],
                              "daregram_seed_sd_vb": predictions["daregram"]["sd"]})
        curve.to_csv(args.out_dir / f"{stem}_curves.csv", index=False, float_format="%.17g")
        manifest["directions"][stem] = {"evaluation_cuts": cuts.tolist(), "seed_count": 5,
                                         "source_only_R2_mean": float(row.source_only_R2_mean),
                                         "daregram_R2_mean": float(row.daregram_R2_mean)}

        # Also show the best-R² run for each method, selected independently.
        metrics_rows = seed_metrics.loc[(seed_metrics.source == source) &
                                        (seed_metrics.target == target)]
        best = {}
        for method in METHODS:
            best_row = metrics_rows.loc[metrics_rows[f"{method}_R2"].idxmax()]
            best_seed = int(best_row.seed)
            prediction = pd.read_csv(args.result_root / stem / f"seed_{best_seed}" /
                                     method / "predictions.csv")
            if prediction.cut_index.tolist() != cuts.tolist() or not np.array_equal(
                prediction.true_vb.to_numpy(), truth
            ):
                raise ValueError(f"Best-R² run uses different evaluation cuts or labels: {stem}, {method}")
            best[method] = {"seed": best_seed, "R2": float(best_row[f"{method}_R2"]),
                            "prediction": prediction.pred_vb.to_numpy(dtype=np.float64)}
        best_fig, best_ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
        best_ax.plot(cuts, truth, color="#242424", lw=2.25, label="True VB")
        for method in METHODS:
            item = best[method]
            best_ax.plot(cuts, item["prediction"], color=COLORS[method], lw=1.75,
                         label=f"{LABELS[method]} seed {item['seed']} (R²={item['R2']:.3f})")
        scope_text = "全程 1–315 评估" if args.evaluation_scope == "full" else "Legacy evaluation range"
        best_ax.set(title=f"{source.upper()} → {target.upper()}  |  Best R² run  |  {scope_text}",
                    xlabel="Target cut index", ylabel="Flank wear (VB)")
        best_ax.set_xlim(cuts[0], cuts[-1])
        low = min(truth.min(), *(item["prediction"].min() for item in best.values()))
        high = max(truth.max(), *(item["prediction"].max() for item in best.values()))
        pad = (high - low) * 0.06
        best_ax.set_ylim(low - pad, high + pad)
        best_ax.grid(alpha=0.22, linewidth=0.7)
        best_ax.legend(loc="lower right", framealpha=0.9)
        best_ax.text(0.012, 0.985, "Seeds selected independently; descriptive comparison",
                     transform=best_ax.transAxes, ha="left", va="top", fontsize=9,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3})
        best_fig.savefig(best_dir / f"{stem}.png", dpi=220)
        best_fig.savefig(best_dir / f"{stem}.pdf")
        plt.close(best_fig)
        pd.DataFrame({"cut_index": cuts, "true_vb": truth,
                      "source_only_best_pred_vb": best["source_only"]["prediction"],
                      "daregram_best_pred_vb": best["daregram"]["prediction"]}).to_csv(
                          best_dir / f"{stem}_curves.csv", index=False, float_format="%.17g")
        manifest["directions"][stem]["best_R2_seed"] = {
            method: {"seed": best[method]["seed"], "R2": best[method]["R2"]}
            for method in METHODS
        }
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig_all.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig_all.savefig(args.out_dir / "six_directions_overview.png", dpi=200)
    fig_all.savefig(args.out_dir / "six_directions_overview.pdf")
    plt.close(fig_all)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    print(f"Saved six mean/SD plots, six best-R2 plots, curves, and overview to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
