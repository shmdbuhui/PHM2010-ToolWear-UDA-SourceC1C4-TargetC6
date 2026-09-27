"""Summarize locked nonnegative-increment predictions after target-only scoring."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_nonnegative_increment import DEFAULT_OUT, pairs, sha


BASELINE = Path("artifacts/full_1_315_baseline_zscore_20260925/per_seed_metrics_full_1_315.csv")
RIDGE = Path("artifacts/frozen_ridge_probe_audit_20260926/target_metrics.csv")
PAVA_SCORED = Path("artifacts/f0_pava_offline_20260926/per_seed_scored")


def render_direction_plots(out):
    """Redraw true VB and the saved five-seed F0+PAVA mean only."""
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    manifest_path = plot_dir / "manifest.json"
    previous = ({row["direction"]: row for row in json.loads(manifest_path.read_text(encoding="utf-8"))}
                if manifest_path.exists() else {})
    records = []
    for source, target, _ in pairs()[::5]:
        direction = f"{source}_to_{target}"
        truth = None
        predictions = []
        for seed in range(42, 47):
            path = PAVA_SCORED / f"{direction}_seed_{seed}.csv"
            frame = pd.read_csv(path, usecols=["cut_index", "true_vb", "pava_pred"])
            if frame.cut_index.tolist() != list(range(1, 316)):
                raise ValueError(f"Incomplete cut range: {path}")
            values = frame[["true_vb", "pava_pred"]].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite curve: {path}")
            if truth is None:
                truth = values[:, 0]
            elif not np.array_equal(truth, values[:, 0]):
                raise ValueError(f"True VB differs across seeds: {path}")
            predictions.append(values[:, 1])
        predicted = np.mean(np.stack(predictions), axis=0)
        if direction in previous:
            if previous[direction]["xlim"] != [1, 315]:
                raise ValueError(f"Previous x limits changed: {direction}")
            low, high = previous[direction]["ylim"]
        else:
            low = float(min(truth.min(), predicted.min()))
            high = float(max(truth.max(), predicted.max()))
            pad = max((high - low) * .05, 1.0)
            low, high = low - pad, high + pad
        if min(truth.min(), predicted.min()) < low or max(truth.max(), predicted.max()) > high:
            raise ValueError(f"Existing y limits would clip a curve: {direction}")
        fig, ax = plt.subplots(figsize=(9, 6.6))
        ax.set_box_aspect(0.70)
        ax.plot(range(1, 316), truth, color="black", linewidth=2, label="True VB")
        ax.plot(range(1, 316), predicted, color="#d04b3f", linewidth=1.55, label="F0 + PAVA")
        ax.set_xlim(1, 315)
        ax.set_ylim(low, high)
        ax.set_title(f"{source.upper()}→{target.upper()}")
        ax.set_xlabel("Cut index")
        ax.set_ylabel("VB")
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) != 2:
            raise AssertionError("Expected exactly two plotted curves")
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.subplots_adjust(left=.11, right=.97, bottom=.12, top=.88)
        fig.canvas.draw()
        box = ax.get_window_extent(fig.canvas.get_renderer())
        box_width_height_ratio = float(box.width / box.height)
        if not 1.38 <= box_width_height_ratio <= 1.48:
            raise AssertionError(f"Wrong axes box aspect: {box_width_height_ratio}")
        name = f"{source.upper()}_to_{target.upper()}.png"
        destination = plot_dir / name
        fig.savefig(destination, dpi=170)
        plt.close(fig)
        records.append({"direction": direction, "path": str(destination.resolve()),
                        "sha256": sha(destination), "axes_width_height_ratio": box_width_height_ratio,
                        "xlim": [1, 315], "ylim": [low, high],
                        "curves": ["True VB", "F0 + PAVA"]})
    if len(records) != 6:
        raise AssertionError("Expected exactly six independent direction figures")
    (plot_dir / "six_directions.png").unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plots-only", action="store_true", help="Redraw six figures without recalculating metrics")
    args = parser.parse_args()
    out = args.out
    if args.plots_only:
        render_direction_plots(out)
        readme = out / "README.md"
        old_line = "- `plots/six_directions.png`: five-seed mean primary curves with one standard deviation bands."
        prior_line = "- `plots/C1_to_C4.png` through `plots/C6_to_C4.png`: six separate five-seed mean primary-curve figures with one standard deviation bands."
        new_line = "- `plots/C1_to_C4.png` through `plots/C6_to_C4.png`: six separate True VB and five-seed mean F0 + PAVA figures."
        content = readme.read_text(encoding="utf-8")
        if old_line in content:
            readme.write_text(content.replace(old_line, new_line), encoding="utf-8")
        elif prior_line in content:
            readme.write_text(content.replace(prior_line, new_line), encoding="utf-8")
        elif new_line not in content:
            raise ValueError("Unexpected README plot description")
        return
    metrics = pd.read_csv(out / "per_seed_metrics.csv")
    curves = pd.read_csv(out / "target_predictions_scored.csv")
    increments = pd.read_csv(out / "increment_distribution.csv")
    drift = pd.read_csv(out / "cumulative_error_drift.csv")
    if len(metrics) != 720 or len(curves) != 56700 or len(increments) != 60 or len(drift) != 180:
        raise ValueError("Expected complete 30-pair, four-group, two-curve where applicable result")
    primary = metrics[metrics.primary].copy()
    if len(primary) != 480 or primary.groupby(["source", "target", "seed", "group"]).size().ne(4).any():
        raise ValueError("Every group needs full and three segment metrics")
    summary = primary.groupby(["source", "target", "group", "stage"], as_index=False).agg(
        n_seeds=("seed", "nunique"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        signed_bias_mean=("signed_bias", "mean"))
    summary.to_csv(out / "direction_segment_summary.csv", index=False)

    original = pd.read_csv(BASELINE)
    reference = original[original.method.isin(["source_only", "daregram"])][
        ["source", "target", "seed", "method", "R2", "MAE", "RMSE"]].copy()
    reference["reference"] = reference.method.map({"source_only": "source_only", "daregram": "dare_original"})
    reference = reference.drop(columns="method")
    ridge = pd.read_csv(RIDGE)
    ridge = ridge[(ridge.method == "daregram") & (ridge["head"] == "probe")][
        ["source", "target", "seed", "R2", "MAE", "RMSE"]].copy()
    ridge["reference"] = "dare_ridge"
    f0 = primary[(primary.stage == "full") & (primary.group == "F0")][
        ["source", "target", "seed", "R2", "MAE", "RMSE"]].copy()
    f0["reference"] = "F0_continued"
    reference = pd.concat([reference, ridge, f0], ignore_index=True)
    if len(reference) != 120:
        raise ValueError("Missing source-only, DARE original, Ridge, or F0 reference")
    full = primary[primary.stage == "full"]
    paired = full.merge(reference, on=["source", "target", "seed"], suffixes=("", "_ref"), validate="many_to_many")
    if len(paired) != 480:
        raise ValueError("Paired comparator count is wrong")
    paired["MAE_gain"] = paired.MAE_ref - paired.MAE
    paired["RMSE_gain"] = paired.RMSE_ref - paired.RMSE
    paired["R2_gain"] = paired.R2 - paired.R2_ref
    paired["MAE_win"] = paired.MAE_gain > 0
    paired["RMSE_win"] = paired.RMSE_gain > 0
    paired.to_csv(out / "paired_reference_comparison.csv", index=False)
    comparison = paired.groupby(["source", "target", "group", "reference"], as_index=False).agg(
        n_seeds=("seed", "nunique"), MAE_gain_mean=("MAE_gain", "mean"),
        RMSE_gain_mean=("RMSE_gain", "mean"), R2_gain_mean=("R2_gain", "mean"),
        MAE_wins=("MAE_win", "sum"), RMSE_wins=("RMSE_win", "sum"))
    comparison.to_csv(out / "direction_reference_summary.csv", index=False)

    physical = []
    for keys, group in curves.groupby(["source", "target", "seed", "group", "curve"], sort=False):
        group = group.sort_values("cut_index")
        if group.cut_index.tolist() != list(range(1, 316)):
            raise ValueError("Missing or shuffled cut in scored predictions")
        p = group.pred_vb.to_numpy(float)
        y = group.true_vb.to_numpy(float)
        late_slope = float(np.polyfit(np.arange(211, 316), p[210:], 1)[0])
        truth_late_slope = float(np.polyfit(np.arange(211, 316), y[210:], 1)[0])
        changes = np.diff(p)
        physical.append(dict(zip(("source", "target", "seed", "group", "curve"), keys),
                             decline_count=int((changes < 0).sum()),
                             max_decline_vb=float(np.maximum(-changes, 0).max()),
                             late_pred_slope_vb_per_cut=late_slope,
                             late_true_slope_vb_per_cut=truth_late_slope,
                             late_slope_gap=late_slope - truth_late_slope,
                             late_pred_change=float(p[-1] - p[210]),
                             late_true_change=float(y[-1] - y[210]),
                             late_signed_bias=float(np.mean(p[210:] - y[210:]))))
    physical = pd.DataFrame(physical)
    physical.to_csv(out / "physical_curve_audit.csv", index=False)
    phys_summary = physical.groupby(["source", "target", "group", "curve"], as_index=False).agg(
        decline_count_mean=("decline_count", "mean"),
        late_pred_slope_mean=("late_pred_slope_vb_per_cut", "mean"),
        late_true_slope_mean=("late_true_slope_vb_per_cut", "mean"),
        late_signed_bias_mean=("late_signed_bias", "mean"))
    phys_summary.to_csv(out / "physical_direction_summary.csv", index=False)
    if physical[physical.curve == "c_vb"].decline_count.ne(0).any():
        raise AssertionError("Cumulative head has negative step")

    render_direction_plots(out)

    lines = ["# Nonnegative increment experiment", "",
             "Six directed transfers × seeds 42–46 × F0–F3; all target cuts 1–315.",
             "F0/F1 primary endpoint: direct head `p`; F2/F3 primary endpoint: cumulative `c`.",
             "The alternative `p` curve for F2/F3 is diagnostic and was not selected using target scores.",
             "F0 replay gate and prediction lock were written before target-wear scoring.", "",
             "## Full-lifecycle primary endpoint (five-seed means)", "",
             "| Direction | Group | R² | MAE | RMSE | Late signed bias |",
             "|---|---|---:|---:|---:|---:|"]
    full_summary = summary[summary.stage == "full"]
    late_summary = summary[summary.stage == "late"]
    for row in full_summary.itertuples():
        bias = late_summary[(late_summary.source == row.source) &
                            (late_summary.target == row.target) &
                            (late_summary.group == row.group)].signed_bias_mean.iloc[0]
        lines.append(f"| {row.source}→{row.target} | {row.group} | {row.R2_mean:.3f} | "
                     f"{row.MAE_mean:.2f} | {row.RMSE_mean:.2f} | {bias:+.2f} |")
    lines += ["", "## Files", "",
              "- `gate.json`: 30 original DARE head replay checks.",
              "- `prediction_lock.json`: checkpoint/config/unscored-prediction hashes before target labels.",
              "- `per_seed_metrics.csv`, `direction_segment_summary.csv`: full/early/middle/late R², MAE, RMSE, bias.",
              "- `increment_distribution.csv`: nonnegative increment quantiles per run.",
              "- `cumulative_error_drift.csv`: late versus early absolute error and error slopes.",
              "- `physical_curve_audit.csv`: decline counts, late prediction and truth slopes.",
              "- `paired_reference_comparison.csv`, `direction_reference_summary.csv`: paired seed comparisons with F0, source-only, DARE original and DARE+Ridge.",
              "- `target_predictions_scored.csv`: both direct and cumulative curves with target wear, read only after lock.",
              "- `plots/C1_to_C4.png` through `plots/C6_to_C4.png`: six separate True VB and five-seed mean F0 + PAVA figures.",
              ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
