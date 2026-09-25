"""Post-hoc label-range and persistent-underestimation audit; no model changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_zscore_stage_errors import labels, sha


TOOLS = ("c1", "c4", "c6")
METHODS = ("source_only", "daregram")
SEEDS = range(42, 47)
SCOPES = {
    "full_cuts_1_315": ((1, 315), (("early", 1, 105), ("middle", 106, 210), ("late", 211, 315))),
    "c6_suffix_cuts_95_315": ((95, 315), (("early", 95, 168), ("middle", 169, 241), ("late", 242, 315))),
}
COLORS = {"source_only": "#2364a0", "daregram": "#d47b24"}
DETAIL_METRICS = ("MAE", "RMSE", "signed_bias", "under_count", "longest_under_run")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_info(mask: np.ndarray, first_cut: int) -> tuple[int, int | str, int | str]:
    best_len, best_start, best_end = 0, "", ""
    run_start, run_len = 0, 0
    for offset, value in enumerate(mask):
        if value:
            if run_len == 0:
                run_start = offset
            run_len += 1
            if run_len > best_len:
                best_len, best_start, best_end = run_len, first_cut + run_start, first_cut + offset
        else:
            run_len = 0
    return best_len, best_start, best_end


def group_metrics(y: np.ndarray, pred: np.ndarray, member: np.ndarray, first_cut: int) -> dict:
    valid = member & np.isfinite(y) & np.isfinite(pred)
    n = int(valid.sum())
    if n == 0:
        return {"n": "不适用", "status": "not_applicable_no_cuts", "MAE": "", "RMSE": "",
                "signed_bias": "", "under_count": "", "longest_under_run": "",
                "longest_under_first_cut": "", "longest_under_last_cut": ""}
    error = pred[valid] - y[valid]
    under = valid & (pred < y)
    longest, start, end = run_info(under, first_cut)
    return {"n": n, "status": "defined", "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "signed_bias": float(error.mean()), "under_count": int(under.sum()),
            "longest_under_run": longest, "longest_under_first_cut": start,
            "longest_under_last_cut": end}


def make_plot(path: Path, source: str, target: str, scope: str, y: np.ndarray,
              prediction: dict, source_max: float, first_above: int | None,
              common_limits: dict) -> None:
    (first, last), stages = SCOPES[scope]
    cuts = np.arange(first, last + 1)
    yy = y[first - 1:last]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    axes[0].plot(cuts, yy, color="black", lw=2.0, label="true VB")
    axes[0].axhline(source_max, color="#96348a", ls="-.", lw=1.1,
                    label=f"source training max {source_max:.2f}")
    for method in METHODS:
        matrix = np.stack([prediction[source, target, seed, method][first - 1:last]
                           for seed in SEEDS])
        for line in matrix:
            axes[0].scatter(cuts, line, s=4, alpha=0.07, color=COLORS[method], linewidths=0)
        mean_pred = matrix.mean(axis=0)
        axes[0].plot(cuts, mean_pred, color=COLORS[method], lw=1.5,
                     label=f"{method} five-seed mean")
        error = matrix - yy[None, :]
        axes[1].plot(cuts, error.mean(axis=0), color=COLORS[method], lw=1.4,
                     label=f"{method} mean signed error")
        pooled_squared = (error ** 2).sum(axis=0)
        axes[2].plot(cuts, 100 * pooled_squared / pooled_squared.sum(),
                     color=COLORS[method], lw=1.4,
                     label=f"{method} cut share of five-seed total SSE")
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[0].set(ylabel="VB", ylim=common_limits["wear"])
    axes[1].set(ylabel="prediction − truth", ylim=common_limits["error"])
    axes[2].set(xlabel="target cut index", ylabel="share of total SSE (%)",
                ylim=common_limits["share_percent"])
    boundaries = (stages[0][2] + 0.5, stages[1][2] + 0.5)
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="#666666", ls="--", lw=0.9)
        if first_above is not None and first <= first_above <= last:
            ax.axvline(first_above - 0.5, color="#b63737", ls=":", lw=1.3,
                       label=f"first target cut above source max: {first_above}" if ax is axes[0] else None)
        ax.set_xlim(first, last)
        ax.grid(alpha=0.15)
        ax.legend(loc="best", frameon=False, fontsize=8)
    if first_above is None:
        axes[0].text(0.99, 0.02, "No target cut above source max",
                     transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9)
    fig.suptitle(f"{source.upper()} → {target.upper()} | {scope} | original cut order; no smoothing")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path,
                        default=Path("artifacts/stage_error_analysis_zscore_20260925"))
    parser.add_argument("--original-root", type=Path, default=Path("artifacts/five_seed_paired"))
    parser.add_argument("--full-c6-root", type=Path,
                        default=Path("artifacts/norm_comparison_20260925/zscore"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path,
                        default=Path("artifacts/late_underestimation_extrapolation_zscore_20260925"))
    args = parser.parse_args()
    stage, original, full_c6, raw, out = (p.resolve() for p in
                                          (args.stage_root, args.original_root,
                                           args.full_c6_root, args.raw_root, args.out_root))
    if out.exists() or any(out == p or p in out.parents for p in (stage, original, full_c6)):
        parser.error("Output must be a new independent directory")
    y = {}
    source_info = {}
    for tool in TOOLS:
        y[tool], _, source_info[tool] = labels(raw, tool)
        if source_info[tool]["cuts_1_315_with_label"] != 315:
            raise ValueError(f"Unexpected missing labels: {tool}")
    provenance = read_csv(stage / "checkpoint_prediction_provenance.csv")
    if len(provenance) != 60:
        raise ValueError("Expected 60 original checkpoint/prediction records")
    pred = {}
    verified = []
    for row in provenance:
        source, target, method, seed = row["source"], row["target"], row["method"], int(row["seed"])
        if not (source in TOOLS and target in TOOLS and source != target and method in METHODS and seed in SEEDS):
            raise ValueError("Invalid provenance identity")
        config = json.loads(Path(row["config"]).read_text(encoding="utf-8"))
        checkpoint = Path(row["checkpoint"])
        if sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash changed: {checkpoint}")
        if (config["source"], config["target"], config["seed"], config["method"]) != (
                source, target, seed, method):
            raise ValueError(f"Config/checkpoint identity mismatch: {checkpoint}")
        source_wear = raw / f"{source}_wear.csv"
        target_wear = raw / f"{target}_wear.csv"
        if (sha(source_wear) != config["source_wear_sha256"] or
            str(source_wear) != str(Path(row["source_wear_csv"])) or
            str(target_wear) != str(Path(row["target_wear_csv"]))):
            raise ValueError(f"Actual source-training label source differs: {checkpoint}")
        prediction_path = Path(row["prediction_file"])
        if not prediction_path.is_file() or sha(prediction_path) != row["prediction_file_sha256"]:
            raise ValueError(f"Prediction file hash changed: {prediction_path}")
        frame = pd.read_csv(prediction_path)
        if frame.cut_index.astype(int).tolist() != list(range(1, 316)):
            raise ValueError(f"Not full-cut predictions: {prediction_path}")
        if not np.allclose(frame.true_vb.to_numpy(dtype=np.float64), y[target], rtol=0, atol=1e-6):
            raise ValueError(f"Prediction label/cut mismatch: {prediction_path}")
        per_cut_path = stage / "per_cut" / f"{source}_to_{target}_{method}_cuts_1_315.csv"
        detail = read_csv(per_cut_path)
        if [int(r["cut_index"]) for r in detail] != list(range(1, 316)):
            raise ValueError(f"Per-cut index mismatch: {per_cut_path}")
        if not np.allclose([float(r["true_vb"]) for r in detail], y[target], rtol=0, atol=1e-6):
            raise ValueError(f"Per-cut label mismatch: {per_cut_path}")
        values = np.asarray([float(r[f"seed_{seed}_pred_vb"]) for r in detail])
        if not np.allclose(values, frame.pred_vb.to_numpy(dtype=np.float64), rtol=0, atol=1e-9):
            raise ValueError(f"Per-cut prediction does not match source file: {per_cut_path}")
        pred[source, target, seed, method] = values
        verified.append({"source": source, "target": target, "seed": seed, "method": method,
                         "checkpoint": str(checkpoint), "checkpoint_sha256": row["checkpoint_sha256"],
                         "prediction_file": str(prediction_path),
                         "prediction_file_sha256": row["prediction_file_sha256"],
                         "source_training_label_file": str(source_wear),
                         "source_training_label_sha256": sha(source_wear),
                         "target_label_file": str(target_wear),
                         "target_label_sha256": sha(target_wear),
                         "all_cut_and_label_matches": True})
    if len(pred) != 60:
        raise ValueError("Missing or duplicate direction/seed/method predictions")
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    write_csv(out / "verified_provenance_full_cuts_1_315.csv", verified)
    range_rows, flags = [], []
    for source in TOOLS:
        source_max = float(np.max(y[source]))
        for target in TOOLS:
            if source == target:
                continue
            above = y[target] > source_max
            first_above = int(np.flatnonzero(above)[0] + 1) if above.any() else None
            range_rows.append({"source": source, "target": target,
                               "source_training_label_file": source_info[source]["path"],
                               "source_training_label_sha256": source_info[source]["sha256"],
                               "target_label_file": source_info[target]["path"],
                               "target_label_sha256": source_info[target]["sha256"],
                               "source_training_label_max": source_max,
                               "target_true_max": float(np.max(y[target])),
                               "target_cuts_above_source_max": int(above.sum()),
                               "first_target_cut_above_source_max": first_above if first_above is not None else "not_applicable"})
            for cut in range(1, 316):
                flags.append({"source": source, "target": target, "cut_index": cut,
                              "true_vb": float(y[target][cut - 1]),
                              "source_training_label_max": source_max,
                              "above_source_max": bool(above[cut - 1]),
                              "first_target_cut_above_source_max": first_above if first_above is not None else "not_applicable"})
    write_csv(out / "direction_label_ranges_full_cuts_1_315.csv", range_rows)
    write_csv(out / "per_cut_range_flags_full_cuts_1_315.csv", flags)
    range_lookup = {(r["source"], r["target"]): r for r in range_rows}

    detail_rows, maxima_rows, reconciliation = [], [], []
    prior_full = read_csv(stage / "per_seed_stage_metrics_full_cuts_1_315.csv")
    prior_suffix = read_csv(stage / "per_seed_stage_metrics_c6_suffix_cuts_95_315.csv")
    prior_metrics = {(r["source"], r["target"], int(r["seed"]), r["method"],
                     "full_cuts_1_315" if r["scope"] == "full_1_315" else "c6_suffix_cuts_95_315"):
                     float(r["RMSE"]) for r in prior_full + prior_suffix if r["segment"] == "all"}
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            source_max = float(range_lookup[source, target]["source_training_label_max"])
            for seed in SEEDS:
                for method in METHODS:
                    values = pred[source, target, seed, method]
                    target_true = y[target]
                    maximum = float(values.max())
                    maxima_rows.append({"source": source, "target": target, "seed": seed,
                                        "method": method, "scope": "full_cuts_1_315",
                                        "source_training_label_max": source_max,
                                        "target_true_max": float(target_true.max()),
                                        "target_true_max_cut": int(np.argmax(target_true) + 1),
                                        "prediction_max": maximum,
                                        "prediction_max_cut": int(np.argmax(values) + 1),
                                        "prediction_cuts_above_source_max": int((values > source_max).sum()),
                                        "target_cuts_above_source_max": int((target_true > source_max).sum()),
                                        "prediction_never_exceeds_source_max": bool(maximum <= source_max),
                                        "late_211_315_under_count": int((values[210:] < target_true[210:]).sum())})
                    for scope, (all_range, stages) in SCOPES.items():
                        if scope.startswith("c6_") and target != "c6":
                            continue
                        for segment, lo, hi in (("all", *all_range), *stages):
                            yy, pp = target_true[lo - 1:hi], values[lo - 1:hi]
                            masks = {"all_labels": np.ones(len(yy), dtype=bool),
                                     "within_source_max": yy <= source_max,
                                     "above_source_max": yy > source_max}
                            for group, mask in masks.items():
                                result = group_metrics(yy, pp, mask, lo)
                                detail_rows.append({"source": source, "target": target,
                                                    "seed": seed, "method": method, "scope": scope,
                                                    "segment": segment, "cut_first": lo, "cut_last": hi,
                                                    "range_group": group,
                                                    "source_training_label_max": source_max,
                                                    **result})
                        all_result = group_metrics(target_true[all_range[0] - 1:all_range[1]],
                                                   values[all_range[0] - 1:all_range[1]],
                                                   np.ones(all_range[1] - all_range[0] + 1, dtype=bool),
                                                   all_range[0])
                        old = prior_metrics[source, target, seed, method, scope]
                        if not np.isclose(all_result["RMSE"], old, rtol=0, atol=1e-10):
                            raise ValueError(f"Prior stage RMSE not reproduced: {source} {target} {seed} {method} {scope}")
                        original_folder = original / f"{source}_to_{target}" / f"seed_{seed}" / method
                        prior_file = (full_c6 / f"{source}_to_{target}" / f"seed_{seed}" / method / "metrics.json"
                                      if target == "c6" and scope == "full_cuts_1_315"
                                      else original_folder / "metrics.json")
                        original_rmse = float(json.loads(prior_file.read_text(encoding="utf-8"))["RMSE"])
                        difference = float(all_result["RMSE"] - original_rmse)
                        if abs(difference) > 0.001:
                            raise ValueError(f"Prior original RMSE differs materially: {prior_file}")
                        reconciliation.append({"source": source, "target": target, "seed": seed,
                                               "method": method, "scope": scope,
                                               "recomputed_from_full_cut_predictions_RMSE": all_result["RMSE"],
                                               "prior_stage_analysis_RMSE": old,
                                               "prior_original_metrics_file": str(prior_file.resolve()),
                                               "prior_original_RMSE": original_rmse,
                                               "recomputed_minus_original_RMSE": difference,
                                               "source_of_difference": "full-cut versus original suffix inference batching"
                                               if target == "c6" and scope == "c6_suffix_cuts_95_315" else "none"})
    write_csv(out / "per_seed_maxima_full_cuts_1_315.csv", maxima_rows)
    write_csv(out / "rmse_reconciliation_full_cuts_1_315_and_c6_suffix_cuts_95_315.csv", reconciliation)
    for scope in SCOPES:
        selected = [r for r in detail_rows if r["scope"] == scope]
        if selected:
            write_csv(out / f"per_seed_range_underestimation_{scope}.csv", selected)

    paired_rows, summaries, paired_summaries = [], [], []
    index = {(r["source"], r["target"], int(r["seed"]), r["scope"],
              r["segment"], r["range_group"], r["method"]): r for r in detail_rows}
    keys = sorted({k[:-1] for k in index})
    for key in keys:
        source, target, seed, scope, segment, group = key
        s, d = (index[(*key, method)] for method in METHODS)
        row = {"source": source, "target": target, "seed": seed, "scope": scope,
               "segment": segment, "range_group": group,
               "cut_first": s["cut_first"], "cut_last": s["cut_last"],
               "n": s["n"], "status": s["status"]}
        for metric in DETAIL_METRICS:
            row[f"source_only_{metric}"] = s[metric]
            row[f"daregram_{metric}"] = d[metric]
            row[f"delta_{metric}"] = float(d[metric]) - float(s[metric]) if s[metric] != "" and d[metric] != "" else ""
        paired_rows.append(row)
    for scope in SCOPES:
        scope_rows = [r for r in paired_rows if r["scope"] == scope]
        if scope_rows:
            write_csv(out / f"paired_seed_range_underestimation_{scope}.csv", scope_rows)
    group_keys = sorted({(r["source"], r["target"], r["scope"], r["segment"],
                          r["range_group"]) for r in detail_rows})
    for source, target, scope, segment, group in group_keys:
        for method in METHODS:
            subset = [r for r in detail_rows if (r["source"], r["target"], r["scope"],
                      r["segment"], r["range_group"], r["method"]) ==
                      (source, target, scope, segment, group, method)]
            row = {"source": source, "target": target, "scope": scope, "segment": segment,
                   "range_group": group, "method": method, "cut_first": subset[0]["cut_first"],
                   "cut_last": subset[0]["cut_last"], "n_per_seed": subset[0]["n"],
                   "n_seeds": len(subset),
                   "status": "not_applicable_no_cuts" if subset[0]["status"] == "not_applicable_no_cuts" else "defined"}
            for metric in DETAIL_METRICS:
                v = np.asarray([float(r[metric]) for r in subset if r[metric] != ""])
                row[f"{metric}_mean"] = float(v.mean()) if len(v) else ""
                row[f"{metric}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else ""
            summaries.append(row)
        subset = [r for r in paired_rows if (r["source"], r["target"], r["scope"],
                  r["segment"], r["range_group"]) == (source, target, scope, segment, group)]
        row = {"source": source, "target": target, "scope": scope, "segment": segment,
               "range_group": group, "cut_first": subset[0]["cut_first"],
               "cut_last": subset[0]["cut_last"], "n_per_seed": subset[0]["n"],
               "n_seeds": len(subset),
               "status": "not_applicable_no_cuts" if subset[0]["status"] == "not_applicable_no_cuts" else "defined"}
        for metric in DETAIL_METRICS:
            v = np.asarray([float(r[f"delta_{metric}"]) for r in subset if r[f"delta_{metric}"] != ""])
            row[f"delta_{metric}_mean"] = float(v.mean()) if len(v) else ""
            row[f"delta_{metric}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else ""
        paired_summaries.append(row)
    for scope in SCOPES:
        a = [r for r in summaries if r["scope"] == scope]
        b = [r for r in paired_summaries if r["scope"] == scope]
        if a:
            write_csv(out / f"range_underestimation_summary_{scope}.csv", a)
            write_csv(out / f"paired_range_underestimation_summary_{scope}.csv", b)

    top_cuts, top_shares, late_coverage, per_seed_top = [], [], [], []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for scope, (scope_range, stages) in SCOPES.items():
                if scope.startswith("c6_") and target != "c6":
                    continue
                lo, hi = scope_range
                yy = y[target][lo - 1:hi]
                per_method_mse = {}
                for method in METHODS:
                    matrix = np.stack([pred[source, target, seed, method][lo - 1:hi] for seed in SEEDS])
                    square = (matrix - yy[None, :]) ** 2
                    per_cut_sse = square.sum(axis=0)
                    total_sse = float(per_cut_sse.sum())
                    order = np.argsort(-per_cut_sse, kind="stable")
                    per_method_mse[method] = square
                    top_shares.append({"source": source, "target": target, "scope": scope,
                                       "method": method, "n_seeds": 5, "n_cuts": len(yy),
                                       "total_squared_error_all_seeds": total_sse,
                                       "top_1_share": float(per_cut_sse[order[:1]].sum() / total_sse),
                                       "top_5_share": float(per_cut_sse[order[:5]].sum() / total_sse),
                                       "top_10_share": float(per_cut_sse[order[:10]].sum() / total_sse)})
                    for rank, offset in enumerate(order[:10], 1):
                        top_cuts.append({"source": source, "target": target, "scope": scope,
                                         "method": method, "rank": rank,
                                         "cut_index": lo + int(offset),
                                         "true_vb": float(yy[offset]),
                                         "sum_squared_error_across_5_seeds": float(per_cut_sse[offset]),
                                         "share_of_method_total_squared_error": float(per_cut_sse[offset] / total_sse),
                                         "mean_signed_error_across_5_seeds": float(
                                             (matrix[:, offset] - yy[offset]).mean())})
                    for j, seed in enumerate(SEEDS):
                        ordered = np.argsort(-square[j], kind="stable")
                        denominator = float(square[j].sum())
                        per_seed_top.append({"source": source, "target": target, "scope": scope,
                                             "method": method, "seed": seed,
                                             "top_1_share": float(square[j, ordered[:1]].sum() / denominator),
                                             "top_5_share": float(square[j, ordered[:5]].sum() / denominator),
                                             "top_10_share": float(square[j, ordered[:10]].sum() / denominator)})
                late_lo, late_hi = stages[-1][1:]
                start, stop = late_lo - lo, late_hi - lo + 1
                s2 = per_method_mse["source_only"][:, start:stop]
                d2 = per_method_mse["daregram"][:, start:stop]
                difference = s2.sum(axis=0) - d2.sum(axis=0)
                positive = np.maximum(difference, 0)
                gains = np.sort(positive)[::-1]
                late_coverage.append({"source": source, "target": target, "scope": scope,
                                      "late_cut_first": late_lo, "late_cut_last": late_hi,
                                      "n_late_cuts": len(difference),
                                      "cuts_daregram_lower_pooled_squared_error": int((difference > 0).sum()),
                                      "cuts_daregram_lower_mean_absolute_error": int((np.abs(
                                          np.stack([pred[source, target, seed, "daregram"][late_lo - 1:late_hi]
                                                    for seed in SEEDS]) - y[target][late_lo - 1:late_hi]).mean(axis=0) <
                                          np.abs(np.stack([pred[source, target, seed, "source_only"][late_lo - 1:late_hi]
                                                           for seed in SEEDS]) - y[target][late_lo - 1:late_hi]).mean(axis=0)).sum()),
                                      "source_only_late_total_SSE": float(s2.sum()),
                                      "daregram_late_total_SSE": float(d2.sum()),
                                      "net_late_SSE_reduction": float(difference.sum()),
                                      "top_1_cut_share_of_positive_SSE_reduction": float(gains[:1].sum() / positive.sum()) if positive.sum() else "",
                                      "top_5_cuts_share_of_positive_SSE_reduction": float(gains[:5].sum() / positive.sum()) if positive.sum() else "",
                                      "top_10_cuts_share_of_positive_SSE_reduction": float(gains[:10].sum() / positive.sum()) if positive.sum() else ""})
    for scope in SCOPES:
        for filename, rows in (("top_10_cuts_by_pooled_squared_error", top_cuts),
                               ("pooled_top_error_shares", top_shares),
                               ("per_seed_top_error_shares", per_seed_top),
                               ("late_improvement_coverage", late_coverage)):
            selected = [r for r in rows if r["scope"] == scope]
            if selected:
                write_csv(out / f"{filename}_{scope}.csv", selected)

    wear_values = np.concatenate([y[t] for t in TOOLS] + list(pred.values()) +
                                 [np.asarray([r["source_training_label_max"] for r in range_rows])])
    errors = np.concatenate([pred[s, t, seed, m] - y[t]
                             for s in TOOLS for t in TOOLS if s != t for seed in SEEDS for m in METHODS])
    max_share = max(100 * float(r["top_1_share"]) for r in top_shares)
    common_limits = {"wear": (float(wear_values.min()) - 5, float(wear_values.max()) + 5),
                     "error": (-1.05 * float(np.max(np.abs(errors))),
                               1.05 * float(np.max(np.abs(errors)))),
                     "share_percent": (0, max_share * 1.12)}
    for source, target in (("c1", "c6"), ("c6", "c4"), ("c6", "c1")):
        first_above = range_lookup[source, target]["first_target_cut_above_source_max"]
        first_above = int(first_above) if first_above != "not_applicable" else None
        scopes = ("full_cuts_1_315", "c6_suffix_cuts_95_315") if target == "c6" else ("full_cuts_1_315",)
        for scope in scopes:
            make_plot(out / "figures" / f"{source}_to_{target}_{scope}_range_underestimation_sse.png",
                      source, target, scope, y[target], pred,
                      float(range_lookup[source, target]["source_training_label_max"]),
                      first_above, common_limits)
    manifest = {"input_stage_analysis": str(stage), "original_experiment_root": str(original),
                "full_C6_prediction_root": str(full_c6), "raw_label_root": str(raw),
                "source_label_definition": "training code: mean of flute_1, flute_2, flute_3, converted to float32",
                "source_label_max_use": "post-hoc grouping only; no model, prediction, or checkpoint changes",
                "scope_segments": {k: {"all": list(v[0]), "segments": [list(z) for z in v[1]]}
                                   for k, v in SCOPES.items()},
                "metrics": "per-seed first; five-seed arithmetic mean and sample SD (ddof=1)",
                "longest_run": "consecutive original cut numbers satisfying both range group and prediction < truth",
                "top_cut_contribution": "sum of squared errors across five individual seeds at a cut / total five-seed squared error",
                "plot_limits": common_limits,
                "models_retrained": False, "prediction_values_changed": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed {len(detail_rows)} per-seed group rows and {len(reconciliation)} RMSE checks: {out}")


if __name__ == "__main__":
    main()
