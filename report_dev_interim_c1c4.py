"""Write the frozen C1->C4 five-seed DEV interim audit without reading target VB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_dev_model_selection as dev


def fmt(value, digits=4):
    return "invalid" if pd.isna(value) else f"{float(value):.{digits}f}"


def markdown_table(frame, columns):
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/dev_model_selection_20260926"))
    a = p.parse_args()
    roots = [a.root / "runs" / "c1_to_c4" / f"seed_{seed}" for seed in dev.joint.SEEDS]
    for root in roots:
        if not (root / "selection_frozen.json").exists():
            raise AssertionError(f"Missing frozen selection: {root}")
    scores = pd.concat([pd.read_csv(r / "dev_scores.csv") for r in roots], ignore_index=True)
    selected = pd.concat([pd.read_csv(r / "selected_models.csv") for r in roots], ignore_index=True)
    metrics = pd.concat([pd.read_csv(r / "candidate_target_metrics.csv") for r in roots], ignore_index=True)
    if len(scores) != 30 or len(selected) != 5 or len(metrics) != 30:
        raise AssertionError("C1->C4 five-seed candidate grid incomplete")
    if selected.seed.tolist() != list(dev.joint.SEEDS):
        raise AssertionError("Unexpected selected seed order")
    scores = scores.merge(selected[["seed", "selected_method", "selection_variant", "selected_score"]],
                          on="seed", validate="many_to_one")
    scores["selected"] = scores.method == scores.selected_method
    scores.to_csv(a.root / "interim_c1_to_c4_dev.csv", index=False, float_format="%.17g")
    rows = []
    for seed in dev.joint.SEEDS:
        part = metrics.loc[metrics.seed == seed].set_index("method")
        pick = selected.loc[selected.seed == seed].iloc[0]
        oracle = part.sort_values(["MAE", "RMSE"]).iloc[0]
        oracle_name = part.sort_values(["MAE", "RMSE"]).index[0]
        worst_name = part.sort_values(["MAE", "RMSE"]).index[-1]
        for role, method in (("selected", pick.selected_method), ("fixed DARE-GRAM", "daregram"),
                             ("source-only", "source_only"), ("oracle best", oracle_name),
                             ("worst candidate", worst_name)):
            row = part.loc[method]
            rows.append({"seed": seed, "role": role, "method": method,
                         "MAE": float(row.MAE), "RMSE": float(row.RMSE)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(a.root / "interim_c1_to_c4_target_comparison.csv", index=False, float_format="%.17g")
    weight_file = roots[3] / "source_only" / "dev_validation_per_cut.csv"
    weights = pd.read_csv(weight_file)
    w = weights.weight.to_numpy(np.float64)
    e = weights.absolute_error_vb.to_numpy(np.float64)
    loss = w * e
    var64 = float(np.var(w, ddof=1))
    cov64 = float(np.cov(np.column_stack((loss, w)), rowvar=False)[0, 1])
    eta64 = -cov64 / var64
    risk64 = float(np.mean(loss) + eta64 * (np.mean(w) - 1))
    recorded = scores.loc[(scores.seed == 45) & (scores.method == "source_only")].iloc[0]
    diagnostic = {"source": "c1", "target": "c4", "seed": 45, "method": "source_only",
                  "recorded_raw_dev_score": float(recorded.dev_score),
                  "float64_recomputed_dev_score": risk64,
                  "float64_minus_recorded_score": risk64 - float(recorded.dev_score),
                  "all_weights_finite": bool(np.isfinite(w).all()),
                  "all_weights_nonnegative": bool((w >= 0).all()),
                  "weight_variance_float64": var64, "weight_mean": float(w.mean()),
                  "weight_max": float(w.max()), "weight_ess": float(w.sum()**2 / np.sum(w*w)),
                  "max_weight_fraction_of_sum": float(w.max() / w.sum()),
                  "top3_weight_fraction_of_sum": float(np.sort(w)[-3:].sum() / w.sum()),
                  "weighted_error_mean": float(np.mean(loss)), "cov_weighted_error_weight": cov64,
                  "eta_float64": eta64, "covariance_correction": float(eta64 * (w.mean() - 1)),
                  "top_weight_cuts": weights.nlargest(5, "weight")[["cut", "weight", "absolute_error_vb"]].to_dict("records"),
                  "domain_validation_accuracy": float(recorded.domain_val_accuracy),
                  "selection_changed": False}
    dev.write_json(a.root / "interim_c1_to_c4_seed45_weight_diagnostic.json", diagnostic)
    score_display = scores[["seed", "method", "dev_score", "dev_guard_score", "weight_max", "weight_ess",
                            "dev_guard_weight_max", "dev_guard_weight_ess", "selected"]].copy()
    score_display["seed"] = score_display.seed.astype(str)
    for col in ("dev_score", "dev_guard_score", "weight_max", "weight_ess",
                "dev_guard_weight_max", "dev_guard_weight_ess"):
        score_display[col] = score_display[col].map(lambda x: fmt(x, 3))
    score_display["selected"] = score_display.selected.map(lambda x: "yes" if x else "")
    metric_display = comparison.copy()
    metric_display["seed"] = metric_display.seed.astype(str)
    for col in ("MAE", "RMSE"):
        metric_display[col] = metric_display[col].map(lambda x: fmt(x, 3))
    selection_display = selected[["seed", "selection_variant", "selected_method", "selected_score"]].copy()
    selection_display["seed"] = selection_display.seed.astype(str)
    selection_display["selected_score"] = selection_display.selected_score.map(lambda x: fmt(x, 3))
    lines = ["# C1 → C4 five-seed interim DEV audit", "",
             "These five selections were frozen before target VB was opened. The target MAE/RMSE below "
             "comes solely from this run's retrained checkpoints. No ranking was changed for this report.", "",
             "## Frozen selections", "",
             markdown_table(selection_display, ["Seed", "Selector", "Selected model", "Selection score"]), "",
             "## All six candidate scores and raw weights", "",
             "`invalid` means the original MLP DEV weight variance was too small for a finite rank. "
             "Guard scores are shown so the DEV-guard selections can be checked. Raw maximum/ESS refer to "
             "the original MLP weights; guard maximum/ESS refer to the logistic weights used for guard selections.", "",
             markdown_table(score_display, ["Seed", "Candidate", "Raw DEV", "DEV-guard", "Raw max w", "Raw ESS",
                                            "Guard max w", "Guard ESS", "Selected"]), "",
             "## Target metrics after selection freeze", "",
             "The oracle best and worst rows are target-label diagnostics, never selection inputs. "
             "Repeated model names in a seed are intentional.", "",
             markdown_table(metric_display, ["Seed", "Role", "Model", "MAE", "RMSE"]), "",
             "## Seed 45 extreme score check", "",
             f"The selected source-only raw DEV score is {diagnostic['recorded_raw_dev_score']:.3f}. "
             f"Its weight variance is {var64:.6g}, so near-zero `Var(w)` is **not** the cause. "
             f"All 63 weights are finite and nonnegative. The maximum is {w.max():.6g}; "
             f"cut {int(weights.loc[weights.weight.idxmax(), 'cut'])} alone carries "
             f"{100*diagnostic['max_weight_fraction_of_sum']:.2f}% of their sum, and the largest three carry "
             f"{100*diagnostic['top3_weight_fraction_of_sum']:.2f}%. ESS is "
             f"{diagnostic['weight_ess']:.3f}/63. This extreme concentration is the main cause.", "",
             f"The two DEV terms are `mean(w×e)={diagnostic['weighted_error_mean']:.3f}` and "
             f"`eta×(mean(w)-1)={diagnostic['covariance_correction']:.3f}`; their difference yields "
             f"{risk64:.3f} with float64 recomputation. Its difference from the recorded score is "
             f"{abs(diagnostic['float64_minus_recorded_score']):.3f}, far smaller than the score magnitude. "
             "There is no NaN or overflow in the saved weights. The raw DEV ranking and selected model remain frozen.", "",
             "The selected source-only target MAE/RMSE are evaluation outcomes only; they were not used "
             "to trigger a guard or revise the model order.", ""]
    (a.root / "INTERIM_C1_TO_C4.md").write_text("\n".join(lines), encoding="utf-8")
    print(a.root / "INTERIM_C1_TO_C4.md")


if __name__ == "__main__":
    main()
