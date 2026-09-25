"""Frozen five-arm OOR-PGA factor decomposition; prepare before evaluate.

A original OOR, B cap original, C delayed pure ratio, D delayed capped,
E saved early-reference persistent gate. No model training or tuning.
"""

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

import analyze_zscore_stage_errors as stage
import run_full_1_315_baseline_protocol as baseline
import run_oor_pga_full_1_315 as original
import run_early_reference_persistent_oor_pga as persistent


DEFAULT_OUT = Path("artifacts/oor_pga_factor_diagnosis_full_1_315_20260925")
ARM_NAMES = ("A_original", "B_original_capped_1p5", "C_delayed_uncapped",
             "D_delayed_capped_1p5", "E_early_reference_persistent")
SEGMENTS = persistent.SEGS


def json_write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, float_format="%.17g")


def first_csv_row(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as f:
        return next(csv.DictReader(f))


def exact_m(source: str, target: str, old_root: Path, new_root: Path) -> dict:
    score = first_csv_row(old_root / "oor_scores" /
                          f"{source}_to_{target}_oor_score_full_1_315.csv")
    with (new_root / "trigger_summary_full_1_315.csv").open(encoding="utf-8", newline="") as f:
        new = next(r for r in csv.DictReader(f) if r["source"] == source and r["target"] == target)
    prior = float(score["source_pga_m"])
    current = float(new["m"])
    return {"old_m_literal": score["source_pga_m"], "new_m_literal": new["m"],
            "old_m": prior, "new_m": current, "m_exact_equal": prior == current,
            "m_absolute_difference": abs(prior - current)}


def input_locks(base_root: Path, old_root: Path, new_root: Path) -> tuple[pd.DataFrame, dict, dict]:
    audit, old_lock = persistent.locked_inputs(base_root, old_root)
    new_lock_path = new_root / "prediction_lock_before_target_labels.json"
    new_lock = stage.json_read(new_lock_path)
    if (stage.sha(new_root / "input_audit_lock.json") != new_lock["input_audit_lock_sha256"] or
        stage.sha(old_root / "prediction_lock_before_target_labels.json") !=
        new_lock["prior_OOR_prediction_lock_sha256"] or
        len(new_lock["unlabeled_prediction_sha256"]) != 60):
        raise ValueError("New persistent prediction lock mismatch")
    for name, digest in new_lock["unlabeled_prediction_sha256"].items():
        if stage.sha(new_root / "unlabeled_predictions" / name) != digest:
            raise ValueError(f"New frozen prediction changed: {name}")
    for name, digest in new_lock["gate_profile_sha256"].items():
        if stage.sha(new_root / "gate_profiles" / name) != digest:
            raise ValueError(f"New frozen gate changed: {name}")
    return audit, old_lock, new_lock


def factors(source: str, target: str, old_root: Path, new_root: Path) -> tuple[pd.DataFrame, dict]:
    old_path = old_root / "oor_scores" / f"{source}_to_{target}_oor_score_full_1_315.csv"
    new_path = new_root / "gate_profiles" / f"{source}_to_{target}_gate_full_1_315.csv"
    prior, current = pd.read_csv(old_path), pd.read_csv(new_path)
    if prior.cut_index.tolist() != baseline.CUTS or current.cut_index.tolist() != baseline.CUTS:
        raise ValueError(f"Factor cut order changed: {source}->{target}")
    if not np.allclose(prior.oor_rate_48.to_numpy(float), current.r_t.to_numpy(float),
                       rtol=0, atol=1e-15):
        raise ValueError(f"OOR rates disagree: {source}->{target}")
    identity = exact_m(source, target, old_root, new_root)
    tau = prior.tau.to_numpy(float)
    if not np.allclose(tau, current.tau.to_numpy(float), rtol=0, atol=1e-15):
        raise ValueError("Lifecycle tau changed")
    old_cut = str(prior.trigger_cut.iloc[0])
    new_cut = str(current.trigger_t0_cut.iloc[0])
    m = identity["old_m"]
    A = prior.multiplicative_factor.to_numpy(float)
    E = current.factor_F_t.to_numpy(float)
    B = np.minimum(A, 1.5)
    C = np.ones(315)
    reason = "triggered"
    if new_cut == "not_triggered":
        reason = "new_gate_not_triggered"
    elif not np.isfinite(m) or m <= 0:
        reason = "nonpositive_or_invalid_source_m"
    else:
        idx = int(new_cut) - 1
        t0 = tau[idx]
        if t0 <= 0:
            raise ValueError("Nonpositive new trigger tau")
        active = np.arange(315) > idx
        C[active] = np.power(tau[active] / t0, m)
    D = np.minimum(C, 1.5)
    if not identity["m_exact_equal"]:
        # Keep A and E untouched; B-D use one common canonical old/source m.
        reason += ";m_disagreement_recorded_common_old_m_used_for_C_D"
    frame = pd.DataFrame({"cut_index": baseline.CUTS, "tau": tau,
                          "r_t": prior.oor_rate_48.to_numpy(float),
                          "new_early_reference_b": current.early_reference_b.to_numpy(float),
                          "new_smoothed_s_t": current.causal_median_s_t.to_numpy(float),
                          "new_excess_d_t": current.excess_d_t.to_numpy(float),
                          "new_weight_w_t": current.weight_w_t.to_numpy(float),
                          "old_trigger_cut": old_cut, "new_confirmed_trigger_cut": new_cut,
                          "common_source_m": m,
                          "factor_A_original": A, "factor_B_original_capped_1p5": B,
                          "factor_C_delayed_uncapped": C, "factor_D_delayed_capped_1p5": D,
                          "factor_E_early_reference_persistent": E})
    factor_cols = [f"factor_{arm}" for arm in ARM_NAMES]
    if not np.isfinite(frame[factor_cols].to_numpy()).all() or \
            (frame[factor_cols].to_numpy() < 1 - 1e-12).any():
        raise ValueError("Invalid factor")
    if new_cut != "not_triggered" and (C[int(new_cut) - 1] != 1 or E[int(new_cut) - 1] != 1):
        raise ValueError("Factor at new confirmation cut must be one")
    return frame, {"source": source, "target": target, "scope": "full_1_315",
                   "old_trigger_cut": old_cut, "new_confirmed_trigger_cut": new_cut,
                   "counterfactual_reason": reason, **identity,
                   "old_oor_score_file": str(old_path.resolve()),
                   "old_oor_score_sha256": stage.sha(old_path),
                   "new_gate_file": str(new_path.resolve()),
                   "new_gate_sha256": stage.sha(new_path)}


def plot_factors(path: Path, source: str, target: str, frame: pd.DataFrame) -> None:
    cuts = frame.cut_index.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    colors = ("#d47b24", "#aa9a58", "#2b678b", "#51a0bf", "#963f8e")
    for arm, color in zip(ARM_NAMES, colors):
        axes[0].plot(cuts, frame[f"factor_{arm}"], color=color, lw=1.35, label=arm)
    axes[0].axhline(1.5, color="#666666", ls=":", lw=0.9, label="cap 1.5")
    axes[1].plot(cuts, frame.r_t, color="#667788", lw=1.2, label="r_t")
    axes[1].plot(cuts, frame.new_weight_w_t, color="#963f8e", lw=1.2, label="new soft weight w_t")
    for ax in axes:
        for trigger, color, label in ((frame.old_trigger_cut.iloc[0], "#d47b24", "original trigger"),
                                      (frame.new_confirmed_trigger_cut.iloc[0], "#963f8e", "new t0")):
            if str(trigger) != "not_triggered":
                ax.axvline(int(trigger) - 0.5, color=color, ls="--", lw=0.9, label=label)
        for cut in (105.5, 210.5):
            ax.axvline(cut, color="#888888", ls=":", lw=0.7)
        ax.set_xlim(1, 315)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("multiplicative factor")
    axes[1].set(xlabel="target cut index", ylabel="OOR rate / soft weight")
    fig.suptitle(f"{source.upper()} -> {target.upper()} | full_1_315 | A-E factor decomposition")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def prepare(out: Path, base_root: Path, old_root: Path, new_root: Path) -> None:
    if out.exists():
        raise FileExistsError("Prepare requires an independent new directory")
    audit, old_lock, new_lock = input_locks(base_root, old_root, new_root)
    input_index = {(r.source, r.target, int(r.seed), r.method): r for r in audit.itertuples(index=False)}
    factor_items, rows, prediction_items = [], [], []
    for source, target in baseline.PAIRS:
        factor, provenance = factors(source, target, old_root, new_root)
        factor_items.append((source, target, factor))
        summaries = {}
        for arm in ARM_NAMES:
            values = factor[f"factor_{arm}"].to_numpy(float)
            summaries[f"{arm}_late_mean_factor"] = float(values[210:].mean())
            summaries[f"{arm}_late_max_factor"] = float(values[210:].max())
            summaries[f"{arm}_late_at_1p5_count"] = int(np.count_nonzero(np.isclose(values[210:], 1.5, rtol=0, atol=1e-12)))
            summaries[f"{arm}_full_at_1p5_count"] = int(np.count_nonzero(np.isclose(values, 1.5, rtol=0, atol=1e-12)))
            summaries[f"{arm}_first_at_1p5_cut"] = (int(np.flatnonzero(np.isclose(values, 1.5, rtol=0, atol=1e-12))[0] + 1)
                                                     if np.any(np.isclose(values, 1.5, rtol=0, atol=1e-12)) else "none")
        rows.append({**provenance, **summaries})
        for seed in baseline.SEEDS:
            prior_name = f"{source}_to_{target}_seed_{seed}_full_1_315.csv"
            prior = pd.read_csv(old_root / "unlabeled_predictions" / prior_name)
            for method in baseline.METHODS:
                new_name = f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
                current = pd.read_csv(new_root / "unlabeled_predictions" / new_name)
                record = input_index[source, target, seed, method]
                original_prediction = pd.read_csv(record.original_prediction_file,
                                                  usecols=["cut_index", "pred_vb"])
                if (prior.cut_index.tolist() != baseline.CUTS or current.cut_index.tolist() != baseline.CUTS or
                    original_prediction.cut_index.tolist() != baseline.CUTS):
                    raise ValueError("Not all 315 ordered cuts present")
                p = original_prediction.pred_vb.to_numpy(float)
                if (np.max(abs(p - current.baseline_pred_vb.to_numpy(float))) > 1e-12 or
                    np.max(abs(p - prior[original.PREDICTION_COLUMNS[method]].to_numpy(float))) > 1e-12):
                    raise ValueError("Frozen common baseline prediction differs")
                if (np.max(abs(p * factor.factor_A_original.to_numpy(float) -
                                   current.original_multiplicative_oor_pga_pred_vb.to_numpy(float))) > 1e-10 or
                    np.max(abs(p * factor.factor_E_early_reference_persistent.to_numpy(float) -
                                   current.early_reference_persistent_oor_pga_pred_vb.to_numpy(float))) > 1e-10):
                    raise ValueError("A or E does not reproduce frozen prediction")
                table = factor.copy()
                table.insert(0, "source", source)
                table.insert(1, "target", target)
                table.insert(2, "seed", seed)
                table.insert(3, "base_method", method)
                table["baseline_pred_vb"] = p
                for arm in ARM_NAMES:
                    values = p * factor[f"factor_{arm}"].to_numpy(float)
                    table[f"pred_{arm}_vb"] = values
                prediction_items.append((new_name, table))
    if len(prediction_items) != 60 or len(rows) != 6:
        raise ValueError("Missing direction or seed")
    out.mkdir(parents=True)
    (out / "factors").mkdir()
    (out / "unlabeled_predictions").mkdir()
    for source, target, frame in factor_items:
        csv_write(out / "factors" / f"{source}_to_{target}_factors_full_1_315.csv", frame)
        plot_factors(out / "factors" / f"{source}_to_{target}_factors_full_1_315.png",
                     source, target, frame)
    for name, frame in prediction_items:
        csv_write(out / "unlabeled_predictions" / name, frame)
    csv_write(out / "factor_trigger_m_audit_full_1_315.csv", pd.DataFrame(rows))
    json_write(out / "prediction_lock_before_target_labels.json", {
        "phase": "counterfactual_predictions_frozen_before_target_labels",
        "target_labels_read": False,
        "base_protocol_lock_sha256": stage.sha(base_root / "protocol_lock_full_1_315.json"),
        "old_OOR_prediction_lock_sha256": stage.sha(old_root / "prediction_lock_before_target_labels.json"),
        "new_gate_prediction_lock_sha256": stage.sha(new_root / "prediction_lock_before_target_labels.json"),
        "input_checkpoint_prediction_audit_sha256": stage.sha(new_root / "input_checkpoint_prediction_audit_full_1_315.csv"),
        "factor_trigger_m_audit_sha256": stage.sha(out / "factor_trigger_m_audit_full_1_315.csv"),
        "factor_csv_sha256": {p.name: stage.sha(p) for p in sorted((out / "factors").glob("*.csv"))},
        "unlabeled_prediction_csv_sha256": {p.name: stage.sha(p) for p in sorted((out / "unlabeled_predictions").glob("*.csv"))},
        "common_m": "source-only Theil-Sen exponent from original per-cut OOR file; A and E preserved; B-D diagnostic only",
        "A": "saved original factor", "B": "minimum of original factor and 1.5",
        "C": "saved new confirmed t0, pure (tau/t0)^m after t0, no cap or soft weight",
        "D": "minimum of C and 1.5", "E": "saved early-reference persistent factor",
        "no_trigger": "keep baseline prediction, reason in factor_trigger_m_audit_full_1_315.csv",
        "command": "python diagnose_oor_pga_factors.py prepare"})
    print(f"Frozen B-D diagnostics and reproduced A/E for 60 label-free predictions: {out}")


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    error = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "R2": float(1 - np.sum(error * error) / sst) if sst > 0 else np.nan,
            "MAE": float(np.mean(abs(error))),
            "RMSE": float(np.sqrt(np.mean(error * error))),
            "signed_bias": float(error.mean()),
            "underprediction_count": int(np.count_nonzero(error < 0)),
            "SSE": float(np.sum(error * error))}


def plot_focus(path: Path, source: str, target: str, method: str, y: np.ndarray,
               frames: list[pd.DataFrame], t0: str, cap_cuts: dict) -> None:
    cuts = np.arange(1, 316)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    axes[0].plot(cuts, y, color="black", lw=1.8, label="true VB")
    colors = {"baseline": "#28649b", "A_original": "#d47b24",
              "B_original_capped_1p5": "#a29d56", "C_delayed_uncapped": "#4798bb",
              "D_delayed_capped_1p5": "#63ad91", "E_early_reference_persistent": "#963f8e"}
    for arm in ("baseline", *ARM_NAMES):
        column = "baseline_pred_vb" if arm == "baseline" else f"pred_{arm}_vb"
        matrix = np.stack([f[column].to_numpy(float) for f in frames])
        mean = matrix.mean(axis=0)
        axes[0].plot(cuts, mean, color=colors[arm], lw=1.2, label=f"{arm} five-seed mean")
        if arm in ("A_original", "E_early_reference_persistent"):
            axes[1].plot(cuts, mean - y, color=colors[arm], lw=1.3,
                         label=f"{arm} mean signed error")
    axes[1].axhline(0, color="black", lw=0.7)
    for ax in axes:
        if t0 != "not_triggered":
            ax.axvline(int(t0) - 0.5, color="#963f8e", ls=":", lw=1.0, label="new confirmed t0")
        for arm, cut in cap_cuts.items():
            if cut != "none":
                ax.axvline(int(cut) - 0.5, color=colors[arm], ls="-.", lw=0.8,
                           label=f"{arm} first 1.5 cap")
        for edge in (105.5, 210.5):
            ax.axvline(edge, color="#777777", ls="--", lw=0.7)
        ax.set_xlim(1, 315)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("VB")
    axes[1].set(xlabel="target cut index", ylabel="prediction - truth")
    fig.suptitle(f"{source.upper()} -> {target.upper()} | {method} | full_1_315 | factor mechanism")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def evaluate(out: Path, base_root: Path, old_root: Path, new_root: Path, raw_root: Path) -> None:
    lock_path = out / "prediction_lock_before_target_labels.json"
    if not lock_path.exists() or (out / "per_seed_metrics_full_1_315.csv").exists():
        raise FileExistsError("Evaluate requires frozen unlabeled files and no prior metrics")
    lock = stage.json_read(lock_path)
    audit, _, _ = input_locks(base_root, old_root, new_root)
    for parent, key in (("factors", "factor_csv_sha256"),
                        ("unlabeled_predictions", "unlabeled_prediction_csv_sha256")):
        for name, digest in lock[key].items():
            if stage.sha(out / parent / name) != digest:
                raise ValueError(f"Frozen file changed: {name}")
    if len(lock["unlabeled_prediction_csv_sha256"]) != 60:
        raise ValueError("Not all 60 counterfactual predictions frozen")
    truth = {}
    for target in stage.TOOLS:
        expected = set(audit.loc[audit.target == target, "target_wear_sha256"])
        if len(expected) != 1 or stage.sha(raw_root / f"{target}_wear.csv") != next(iter(expected)):
            raise ValueError(f"Target label provenance changed: {target}")
        truth[target], _, info = stage.labels(raw_root, target)
        if info["cuts_1_315_with_label"] != 315:
            raise ValueError("Missing target wear")
    earlier = pd.read_csv(new_root / "per_seed_metrics_full_1_315.csv")
    factor_audit = pd.read_csv(out / "factor_trigger_m_audit_full_1_315.csv", keep_default_na=False)
    (out / "evaluated_per_seed").mkdir()
    (out / "focus_figures").mkdir()
    per_seed, paired, per_cut = [], [], []
    for source, target in baseline.PAIRS:
        info = factor_audit[(factor_audit.source == source) & (factor_audit.target == target)].iloc[0]
        for method in baseline.METHODS:
            frames = []
            for seed in baseline.SEEDS:
                name = f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
                frame = pd.read_csv(out / "unlabeled_predictions" / name)
                if frame.cut_index.tolist() != baseline.CUTS:
                    raise ValueError("Cut order invalid")
                frames.append(frame)
                evaluated = frame.copy()
                evaluated.insert(5, "true_vb", truth[target])
                for arm in ARM_NAMES:
                    evaluated[f"signed_error_{arm}"] = frame[f"pred_{arm}_vb"].to_numpy(float) - truth[target]
                    for segment, lo, hi in SEGMENTS:
                        result = metric(truth[target][lo - 1:hi],
                                        frame[f"pred_{arm}_vb"].to_numpy(float)[lo - 1:hi])
                        per_seed.append({"source": source, "target": target, "seed": seed,
                                         "base_method": method, "arm": arm,
                                         "scope": "full_1_315", "segment": segment,
                                         "cut_first": lo, "cut_last": hi, **result})
                        if arm in ("A_original", "E_early_reference_persistent"):
                            variant = ("original_multiplicative_oor_pga" if arm == "A_original"
                                       else "early_reference_persistent_oor_pga")
                            prev = earlier[(earlier.source == source) & (earlier.target == target) &
                                           (earlier.seed == seed) & (earlier.base_method == method) &
                                           (earlier.variant == variant) & (earlier.segment == segment)]
                            if len(prev) != 1 or any(abs(result[k] - prev[k].iloc[0]) > 1e-9
                                                     for k in ("R2", "MAE", "RMSE", "signed_bias")):
                                raise ValueError(f"A/E existing metric does not reproduce: {name} {arm} {segment}")
                csv_write(out / "evaluated_per_seed" / name, evaluated)
            if (source, target) in (("c1", "c6"), ("c4", "c1"), ("c6", "c1")):
                cap_cuts = {arm: info[f"{arm}_first_at_1p5_cut"] for arm in ARM_NAMES}
                plot_focus(out / "focus_figures" /
                           f"{source}_to_{target}_{method}_predictions_errors_full_1_315.png",
                           source, target, method, truth[target], frames,
                           str(info.new_confirmed_trigger_cut), cap_cuts)
            matrix = {arm: np.stack([f[f"pred_{arm}_vb"].to_numpy(float) for f in frames])
                      for arm in ARM_NAMES}
            for cut in baseline.CUTS:
                j = cut - 1
                row = {"source": source, "target": target, "base_method": method,
                       "scope": "full_1_315", "cut_index": cut, "true_vb": truth[target][j]}
                for arm in ARM_NAMES:
                    error = matrix[arm][:, j] - truth[target][j]
                    row[f"{arm}_mean_signed_error"] = float(error.mean())
                    row[f"{arm}_sum_squared_error_five_seeds"] = float(np.sum(error * error))
                    row[f"{arm}_underpredicting_seeds"] = int(np.count_nonzero(error < 0))
                per_cut.append(row)
    metrics = pd.DataFrame(per_seed)
    if len(metrics) != 1200:
        raise ValueError("Expected 1200 per-seed segment rows")
    csv_write(out / "per_seed_metrics_full_1_315.csv", metrics)
    csv_write(out / "per_cut_error_decomposition_full_1_315.csv", pd.DataFrame(per_cut))
    summary = []
    for (source, target, method, arm, segment), group in metrics.groupby(
            ["source", "target", "base_method", "arm", "segment"], sort=True):
        row = {"source": source, "target": target, "base_method": method,
               "arm": arm, "segment": segment, "scope": "full_1_315",
               "n_seeds": 5, "n_cuts_per_seed": int(group.n.iloc[0])}
        for k in ("R2", "MAE", "RMSE", "signed_bias", "underprediction_count", "SSE"):
            row[f"{k}_mean"] = float(group[k].mean())
            row[f"{k}_sd"] = float(group[k].std(ddof=1))
        summary.append(row)
    csv_write(out / "five_seed_summary_full_1_315.csv", pd.DataFrame(summary))
    index = {(r.source, r.target, int(r.seed), r.base_method, r.segment, r.arm): r
             for r in metrics.itertuples(index=False)}
    for source, target in baseline.PAIRS:
        for seed in baseline.SEEDS:
            for method in baseline.METHODS:
                for segment, _, _ in SEGMENTS:
                    for arm in ("B_original_capped_1p5", "C_delayed_uncapped",
                                "D_delayed_capped_1p5"):
                        for reference in ("A_original", "E_early_reference_persistent"):
                            a = index[source, target, seed, method, segment, arm]
                            b = index[source, target, seed, method, segment, reference]
                            paired.append({"source": source, "target": target, "seed": seed,
                                           "base_method": method, "arm": arm,
                                           "reference": reference, "segment": segment,
                                           "scope": "full_1_315", "n": a.n,
                                           **{f"delta_{k}": getattr(a, k) - getattr(b, k)
                                              for k in ("R2", "MAE", "RMSE", "signed_bias", "SSE")}})
    paired_df = pd.DataFrame(paired)
    csv_write(out / "paired_seed_delta_vs_A_E_full_1_315.csv", paired_df)
    paired_summary = []
    for (source, target, method, arm, reference, segment), group in paired_df.groupby(
            ["source", "target", "base_method", "arm", "reference", "segment"], sort=True):
        row = {"source": source, "target": target, "base_method": method,
               "arm": arm, "reference": reference, "segment": segment,
               "scope": "full_1_315", "n_seeds": 5,
               "improved_seed_count_RMSE": int((group.delta_RMSE < 0).sum())}
        for k in ("R2", "MAE", "RMSE", "signed_bias", "SSE"):
            row[f"delta_{k}_mean"] = float(group[f"delta_{k}"].mean())
            row[f"delta_{k}_sd"] = float(group[f"delta_{k}"].std(ddof=1))
        paired_summary.append(row)
    csv_write(out / "paired_five_seed_delta_vs_A_E_full_1_315.csv", pd.DataFrame(paired_summary))
    json_write(out / "evaluation_manifest_full_1_315.json", {
        "prediction_lock_sha256": stage.sha(lock_path),
        "target_label_reads_after_freezing": True,
        "target_wear_sha256": {t: stage.sha(raw_root / f"{t}_wear.csv") for t in stage.TOOLS},
        "per_seed_metrics_sha256": stage.sha(out / "per_seed_metrics_full_1_315.csv"),
        "per_cut_error_decomposition_sha256": stage.sha(out / "per_cut_error_decomposition_full_1_315.csv"),
        "command": "python diagnose_oor_pga_factors.py evaluate"})
    print(f"Evaluated five fixed factor arms across six directions and five seeds: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-root", type=Path, default=original.BASELINE)
    parser.add_argument("--old-oor-root", type=Path, default=original.ROOT)
    parser.add_argument("--new-gate-root", type=Path, default=persistent.DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=original.RAW)
    args = parser.parse_args()
    out, base_root, old_root, new_root, raw_root = (
        p.resolve() for p in (args.out_root, args.baseline_root, args.old_oor_root,
                              args.new_gate_root, args.raw_root))
    if args.phase == "prepare":
        prepare(out, base_root, old_root, new_root)
    else:
        evaluate(out, base_root, old_root, new_root, raw_root)


if __name__ == "__main__":
    main()
