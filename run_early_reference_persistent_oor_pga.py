"""Exploratory early-reference persistent OOR-PGA for frozen PHM2010 predictions.

Run audit, prepare, evaluate in order. Audit and prepare never read target wear.
The same fixed rule is used in all six directions and all five seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import analyze_zscore_stage_errors as stage
import run_full_1_315_baseline_protocol as baseline
import run_oor_pga_full_1_315 as old
import run_single_source_pairs as base


DEFAULT_OUT = Path("artifacts/early_reference_persistent_oor_pga_full_1_315_20260925")
DELTA_L, DELTA_H, K, FMAX = 2 / 48, 5 / 48, 5, 1.5
SOURCE_FRACTION = 0.6
SEGS = (("full", 1, 315), ("early", 1, 105), ("middle", 106, 210), ("late", 211, 315))


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def locked_inputs(base_root: Path, old_root: Path) -> tuple[pd.DataFrame, dict]:
    audit = old.check_baseline_lock(base_root)
    prior = stage.json_read(old_root / "prediction_lock_before_target_labels.json")
    if (stage.sha(base_root / "protocol_lock_full_1_315.json") != prior["baseline_lock_sha256"] or
        stage.sha(base_root / "checkpoint_audit_full_1_315.csv") != prior["baseline_checkpoint_audit_sha256"]):
        raise ValueError("Existing OOR predictions no longer match frozen baseline")
    for name, digest in prior["unlabeled_prediction_sha256"].items():
        if stage.sha(old_root / "unlabeled_predictions" / name) != digest:
            raise ValueError(f"Existing OOR prediction changed: {name}")
    for name, digest in prior["oor_score_sha256"].items():
        if stage.sha(old_root / "oor_scores" / name) != digest:
            raise ValueError(f"Existing OOR score changed: {name}")
    if stage.sha(old_root / "source_48_feature_support.csv") != prior["source_feature_support_sha256"]:
        raise ValueError("Existing 48-feature support changed")
    return audit, prior


def pair_inputs(source: str, target: str, audit: pd.DataFrame,
                old_root: Path, cache_root: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    row = audit[(audit.source == source) & (audit.target == target) &
                (audit.seed == 42) & (audit.method == "source_only")].iloc[0]
    xs = old.features48(cache_root / f"{source}_stft.npy", row.source_stft_sha256)
    xt = old.features48(cache_root / f"{target}_stft.npy", row.target_stft_sha256)
    mu, sd = xs.mean(axis=0), xs.std(axis=0)
    zs = (xs - mu) / np.maximum(sd, 1e-12)
    zt = (xt - mu) / np.maximum(sd, 1e-12)
    lower, upper = zs.min(axis=0), zs.max(axis=0)
    flags = (zt < lower) | (zt > upper)
    rate = flags.mean(axis=1)
    score_file = old_root / "oor_scores" / f"{source}_to_{target}_oor_score_full_1_315.csv"
    score = pd.read_csv(score_file)
    if score.cut_index.tolist() != baseline.CUTS or len(score) != 315:
        raise ValueError(f"Existing OOR cut mismatch: {score_file}")
    if not np.array_equal(flags.sum(axis=1), score.oor_flag_count_48.to_numpy(int)) or \
            not np.allclose(rate, score.oor_rate_48.to_numpy(float), rtol=0, atol=1e-15):
        raise ValueError(f"Recomputed 48-feature OOR differs: {source}->{target}")
    support = pd.read_csv(old_root / "source_48_feature_support.csv")
    support = support[(support.source == source) & (support.target == target)].sort_values("feature_index")
    if len(support) != 48 or support.feature_index.tolist() != list(range(48)):
        raise ValueError("48-feature order mismatch")
    for column, actual in (("source_raw_min", xs.min(axis=0)), ("source_raw_max", xs.max(axis=0)),
                           ("source_z_min", lower), ("source_z_max", upper)):
        if not np.allclose(support[column].to_numpy(float), actual, rtol=0, atol=1e-12):
            raise ValueError(f"Source support mismatch: {source}->{target} {column}")
    if (support.resized_frequency_row_first.tolist() != [16 * (i % 8) for i in range(48)] or
        support.resized_frequency_row_last.tolist() != [16 * (i % 8) + 15 for i in range(48)]):
        raise ValueError("STFT 48-band definition mismatch")
    return rate, flags, score, {"source_stft_sha256": row.source_stft_sha256,
                                "target_stft_sha256": row.target_stft_sha256,
                                "source_raw_min": float(xs.min()), "source_raw_max": float(xs.max()),
                                "target_raw_min": float(xt.min()), "target_raw_max": float(xt.max()),
                                "source_feature_constant_count": int((sd <= 1e-12).sum()),
                                "source_support_file": str((old_root / "source_48_feature_support.csv").resolve()),
                                "source_support_sha256": stage.sha(old_root / "source_48_feature_support.csv")}


def plot_profile(path: Path, source: str, target: str, rate: np.ndarray) -> None:
    cuts = np.arange(1, 316)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(cuts, rate, lw=1.3, color="#345d84", label="r: out-of-range fraction (48 features)")
    ax.axvspan(1, 32, alpha=0.1, color="#a85987", label="early reference cuts 1-32")
    for edge in (105.5, 210.5):
        ax.axvline(edge, color="#777777", lw=0.8, ls="--")
    ax.set(xlim=(1, 315), ylim=(0, max(0.5, float(rate.max()) * 1.05)),
           xlabel="target cut index", ylabel="OOR fraction",
           title=f"{source.upper()} -> {target.upper()} | r_t | full_1_315 | unlabeled STFT-48")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def audit_phase(out: Path, base_root: Path, old_root: Path, cache_root: Path) -> None:
    if out.exists():
        raise FileExistsError("Input audit requires a new independent directory")
    audit, prior = locked_inputs(base_root, old_root)
    profiles, checks = [], []
    for source, target in baseline.PAIRS:
        rate, _, score, info = pair_inputs(source, target, audit, old_root, cache_root)
        checks.append({"source": source, "target": target, "scope": "full_1_315",
                       "n_cuts": 315, "n_features": 48,
                       "r_early_1_32_median": float(np.median(rate[:32])),
                       "r_early_1_105_median": float(np.median(rate[:105])),
                       "r_late_211_315_median": float(np.median(rate[210:])),
                       "r_late_minus_early_median": float(np.median(rate[210:]) - np.median(rate[:32])),
                       "r_cut_1": float(rate[0]), "r_cut_2": float(rate[1]), "r_cut_3": float(rate[2]),
                       "r_cut_315": float(rate[-1]),
                       "original_oor_trigger_cut": str(score.trigger_cut.iloc[0]), **info})
        profiles.append((source, target, rate))
    prediction_checks = []
    for row in audit.itertuples(index=False):
        source, target, seed, method = row.source, row.target, int(row.seed), row.method
        name = f"{source}_to_{target}_seed_{seed}_full_1_315.csv"
        old_file = old_root / "unlabeled_predictions" / name
        original = Path(row.original_prediction_file)
        if stage.sha(original) != row.original_prediction_sha256 or stage.sha(Path(row.checkpoint)) != row.checkpoint_sha256:
            raise ValueError(f"Original checkpoint/prediction hash changed: {source}->{target} {seed} {method}")
        prior_pred = pd.read_csv(old_file, usecols=["cut_index", old.PREDICTION_COLUMNS[method]])
        raw_pred = pd.read_csv(original, usecols=["cut_index", "pred_vb"])
        max_difference = float(np.max(np.abs(prior_pred.iloc[:, 1].to_numpy(float) -
                                             raw_pred.pred_vb.to_numpy(float))))
        if prior_pred.cut_index.tolist() != baseline.CUTS or raw_pred.cut_index.tolist() != baseline.CUTS or \
                max_difference > 1e-12:
            raise ValueError(f"Original predictions differ: {old_file}")
        prediction_checks.append({"source": source, "target": target, "seed": seed, "method": method,
                                  "scope": "full_1_315", "checkpoint": row.checkpoint,
                                  "checkpoint_sha256": row.checkpoint_sha256,
                                  "original_prediction_file": row.original_prediction_file,
                                  "original_prediction_sha256": row.original_prediction_sha256,
                                  "prior_unlabeled_prediction_file": str(old_file.resolve()),
                                  "prior_unlabeled_prediction_sha256": stage.sha(old_file),
                                  "n_unique_ordered_cuts": 315,
                                  "max_serialized_prediction_abs_difference": max_difference,
                                  "prediction_equal_within_csv_roundtrip_1e_minus_12": True})
    if len(prediction_checks) != 60:
        raise ValueError("Expected 60 frozen prediction inputs")
    out.mkdir(parents=True)
    (out / "r_profiles").mkdir()
    for source, target, rate in profiles:
        csv_write(out / "r_profiles" / f"{source}_to_{target}_r_full_1_315.csv",
                  pd.DataFrame({"cut_index": baseline.CUTS, "tau": old.TAU, "r_t": rate}))
        plot_profile(out / "r_profiles" / f"{source}_to_{target}_r_full_1_315.png", source, target, rate)
    csv_write(out / "input_feature_audit_full_1_315.csv", pd.DataFrame(checks))
    csv_write(out / "input_checkpoint_prediction_audit_full_1_315.csv", pd.DataFrame(prediction_checks))
    json_write(out / "input_audit_lock.json", {
        "phase": "audit_before_source_trend_and_target_label_reads",
        "target_labels_read": False, "n_prediction_inputs": 60,
        "baseline_lock_sha256": stage.sha(base_root / "protocol_lock_full_1_315.json"),
        "prior_OOR_prediction_lock_sha256": stage.sha(old_root / "prediction_lock_before_target_labels.json"),
        "input_feature_audit_sha256": stage.sha(out / "input_feature_audit_full_1_315.csv"),
        "input_checkpoint_prediction_audit_sha256": stage.sha(out / "input_checkpoint_prediction_audit_full_1_315.csv"),
        "r_profile_sha256": {p.name: stage.sha(p) for p in sorted((out / "r_profiles").glob("*.csv"))},
        "cache_manifest_sha256": stage.sha(cache_root / "manifest.json"),
        "feature_definition": "6 channels x 8 contiguous 16-row bands of 128x128 full-frequency STFT; mean over band and time",
        "source_support": "source-fitted 48-feature Z-score then source feature-wise min/max; target unlabeled STFT only",
        "fixed_delta_L": DELTA_L, "fixed_delta_H": DELTA_H, "fixed_K": K,
        "fixed_Fmax": FMAX, "fixed_source_stage_fraction": SOURCE_FRACTION,
        "command": "python run_early_reference_persistent_oor_pga.py audit"})
    print(f"Audited 60 label-free prediction inputs and six OOR profiles: {out}")


def causal_median(rate: np.ndarray) -> np.ndarray:
    return np.asarray([np.median(rate[max(0, t - 4):t + 1]) for t in range(315)], dtype=float)


def source_stage(source: str, raw_root: Path, expected_sha: str) -> dict:
    path = raw_root / f"{source}_wear.csv"
    if stage.sha(path) != expected_sha:
        raise ValueError(f"Source label hash mismatch: {path}")
    wear = base.wear_labels(raw_root, source).astype(float)
    if not np.isfinite(wear).all():
        return {"status": "invalid_source_labels", "tau_source": None, "cut_source": None,
                "trend": np.full(315, np.nan)}
    trend = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(old.TAU, wear)
    total = float(trend[-1] - trend[0])
    if not np.isfinite(trend).all() or total <= 0:
        return {"status": "invalid_source_trend_or_nonpositive_increment", "tau_source": None,
                "cut_source": None, "trend": trend, "total_increment": total}
    threshold = float(trend[0] + SOURCE_FRACTION * total)
    hits = np.flatnonzero(trend >= threshold - 1e-12)
    if len(hits) == 0:
        return {"status": "source_stage_never_reached", "tau_source": None,
                "cut_source": None, "trend": trend, "total_increment": total}
    idx = int(hits[0])
    return {"status": "defined", "tau_source": float(old.TAU[idx]), "cut_source": idx + 1,
            "trend": trend, "total_increment": total,
            "trend_start": float(trend[0]), "trend_end": float(trend[-1]),
            "threshold_wear": threshold}


def gate_rule(rate: np.ndarray, tau_source: float | None, m: float,
              source_status: str) -> tuple[pd.DataFrame, dict]:
    b = float(np.median(rate[:32]))
    smoothed = causal_median(rate)
    d = np.maximum(0, smoothed - b)
    eligible = (np.arange(1, 316) > 32) & (d >= DELTA_H)
    if tau_source is not None:
        eligible &= old.TAU >= tau_source
    else:
        eligible[:] = False
    count = np.zeros(315, dtype=int)
    trigger = None
    run = 0
    for idx, valid in enumerate(eligible):
        run = run + 1 if valid else 0
        count[idx] = run
        if trigger is None and run >= K:
            trigger = idx
    reason = "triggered" if trigger is not None else "no_persistent_eligible_trigger"
    if source_status != "defined":
        reason = source_status
        trigger = None
    if not np.isfinite(m) or m <= 0:
        reason = "nonpositive_or_invalid_source_pga_m"
        trigger = None
    weight = np.zeros(315)
    factor = np.ones(315)
    if trigger is not None:
        t0 = float(old.TAU[trigger])
        if t0 <= 0:
            raise ValueError("Invalid positive trigger tau")
        weight[trigger + 1:] = np.clip((d[trigger + 1:] - DELTA_L) / (DELTA_H - DELTA_L), 0, 1)
        growth = np.clip(np.power(old.TAU[trigger + 1:] / t0, m) - 1, 0, FMAX - 1)
        factor[trigger + 1:] = 1 + weight[trigger + 1:] * growth
    if not (np.isfinite(factor).all() and np.isfinite(weight).all() and
            factor.min() >= 1 - 1e-12 and factor.max() <= FMAX + 1e-12 and
            np.all(factor[:32] == 1)):
        raise ValueError("Factor bounds or early freeze failed")
    if trigger is not None and factor[trigger] != 1:
        raise ValueError("Factor at fifth confirmation cut must be one")
    frame = pd.DataFrame({"cut_index": baseline.CUTS, "tau": old.TAU,
                          "r_t": rate, "early_reference_b": b,
                          "causal_median_s_t": smoothed, "excess_d_t": d,
                          "eligible_high_threshold": eligible,
                          "consecutive_confirmation_count": count,
                          "tau_source": tau_source if tau_source is not None else np.nan,
                          "trigger_t0_cut": trigger + 1 if trigger is not None else "not_triggered",
                          "trigger_tau_t0": old.TAU[trigger] if trigger is not None else np.nan,
                          "weight_w_t": weight, "factor_F_t": factor})
    return frame, {"status": reason, "trigger_cut": trigger + 1 if trigger is not None else "not_triggered",
                   "trigger_tau": float(old.TAU[trigger]) if trigger is not None else "",
                   "early_reference_b": b, "m": m, "tau_source": tau_source if tau_source is not None else "",
                   "delta_L": DELTA_L, "delta_H": DELTA_H, "K": K, "Fmax": FMAX,
                   "max_factor": float(factor.max()), "n_changed_cuts": int((factor > 1 + 1e-12).sum())}


def plot_gate(path: Path, source: str, target: str, frame: pd.DataFrame, status: dict) -> None:
    cuts = frame.cut_index.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(cuts, frame.r_t, color="#2b6089", lw=1.2, label="r_t")
    axes[0].axhline(status["early_reference_b"], color="#777777", ls=":", lw=1,
                    label="early median b")
    axes[1].plot(cuts, frame.excess_d_t, color="#a35330", lw=1.2, label="d_t")
    axes[1].axhline(DELTA_L, color="#6b8b4a", ls=":", lw=1, label="delta_L=2/48")
    axes[1].axhline(DELTA_H, color="#a73554", ls="--", lw=1, label="delta_H=5/48")
    for ax in axes:
        ax.axvspan(1, 32, color="#777777", alpha=0.08)
        if status["tau_source"] != "":
            ax.axvline(round(float(status["tau_source"]) * 314 + 1) - 0.5,
                       color="#6d4b94", ls="-.", lw=1, label="source 60% stage")
        if status["trigger_cut"] != "not_triggered":
            ax.axvline(int(status["trigger_cut"]) - 0.5, color="#c33424", lw=1.1,
                       label="fifth confirmation t0")
        for edge in (105.5, 210.5):
            ax.axvline(edge, color="#888888", lw=0.7, ls="--")
        ax.set_xlim(1, 315)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("target cut index")
    axes[0].set_ylabel("OOR rate")
    axes[1].set_ylabel("excess above early state")
    fig.suptitle(f"{source.upper()} -> {target.upper()} | full_1_315 | early-reference persistent gate")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_phase(out: Path, base_root: Path, old_root: Path, cache_root: Path, raw_root: Path) -> None:
    audit_lock = stage.json_read(out / "input_audit_lock.json")
    if (out / "prediction_lock_before_target_labels.json").exists():
        raise FileExistsError("Predictions already frozen")
    audit, prior = locked_inputs(base_root, old_root)
    if (stage.sha(out / "input_feature_audit_full_1_315.csv") != audit_lock["input_feature_audit_sha256"] or
        stage.sha(out / "input_checkpoint_prediction_audit_full_1_315.csv") !=
        audit_lock["input_checkpoint_prediction_audit_sha256"]):
        raise ValueError("Input audit changed")
    exponent_file = old_root / "source_pga_exponents.csv"
    if stage.sha(exponent_file) != prior["source_pga_exponents_sha256"]:
        raise ValueError("Existing source PGA exponents changed")
    exponents = pd.read_csv(exponent_file).set_index("source")
    audit_index = {(r.source, r.target, int(r.seed), r.method): r for r in audit.itertuples(index=False)}
    source_trend, source_status = {}, {}
    for source in stage.TOOLS:
        r = audit_index[source, next(t for t in stage.TOOLS if t != source), 42, "source_only"]
        info = source_stage(source, raw_root, r.source_wear_sha256)
        source_trend[source] = info
        reference_m = old.source_power(raw_root / f"{source}_wear.csv", r.source_wear_sha256)["m"]
        if abs(reference_m - float(exponents.loc[source, "m"])) > 1e-12:
            raise ValueError("Existing source-only PGA exponent mismatch")
        source_status[source] = {k: v for k, v in info.items() if k != "trend"}
    (out / "unlabeled_predictions").mkdir()
    (out / "gate_profiles").mkdir()
    (out / "source_trends").mkdir()
    trigger_rows = []
    for source in stage.TOOLS:
        trend = source_trend[source]["trend"]
        csv_write(out / "source_trends" / f"{source}_isotonic_source_trend.csv",
                  pd.DataFrame({"cut_index": baseline.CUTS, "tau": old.TAU,
                                "source_isotonic_wear": trend}))
    for source, target in baseline.PAIRS:
        rate, _, _, _ = pair_inputs(source, target, audit, old_root, cache_root)
        source_info = source_trend[source]
        m = float(exponents.loc[source, "m"])
        gate, status = gate_rule(rate, source_info["tau_source"], m, source_info["status"])
        status.update({"source": source, "target": target, "scope": "full_1_315",
                       "source_stage_method": "isotonic regression, increasing=True, all source cuts 1..315",
                       "source_stage_fraction": SOURCE_FRACTION,
                       "source_stage_cut": source_info["cut_source"] if source_info["cut_source"] is not None else "",
                       "source_trend_status": source_info["status"],
                       "source_trend_total_increment": source_info.get("total_increment", ""),
                       "target_label_reads": 0})
        trigger_rows.append(status)
        csv_write(out / "gate_profiles" / f"{source}_to_{target}_gate_full_1_315.csv", gate)
        plot_gate(out / "gate_profiles" / f"{source}_to_{target}_gate_full_1_315.png",
                  source, target, gate, status)
        for seed in baseline.SEEDS:
            previous_name = f"{source}_to_{target}_seed_{seed}_full_1_315.csv"
            previous = pd.read_csv(old_root / "unlabeled_predictions" / previous_name)
            if previous.cut_index.tolist() != baseline.CUTS:
                raise ValueError("Prior prediction cut mismatch")
            for method in baseline.METHODS:
                row = audit_index[source, target, seed, method]
                original = pd.read_csv(row.original_prediction_file,
                                       usecols=["cut_index", "pred_vb"])
                if original.cut_index.tolist() != baseline.CUTS:
                    raise ValueError("Original prediction cut mismatch")
                p = original.pred_vb.to_numpy(float)
                if np.max(np.abs(p - previous[old.PREDICTION_COLUMNS[method]].to_numpy(float))) > 1e-12:
                    raise ValueError("Prior original prediction changed")
                corrected = p * gate.factor_F_t.to_numpy(float)
                if not np.isfinite(corrected).all():
                    raise ValueError("Nonfinite corrected prediction")
                table = gate.copy()
                table["method"] = method
                table["seed"] = seed
                table["source"] = source
                table["target"] = target
                table["baseline_pred_vb"] = p
                table["original_multiplicative_oor_pga_pred_vb"] = previous[
                    old.PREDICTION_COLUMNS[method + "_oor_pga"]].to_numpy(float)
                table["early_reference_persistent_oor_pga_pred_vb"] = corrected
                name = f"{source}_to_{target}_seed_{seed}_{method}_full_1_315.csv"
                csv_write(out / "unlabeled_predictions" / name, table)
    if len(list((out / "unlabeled_predictions").glob("*.csv"))) != 60:
        raise ValueError("Expected 60 frozen label-free prediction files")
    csv_write(out / "trigger_summary_full_1_315.csv", pd.DataFrame(trigger_rows))
    json_write(out / "source_stage_fit_audit.json", source_status)
    json_write(out / "prediction_lock_before_target_labels.json", {
        "phase": "prepare_before_target_label_reads", "target_labels_read": False,
        "input_audit_lock_sha256": stage.sha(out / "input_audit_lock.json"),
        "prior_OOR_prediction_lock_sha256": stage.sha(old_root / "prediction_lock_before_target_labels.json"),
        "source_pga_exponents_sha256": stage.sha(exponent_file),
        "source_wear_sha256": {source: stage.sha(raw_root / f"{source}_wear.csv") for source in stage.TOOLS},
        "checkpoint_prediction_provenance_sha256": stage.sha(out / "input_checkpoint_prediction_audit_full_1_315.csv"),
        "source_stage_fit_method": "IsotonicRegression(increasing=True) on all 315 source training labels; first fitted wear at 60% of fitted end-start increment",
        "continuity_rule": "eligible high threshold requires t>32 and tau>=tau_source; count resets outside eligibility; fifth cut triggers; no backfill",
        "causal_smoothing": "median of current and previous at most four OOR rates",
        "b": "median of first 32 unlabeled OOR rates; first 32 predictions unchanged",
        "fixed_delta_L": DELTA_L, "fixed_delta_H": DELTA_H, "fixed_K": K,
        "fixed_Fmax": FMAX, "fixed_source_stage_fraction": SOURCE_FRACTION,
        "gate_profile_sha256": {p.name: stage.sha(p) for p in sorted((out / "gate_profiles").glob("*.csv"))},
        "unlabeled_prediction_sha256": {p.name: stage.sha(p) for p in sorted((out / "unlabeled_predictions").glob("*.csv"))},
        "trigger_summary_sha256": stage.sha(out / "trigger_summary_full_1_315.csv"),
        "source_stage_fit_audit_sha256": stage.sha(out / "source_stage_fit_audit.json"),
        "command": "python run_early_reference_persistent_oor_pga.py prepare"})
    print(f"Frozen 60 label-free corrected prediction files: {out}")


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "R2": float(1 - np.sum(e * e) / sst) if sst > 0 else np.nan,
            "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e * e))),
            "signed_bias": float(np.mean(e))}


def plot_predictions(path: Path, source: str, target: str, y: np.ndarray,
                     per_method: dict[str, list[pd.DataFrame]], trigger: str) -> None:
    cuts = np.arange(1, 316)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    colors = ("#205d90", "#d47b24", "#963f8e")
    keys = ("baseline_pred_vb", "original_multiplicative_oor_pga_pred_vb",
            "early_reference_persistent_oor_pga_pred_vb")
    names = ("baseline", "original multiplicative OOR-PGA", "early-reference persistent OOR-PGA")
    for ax, method in zip(axes, baseline.METHODS):
        ax.plot(cuts, y, color="black", lw=1.7, label="true VB")
        for key, name, color in zip(keys, names, colors):
            matrix = np.stack([f[key].to_numpy(float) for f in per_method[method]])
            ax.plot(cuts, matrix.mean(axis=0), color=color, lw=1.4,
                    label=f"{method} {name}, five-seed mean")
        if trigger != "not_triggered":
            ax.axvline(int(trigger) - 0.5, color="#b32e2e", ls=":", lw=1,
                       label="confirmed t0")
        for edge in (105.5, 210.5):
            ax.axvline(edge, color="#777777", ls="--", lw=0.8)
        ax.set(xlim=(1, 315), ylabel="VB")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("target cut index")
    fig.suptitle(f"{source.upper()} -> {target.upper()} | full_1_315 | frozen predictions")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_phase(out: Path, base_root: Path, old_root: Path, raw_root: Path) -> None:
    lock_path = out / "prediction_lock_before_target_labels.json"
    if not lock_path.is_file() or (out / "per_seed_metrics_full_1_315.csv").exists():
        raise FileExistsError("Evaluate requires frozen predictions and no existing results")
    lock = stage.json_read(lock_path)
    if stage.sha(out / "input_audit_lock.json") != lock["input_audit_lock_sha256"]:
        raise ValueError("Input audit lock changed")
    audit, prior = locked_inputs(base_root, old_root)
    for name, digest in lock["unlabeled_prediction_sha256"].items():
        if stage.sha(out / "unlabeled_predictions" / name) != digest:
            raise ValueError(f"Frozen prediction changed: {name}")
    for name, digest in lock["gate_profile_sha256"].items():
        if stage.sha(out / "gate_profiles" / name) != digest:
            raise ValueError(f"Frozen gate changed: {name}")
    if len(lock["unlabeled_prediction_sha256"]) != 60:
        raise ValueError("Not all 60 predictions frozen")
    truth = {}
    for target in stage.TOOLS:
        subset = audit[audit.target == target]
        hashes = set(subset.target_wear_sha256)
        if len(hashes) != 1 or stage.sha(raw_root / f"{target}_wear.csv") != next(iter(hashes)):
            raise ValueError(f"Target label source mismatch: {target}")
        truth[target], _, info = stage.labels(raw_root, target)
        if info["cuts_1_315_with_label"] != 315:
            raise ValueError(f"Target labels incomplete: {target}")
    (out / "evaluated_per_seed").mkdir()
    (out / "prediction_figures").mkdir()
    metric_rows, delta_rows = [], []
    previous_metric = pd.read_csv(old_root / "per_seed_metrics_full_1_315.csv")
    method_cols = {"baseline": "baseline_pred_vb",
                   "original_multiplicative_oor_pga": "original_multiplicative_oor_pga_pred_vb",
                   "early_reference_persistent_oor_pga": "early_reference_persistent_oor_pga_pred_vb"}
    trigger_table = pd.read_csv(out / "trigger_summary_full_1_315.csv", keep_default_na=False)
    for source, target in baseline.PAIRS:
        per_method = {m: [] for m in baseline.METHODS}
        trigger = str(trigger_table[(trigger_table.source == source) &
                                    (trigger_table.target == target)].iloc[0].trigger_cut)
        for seed in baseline.SEEDS:
            for base_method in baseline.METHODS:
                name = f"{source}_to_{target}_seed_{seed}_{base_method}_full_1_315.csv"
                frame = pd.read_csv(out / "unlabeled_predictions" / name)
                if frame.cut_index.tolist() != baseline.CUTS:
                    raise ValueError(f"Cut mismatch: {name}")
                per_method[base_method].append(frame)
                evaluated = frame.copy()
                evaluated.insert(1, "true_vb", truth[target])
                for key, col in method_cols.items():
                    evaluated[f"{key}_signed_error"] = frame[col].to_numpy(float) - truth[target]
                    for segment, lo, hi in SEGS:
                        values = metric(truth[target][lo - 1:hi], frame[col].to_numpy(float)[lo - 1:hi])
                        metric_rows.append({"source": source, "target": target, "seed": seed,
                                            "base_method": base_method, "variant": key,
                                            "scope": "full_1_315", "segment": segment,
                                            "cut_first": lo, "cut_last": hi, **values})
                        if segment == "full" and key in ("baseline", "original_multiplicative_oor_pga"):
                            old_method = base_method if key == "baseline" else base_method + "_oor_pga"
                            prior_row = previous_metric[(previous_metric.source == source) &
                                                        (previous_metric.target == target) &
                                                        (previous_metric.seed == seed) &
                                                        (previous_metric.method == old_method) &
                                                        (previous_metric.segment == "full")]
                            if len(prior_row) != 1 or abs(values["RMSE"] - prior_row.RMSE.iloc[0]) > 1e-10:
                                raise ValueError("Prior baseline/OOR metric not reproduced")
                for comparison in ("original_multiplicative_oor_pga", "early_reference_persistent_oor_pga"):
                    for segment, lo, hi in SEGS:
                        y = truth[target][lo - 1:hi]
                        a = metric(y, frame.baseline_pred_vb.to_numpy(float)[lo - 1:hi])
                        b = metric(y, frame[method_cols[comparison]].to_numpy(float)[lo - 1:hi])
                        delta_rows.append({"source": source, "target": target, "seed": seed,
                                           "base_method": base_method, "variant": comparison,
                                           "scope": "full_1_315", "segment": segment,
                                           "cut_first": lo, "cut_last": hi,
                                           "baseline_RMSE": a["RMSE"], "corrected_RMSE": b["RMSE"],
                                           "delta_RMSE": b["RMSE"] - a["RMSE"],
                                           "delta_MAE": b["MAE"] - a["MAE"],
                                           "delta_signed_bias": b["signed_bias"] - a["signed_bias"]})
                csv_write(out / "evaluated_per_seed" / name, evaluated)
        plot_predictions(out / "prediction_figures" /
                         f"{source}_to_{target}_predictions_full_1_315.png",
                         source, target, truth[target], per_method, trigger)
    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    csv_write(out / "per_seed_metrics_full_1_315.csv", metrics)
    csv_write(out / "paired_seed_delta_full_1_315.csv", deltas)
    summary = []
    for (source, target, method, variant, segment), group in metrics.groupby(
            ["source", "target", "base_method", "variant", "segment"], sort=True):
        row = {"source": source, "target": target, "base_method": method,
               "variant": variant, "segment": segment, "scope": "full_1_315",
               "n_seeds": 5, "n_cuts_per_seed": int(group.n.iloc[0])}
        for name in ("R2", "MAE", "RMSE", "signed_bias"):
            row[name + "_mean"] = float(group[name].mean())
            row[name + "_sd"] = float(group[name].std(ddof=1))
        summary.append(row)
    csv_write(out / "five_seed_summary_full_1_315.csv", pd.DataFrame(summary))
    paired_summary = []
    for (source, target, method, variant, segment), group in deltas.groupby(
            ["source", "target", "base_method", "variant", "segment"], sort=True):
        row = {"source": source, "target": target, "base_method": method,
               "variant": variant, "segment": segment, "scope": "full_1_315",
               "n_seeds": 5, "improved_seed_count_RMSE": int((group.delta_RMSE < 0).sum())}
        for name in ("delta_RMSE", "delta_MAE", "delta_signed_bias"):
            row[name + "_mean"] = float(group[name].mean())
            row[name + "_sd"] = float(group[name].std(ddof=1))
        paired_summary.append(row)
    csv_write(out / "paired_five_seed_delta_full_1_315.csv", pd.DataFrame(paired_summary))
    json_write(out / "evaluation_manifest_full_1_315.json", {
        "prediction_lock_sha256": stage.sha(lock_path),
        "target_label_reads_started_after_prediction_lock": True,
        "target_wear_sha256": {target: stage.sha(raw_root / f"{target}_wear.csv") for target in stage.TOOLS},
        "per_seed_metrics_sha256": stage.sha(out / "per_seed_metrics_full_1_315.csv"),
        "paired_seed_delta_sha256": stage.sha(out / "paired_seed_delta_full_1_315.csv"),
        "command": "python run_early_reference_persistent_oor_pga.py evaluate"})
    print(f"Evaluated 60 frozen corrected predictions and retained original OOR comparison: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "prepare", "evaluate"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-root", type=Path, default=old.BASELINE)
    parser.add_argument("--old-oor-root", type=Path, default=old.ROOT)
    parser.add_argument("--cache-root", type=Path, default=old.CACHE)
    parser.add_argument("--raw-root", type=Path, default=old.RAW)
    args = parser.parse_args()
    out, base_root, old_root, cache_root, raw_root = (
        p.resolve() for p in (args.out_root, args.baseline_root, args.old_oor_root,
                              args.cache_root, args.raw_root))
    if args.phase == "audit":
        audit_phase(out, base_root, old_root, cache_root)
    elif args.phase == "prepare":
        prepare_phase(out, base_root, old_root, cache_root, raw_root)
    else:
        evaluate_phase(out, base_root, old_root, raw_root)


if __name__ == "__main__":
    main()
