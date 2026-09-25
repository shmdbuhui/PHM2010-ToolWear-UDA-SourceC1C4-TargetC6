"""Frozen-baseline multiplicative OOR-PGA on PHM2010 STFT, with separate evaluation.

Run `prepare` first. It reads source wear only and saves all target predictions before
`evaluate` opens any target wear file. This is a 48-feature STFT port, not the
43-feature EEMD method or an OOR-PGA-Like additive correction.
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
from scipy.stats import theilslopes

import analyze_zscore_stage_errors as stage
import run_full_1_315_baseline_protocol as baseline
import run_single_source_pairs as base


ROOT = Path("artifacts/oor_pga_full_1_315_zscore_20260925")
BASELINE = Path("artifacts/full_1_315_baseline_zscore_20260925")
CACHE = Path("artifacts/five_seed_paired/feature_cache")
RAW = Path(r"E:\QLP\source\source_mill")
TAU = np.linspace(0.0, 1.0, 315)
TL, TH = 3.0 / 48.0, 6.0 / 48.0
PREDICTION_COLUMNS = {"source_only": "source_only_pred_vb",
                      "source_only_oor_pga": "source_only_oor_pga_pred_vb",
                      "daregram": "daregram_pred_vb",
                      "daregram_oor_pga": "daregram_oor_pga_pred_vb"}
SEGMENTS = (("full", 1, 315), ("early", 1, 105), ("middle", 106, 210), ("late", 211, 315))


def write_json(path: Path, item: dict) -> None:
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def features48(path: Path, expected_sha: str) -> np.ndarray:
    if stage.sha(path) != expected_sha:
        raise ValueError(f"STFT cache hash mismatch: {path}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (315, 6, 128, 128) or x.dtype != np.float32:
        raise ValueError(f"STFT shape/dtype mismatch: {path}")
    # Axis 2 is resized frequency, axis 3 is resized time. Eight contiguous
    # 16-row bands of the full (uncropped) 0..25 kHz STFT per channel.
    values = np.asarray(x.reshape(315, 6, 8, 16, 128).mean(axis=(3, 4)), dtype=np.float64)
    if values.shape != (315, 6, 8) or not np.isfinite(values).all():
        raise ValueError(f"Nonfinite 48-feature STFT: {path}")
    return values.reshape(315, 48)


def source_power(path: Path, expected_sha: str) -> dict:
    if stage.sha(path) != expected_sha:
        raise ValueError(f"Source wear hash mismatch: {path}")
    wear = base.wear_labels(path.parent, path.name.split("_")[0]).astype(np.float64)
    valid = (TAU > 0) & (wear > 0) & np.isfinite(wear)
    if valid.sum() < 3:
        raise ValueError("Insufficient source labels for source-only PGA fit")
    slope, intercept, low, high = theilslopes(np.log(wear[valid]), np.log(TAU[valid]))
    if not np.isfinite([slope, intercept, low, high]).all():
        raise ValueError("Nonfinite PGA exponent")
    return {"source": path.name.split("_")[0], "source_wear_file": str(path.resolve()),
            "source_wear_sha256": expected_sha, "fit": "Theil-Sen log(wear) vs log(tau)",
            "n_fit": int(valid.sum()), "m": float(slope), "A": float(np.exp(intercept)),
            "m_low_95": float(low), "m_high_95": float(high),
            "target_labels_used": False}


def check_baseline_lock(root: Path) -> pd.DataFrame:
    lock = stage.json_read(root / "protocol_lock_full_1_315.json")
    if lock["status"] != "audited_baseline_frozen_before_OOR_PGA" or lock["checkpoint_count"] != 60:
        raise ValueError("Full-cut baseline has not been frozen")
    for filename, key in (("checkpoint_audit_full_1_315.csv", "checkpoint_audit_sha256"),
                          ("per_seed_metrics_full_1_315.csv", "per_seed_metrics_sha256"),
                          ("five_seed_summary_full_1_315.csv", "five_seed_summary_sha256")):
        if stage.sha(root / filename) != lock[key]:
            raise ValueError(f"Baseline lock file changed: {filename}")
    audit = pd.read_csv(root / "checkpoint_audit_full_1_315.csv")
    if len(audit) != 60 or not (audit.prediction_count == 315).all():
        raise ValueError("Incomplete full-cut baseline audit")
    return audit


def prepare(out: Path, frozen: Path, cache: Path, raw: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Prepare requires a new independent directory: {out}")
    audit = check_baseline_lock(frozen)
    cache_manifest = stage.json_read(cache / "manifest.json")
    if stage.sha(cache / "manifest.json") != stage.json_read(frozen / "protocol_lock_full_1_315.json")["cache_manifest_sha256"]:
        raise ValueError("Feature cache manifest changed")
    audit_index = {(r.source, r.target, int(r.seed), r.method): r for r in audit.itertuples()}
    source_m = {}
    for source in stage.TOOLS:
        r = audit_index[source, next(t for t in stage.TOOLS if t != source), 42, "source_only"]
        source_m[source] = source_power(raw / f"{source}_wear.csv", r.source_wear_sha256)
    output_rows, factors, trigger_rows, feature_rows, predictions = [], [], [], [], []
    for source, target in baseline.PAIRS:
        record = audit_index[source, target, 42, "source_only"]
        xs = features48(cache / f"{source}_stft.npy", record.source_stft_sha256)
        xt = features48(cache / f"{target}_stft.npy", record.target_stft_sha256)
        # Source-fitted feature Z-score, then source min/max support, as in the
        # EEMD OOR definition. Positive affine scaling leaves flags unchanged.
        mu = xs.mean(axis=0)
        sd = xs.std(axis=0)
        scale = np.maximum(sd, 1e-12)
        zs, zt = (xs - mu) / scale, (xt - mu) / scale
        lower, upper = zs.min(axis=0), zs.max(axis=0)
        flags = (zt < lower) | (zt > upper)
        count = flags.sum(axis=1)
        rate = count / 48.0
        gate = np.clip((rate - TL) / (TH - TL), 0.0, 1.0)
        hits = np.flatnonzero((TAU > 0) & (gate >= 1.0 - 1e-12))
        trigger_index = int(hits[0]) if len(hits) else None
        t_oor = float(TAU[trigger_index]) if trigger_index is not None else None
        factor = np.ones(315, dtype=np.float64)
        m = source_m[source]["m"]
        if trigger_index is not None:
            active = TAU > t_oor
            factor[active] = np.power(TAU[active] / t_oor, m)
            if factor[trigger_index] != 1.0 or not np.all(factor[:trigger_index + 1] == 1):
                raise ValueError("Multiplicative trigger rule failed")
        if not np.isfinite(factor).all():
            raise ValueError("Nonfinite PGA factor")
        score = pd.DataFrame({"cut_index": baseline.CUTS, "tau": TAU, "oor_flag_count_48": count,
                              "oor_rate_48": rate, "gate": gate, "fully_open": gate >= 1 - 1e-12,
                              "trigger_cut": trigger_index + 1 if trigger_index is not None else "not_triggered",
                              "t_oor": t_oor if t_oor is not None else np.nan,
                              "source_pga_m": m, "multiplicative_factor": factor})
        output_rows.append((source, target, score, flags))
        trigger_rows.append({"source": source, "target": target, "scope": "full_1_315",
                             "trigger_status": "triggered" if trigger_index is not None else "not_triggered",
                             "trigger_cut": trigger_index + 1 if trigger_index is not None else "not_triggered",
                             "t_oor": t_oor if t_oor is not None else "",
                             "first_cut_gate_fully_open_including_tau0": int(np.flatnonzero(gate >= 1 - 1e-12)[0] + 1)
                             if np.any(gate >= 1 - 1e-12) else "not_triggered",
                             "m": m, "TL": TL, "TH": TH,
                             "source_constant_feature_count": int((sd <= 1e-12).sum()),
                             "source_feature_min": float(xs.min()), "source_feature_max": float(xs.max()),
                             "target_feature_min": float(xt.min()), "target_feature_max": float(xt.max()),
                             "max_oor_flag_count": int(count.max()),
                             "factor_at_cut_315": float(factor[-1]), "target_label_reads": 0})
        for channel in range(6):
            for band in range(8):
                k = channel * 8 + band
                feature_rows.append({"source": source, "target": target, "channel": channel,
                                     "band": band, "feature_index": k,
                                     "resized_frequency_row_first": 16 * band,
                                     "resized_frequency_row_last": 16 * band + 15,
                                     "source_raw_min": float(xs[:, k].min()),
                                     "source_raw_max": float(xs[:, k].max()),
                                     "source_feature_mean": float(mu[k]),
                                     "source_feature_std": float(sd[k]),
                                     "source_z_min": float(lower[k]), "source_z_max": float(upper[k])})
        for seed in baseline.SEEDS:
            row = {"source": source, "target": target, "seed": seed, "scope": "full_1_315"}
            for method in baseline.METHODS:
                record = audit_index[source, target, seed, method]
                prediction_path = Path(record.original_prediction_file)
                if stage.sha(prediction_path) != record.original_prediction_sha256:
                    raise ValueError(f"Original prediction changed: {prediction_path}")
                # Crucial separation: only cut and prediction columns are read here.
                frame = pd.read_csv(prediction_path, usecols=["cut_index", "pred_vb"])
                if frame.cut_index.tolist() != baseline.CUTS:
                    raise ValueError(f"Missing/duplicate prediction cut: {prediction_path}")
                pred = frame.pred_vb.to_numpy(float)
                if not np.isfinite(pred).all():
                    raise ValueError(f"Nonfinite base prediction: {prediction_path}")
                row[PREDICTION_COLUMNS[method]] = pred
                row[PREDICTION_COLUMNS[method + "_oor_pga"]] = pred * factor
            table = pd.DataFrame({"cut_index": baseline.CUTS, "tau": TAU,
                                  "oor_rate_48": rate, "gate": gate,
                                  "trigger_cut": trigger_index + 1 if trigger_index is not None else "not_triggered",
                                  "source_pga_m": m, "multiplicative_factor": factor,
                                  **{key: row[key] for key in PREDICTION_COLUMNS.values()}})
            if not np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all():
                raise ValueError("Nonfinite unlabeled prediction output")
            predictions.append((source, target, seed, table))
    out.mkdir(parents=True)
    (out / "oor_scores").mkdir()
    (out / "unlabeled_predictions").mkdir()
    pd.DataFrame(source_m.values()).to_csv(out / "source_pga_exponents.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out / "source_48_feature_support.csv", index=False)
    pd.DataFrame(trigger_rows).to_csv(out / "triggers_and_factors_full_1_315.csv", index=False)
    for source, target, score, flags in output_rows:
        score.to_csv(out / "oor_scores" / f"{source}_to_{target}_oor_score_full_1_315.csv",
                     index=False, float_format="%.17g")
        pd.DataFrame(flags.astype(np.uint8), columns=[f"channel_{c}_band_{b}" for c in range(6)
                                                    for b in range(8)]).assign(cut_index=baseline.CUTS).to_csv(
            out / "oor_scores" / f"{source}_to_{target}_flags_48_full_1_315.csv", index=False)
    for source, target, seed, table in predictions:
        path = out / "unlabeled_predictions" / f"{source}_to_{target}_seed_{seed}_full_1_315.csv"
        table.to_csv(path, index=False, float_format="%.17g")
    inputs = {"baseline_lock_file": str((frozen / "protocol_lock_full_1_315.json").resolve()),
              "baseline_lock_sha256": stage.sha(frozen / "protocol_lock_full_1_315.json"),
              "baseline_checkpoint_audit_sha256": stage.sha(frozen / "checkpoint_audit_full_1_315.csv"),
              "source_pga_exponents_sha256": stage.sha(out / "source_pga_exponents.csv"),
              "trigger_table_sha256": stage.sha(out / "triggers_and_factors_full_1_315.csv"),
              "unlabeled_prediction_sha256": {p.name: stage.sha(p) for p in sorted((out / "unlabeled_predictions").glob("*.csv"))},
              "oor_score_sha256": {p.name: stage.sha(p) for p in sorted((out / "oor_scores").glob("*.csv"))},
              "source_feature_support_sha256": stage.sha(out / "source_48_feature_support.csv"),
              "STFT_cache_manifest_sha256": stage.sha(cache / "manifest.json"),
              "STFT_cache_manifest_file": str((cache / "manifest.json").resolve()),
              "EEMD_reference_local_file": str(Path(r"E:\QLP\EEMD-7.22\run_oor_triggered_pga.py")),
              "EEMD_reference_sha256": stage.sha(Path(r"E:\QLP\EEMD-7.22\run_oor_triggered_pga.py")),
              "PR_1_reference": "unavailable locally; GitHub network unreachable; no unverified implementation reused",
              "method": "multiplicative OOR-PGA STFT 48-feature port; no additive PGA-like",
              "frequency_axis": "six channels; full uncropped 0..25 kHz STFT; 129 native bins resized to 128 rows; eight contiguous 16-row bands; mean over band and time",
              "source_support": "source-fitted per-feature Z-score then source min/max; out-of-range flags on target unlabeled STFT",
              "TL": TL, "TH": TH, "threshold_status": "3/48 and 6/48 preserve flag counts, unvalidated transfer from 3/43 and 6/43",
              "trigger": "first tau>0 with gate fully open, K=1; no trigger keeps predictions unchanged",
              "tau": "(cut_index-1)/314", "factor": "1 through trigger; (tau/t_oor)^m afterward",
              "target_labels_opened_in_prepare": False,
              "command": "python run_oor_pga_full_1_315.py prepare"}
    write_json(out / "prediction_lock_before_target_labels.json", inputs)
    print(f"Saved 30 unlabeled corrected prediction files before target label reads: {out}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {"n": len(y), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))), "signed_bias": float(err.mean()),
            "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "MAPE_percent": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-8)) * 100)}


def evaluation_plot(path: Path, source: str, target: str, y: np.ndarray,
                    seed_tables: list[pd.DataFrame], score: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    cuts = np.arange(1, 316)
    axes[0].plot(cuts, y, c="black", lw=1.8, label="true VB")
    colors = {"source_only": "#2364a0", "source_only_oor_pga": "#70a6d1",
              "daregram": "#d47b24", "daregram_oor_pga": "#a53221"}
    for method, col in PREDICTION_COLUMNS.items():
        matrix = np.stack([f[col].to_numpy(float) for f in seed_tables])
        axes[0].plot(cuts, matrix.mean(axis=0), color=colors[method], lw=1.45,
                     label=f"{method} five-seed mean")
        axes[1].plot(cuts, (matrix - y[None, :]).mean(axis=0), color=colors[method], lw=1.2,
                     label=method)
    axes[0].set_ylabel("VB")
    axes[1].set_ylabel("prediction - truth")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[2].plot(cuts, score.oor_rate_48, color="#506070", lw=1.1, label="OOR rate / 48")
    axes[2].plot(cuts, score.gate, color="#9a3c80", lw=1.1, label="gate")
    axes[2].axhline(TH, color="#666666", ls=":", lw=0.8, label="TH=6/48")
    trigger = score.trigger_cut.iloc[0]
    for ax in axes:
        for edge in (105.5, 210.5):
            ax.axvline(edge, color="#777777", ls="--", lw=0.8)
        if str(trigger) != "not_triggered":
            ax.axvline(int(trigger) - 0.5, color="#b32e2e", ls=":", lw=1)
        ax.set_xlim(1, 315)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, frameon=False)
    axes[2].set(xlabel="target cut index", ylabel="OOR rate / gate")
    fig.suptitle(f"{source.upper()} -> {target.upper()} | full_1_315 | multiplicative STFT-48 OOR-PGA")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def evaluate(out: Path, frozen: Path, raw: Path) -> None:
    if not out.is_dir() or (out / "per_seed_metrics_full_1_315.csv").exists():
        raise FileExistsError("Evaluation requires completed prepare and no existing metrics")
    check_baseline_lock(frozen)
    lock = stage.json_read(out / "prediction_lock_before_target_labels.json")
    if stage.sha(frozen / "protocol_lock_full_1_315.json") != lock["baseline_lock_sha256"]:
        raise ValueError("Baseline changed since prepare")
    for folder, key in (("unlabeled_predictions", "unlabeled_prediction_sha256"),
                        ("oor_scores", "oor_score_sha256")):
        for name, digest in lock[key].items():
            if stage.sha(out / folder / name) != digest:
                raise ValueError(f"Frozen unlabeled file changed: {name}")
    audit = pd.read_csv(frozen / "checkpoint_audit_full_1_315.csv")
    truth = {}
    for target in stage.TOOLS:
        target_rows = audit[audit.target == target]
        label_hashes = set(target_rows.target_wear_sha256)
        if len(label_hashes) != 1 or stage.sha(raw / f"{target}_wear.csv") != next(iter(label_hashes)):
            raise ValueError(f"Target wear provenance mismatch: {target}")
        truth[target], _, info = stage.labels(raw, target)
        if info["cuts_1_315_with_label"] != 315:
            raise ValueError(f"Missing label: {target}")
    (out / "evaluated_per_seed").mkdir()
    (out / "figures").mkdir()
    metric_rows, paired_rows = [], []
    base_metrics = pd.read_csv(frozen / "per_seed_metrics_full_1_315.csv")
    for source, target in baseline.PAIRS:
        frames = []
        score = pd.read_csv(out / "oor_scores" / f"{source}_to_{target}_oor_score_full_1_315.csv")
        for seed in baseline.SEEDS:
            path = out / "unlabeled_predictions" / f"{source}_to_{target}_seed_{seed}_full_1_315.csv"
            frame = pd.read_csv(path)
            if frame.cut_index.tolist() != baseline.CUTS or not np.isfinite(frame[list(PREDICTION_COLUMNS.values())].to_numpy()).all():
                raise ValueError(f"Invalid corrected prediction: {path}")
            frames.append(frame)
            evaluated = frame.copy()
            evaluated.insert(1, "true_vb", truth[target])
            for method, col in PREDICTION_COLUMNS.items():
                evaluated[f"{method}_signed_error"] = frame[col].to_numpy(float) - truth[target]
                for segment, lo, hi in SEGMENTS:
                    result = metrics(truth[target][lo - 1:hi], frame[col].to_numpy(float)[lo - 1:hi])
                    metric_rows.append({"source": source, "target": target, "seed": seed,
                                        "method": method, "scope": "full_1_315", "segment": segment,
                                        "cut_first": lo, "cut_last": hi, **result})
                    if segment == "full" and method in baseline.METHODS:
                        earlier = base_metrics[(base_metrics.source == source) &
                                               (base_metrics.target == target) &
                                               (base_metrics.seed == seed) &
                                               (base_metrics.method == method)]
                        if len(earlier) != 1 or abs(result["RMSE"] - earlier.RMSE.iloc[0]) > 1e-10:
                            raise ValueError("Baseline RMSE changed after OOR preparation")
                if method in baseline.METHODS:
                    corrected = method + "_oor_pga"
                    for segment, lo, hi in SEGMENTS:
                        y = truth[target][lo - 1:hi]
                        p0 = frame[PREDICTION_COLUMNS[method]].to_numpy(float)[lo - 1:hi]
                        p1 = frame[PREDICTION_COLUMNS[corrected]].to_numpy(float)[lo - 1:hi]
                        a, b = metrics(y, p0), metrics(y, p1)
                        paired_rows.append({"source": source, "target": target, "seed": seed,
                                            "baseline_method": method, "scope": "full_1_315",
                                            "segment": segment, "cut_first": lo, "cut_last": hi,
                                            "baseline_RMSE": a["RMSE"], "oor_pga_RMSE": b["RMSE"],
                                            "delta_RMSE": b["RMSE"] - a["RMSE"],
                                            "baseline_MAE": a["MAE"], "oor_pga_MAE": b["MAE"],
                                            "delta_MAE": b["MAE"] - a["MAE"],
                                            "baseline_signed_bias": a["signed_bias"],
                                            "oor_pga_signed_bias": b["signed_bias"],
                                            "delta_signed_bias": b["signed_bias"] - a["signed_bias"]})
            evaluated.to_csv(out / "evaluated_per_seed" /
                             f"{source}_to_{target}_seed_{seed}_full_1_315.csv",
                             index=False, float_format="%.17g")
        evaluation_plot(out / "figures" / f"{source}_to_{target}_predictions_full_1_315.png",
                        source, target, truth[target], frames, score)
    per_seed = pd.DataFrame(metric_rows)
    paired = pd.DataFrame(paired_rows)
    per_seed.to_csv(out / "per_seed_metrics_full_1_315.csv", index=False)
    paired.to_csv(out / "paired_seed_delta_full_1_315.csv", index=False)
    summary = []
    for (source, target, method, segment), group in per_seed.groupby(
            ["source", "target", "method", "segment"], sort=True):
        row = {"source": source, "target": target, "method": method,
               "scope": "full_1_315", "segment": segment, "n_seeds": 5,
               "n_cuts_per_seed": int(group.n.iloc[0])}
        for metric in ("RMSE", "MAE", "signed_bias", "R2", "MAPE_percent"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "five_seed_summary_full_1_315.csv", index=False)
    paired_summary = []
    for (source, target, method, segment), group in paired.groupby(
            ["source", "target", "baseline_method", "segment"], sort=True):
        row = {"source": source, "target": target, "baseline_method": method,
               "scope": "full_1_315", "segment": segment, "n_seeds": 5,
               "improved_seed_count_RMSE": int((group.delta_RMSE < 0).sum())}
        for metric in ("delta_RMSE", "delta_MAE", "delta_signed_bias"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        paired_summary.append(row)
    pd.DataFrame(paired_summary).to_csv(out / "paired_five_seed_delta_full_1_315.csv", index=False)
    write_json(out / "evaluation_manifest_full_1_315.json", {
        "prediction_lock_sha256": stage.sha(out / "prediction_lock_before_target_labels.json"),
        "scope": "full_1_315", "label_reads_started_after_all_unlabeled_predictions_saved": True,
        "target_wear_sha256": {tool: stage.sha(raw / f"{tool}_wear.csv") for tool in stage.TOOLS},
        "per_seed_metrics_sha256": stage.sha(out / "per_seed_metrics_full_1_315.csv"),
        "paired_seed_delta_sha256": stage.sha(out / "paired_seed_delta_full_1_315.csv"),
        "command": "python run_oor_pga_full_1_315.py evaluate"})
    print(f"Evaluated four frozen methods in six directions, five seeds, full 1..315: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--out-root", type=Path, default=ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE)
    parser.add_argument("--cache-root", type=Path, default=CACHE)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out_root.resolve(), args.baseline_root.resolve(),
                args.cache_root.resolve(), args.raw_root.resolve())
    else:
        evaluate(args.out_root.resolve(), args.baseline_root.resolve(), args.raw_root.resolve())


if __name__ == "__main__":
    main()
