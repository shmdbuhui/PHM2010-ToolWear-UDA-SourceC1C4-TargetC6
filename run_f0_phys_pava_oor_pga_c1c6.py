"""Apply the frozen multiplicative STFT-48 OOR-PGA after C1->C6 F0/PAVA.

Prepare saves only unlabeled predictions. Evaluate opens C6 wear after the
prediction lock exists. Run from upstream-reproduction with Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_oor_pga_full_1_315 import TL, TH, TAU, features48, source_power


ART = Path("artifacts")
F0 = ART / "nonnegative_increment_20260926"
PAVA = ART / "f0_pava_offline_20260926"
OLD_OOR = ART / "oor_pga_full_1_315_zscore_20260925"
CACHE = ART / "five_seed_paired/feature_cache"
OUT = ART / "f0_phys_pava_oor_pga_c1c6_20260927"
SOURCE_WEAR = Path(r"E:\QLP\source\source_mill\c1_wear.csv")
TARGET_WEAR = Path(r"E:\QLP\source\source_mill\c6_wear.csv")
CUTS = np.arange(1, 316)
SEEDS = range(42, 47)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["cut_index", *columns])
    if not np.array_equal(frame.cut_index.to_numpy(), CUTS):
        raise ValueError(f"Wrong cut index: {path}")
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError(f"Nonfinite values: {path}")
    return frame


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    f0_lock_path = F0 / "prediction_lock.json"
    pava_lock_path = PAVA / "prediction_lock.json"
    f0_lock = json.loads(f0_lock_path.read_text(encoding="utf-8"))
    pava_lock = json.loads(pava_lock_path.read_text(encoding="utf-8"))
    if pava_lock["upstream_lock_sha256"] != sha(f0_lock_path):
        raise ValueError("F0/PAVA upstream lock changed")
    cache_manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if cache_manifest["cuts"] != CUTS.tolist():
        raise ValueError("STFT cache cut mapping changed")
    # This is the project's existing OOR feature formula: 6 channels x 8
    # consecutive 16-row frequency bands, averaged across rows and time.
    xs = features48(CACHE / "c1_stft.npy", cache_manifest["tools"]["c1"]["feature_sha256"])
    xt = features48(CACHE / "c6_stft.npy", cache_manifest["tools"]["c6"]["feature_sha256"])
    mu, sd = xs.mean(axis=0), xs.std(axis=0)
    scale = np.maximum(sd, 1e-12)
    zs, zt = (xs - mu) / scale, (xt - mu) / scale
    lower, upper = zs.min(axis=0), zs.max(axis=0)
    flags = (zt < lower) | (zt > upper)
    count = flags.sum(axis=1)
    rate = count / 48.0
    gate = np.clip((rate - TL) / (TH - TL), 0.0, 1.0)
    old_score = checked_frame(OLD_OOR / "oor_scores/c1_to_c6_oor_score_full_1_315.csv",
                              ["oor_rate_48", "gate", "multiplicative_factor"])
    if not np.allclose(rate, old_score.oor_rate_48.to_numpy(), rtol=0, atol=2e-15) or not np.allclose(
            gate, old_score.gate.to_numpy(), rtol=0, atol=2e-15):
        raise ValueError("Recomputed OOR differs from frozen STFT-48 OOR")
    source_hash = sha(SOURCE_WEAR)
    power = source_power(SOURCE_WEAR, source_hash)
    old_power = pd.read_csv(OLD_OOR / "source_pga_exponents.csv").query("source == 'c1'").iloc[0]
    if source_hash != old_power.source_wear_sha256 or abs(power["m"] - old_power.m) > 1e-12:
        raise ValueError("C1 source PGA fit changed")
    hits = np.flatnonzero((TAU > 0) & (gate >= 1 - 1e-12))
    trigger = int(hits[0]) if len(hits) else None
    factor = np.ones(315, dtype=float)
    if trigger is not None:
        active = TAU > TAU[trigger]
        factor[active] = (TAU[active] / TAU[trigger]) ** power["m"]
    if not np.allclose(factor, old_score.multiplicative_factor.to_numpy(), rtol=0, atol=2e-15):
        raise ValueError("Recomputed PGA factor differs from frozen rule")
    support = pd.DataFrame({"feature_index": np.arange(48), "channel": np.repeat(np.arange(6), 8),
                            "band": np.tile(np.arange(8), 6), "source_mean": mu,
                            "source_std": sd, "source_z_min": lower, "source_z_max": upper})
    manifest = json.loads((F0 / "plots/manifest.json").read_text(encoding="utf-8"))
    plot_record = next(row for row in manifest if row["direction"] == "c1_to_c6")
    if sha(Path(plot_record["path"])) != plot_record["sha256"]:
        raise ValueError("Original C1->C6 figure changed")
    tables, sources, checkpoints = [], {}, {}
    for seed in SEEDS:
        name = f"c1_to_c6_seed_{seed}.csv"
        src = F0 / f"c1_to_c6/seed_{seed}/F0/target_predictions_unscored.csv"
        checkpoint = src.parent / "final.pth"
        pava_path = PAVA / "per_seed_unscored" / name
        for path in (src, checkpoint):
            if sha(path) != f0_lock["files"][str(path.resolve())]:
                raise ValueError(f"F0 prediction/checkpoint lock mismatch: {path}")
        if sha(pava_path) != pava_lock["outputs"][str(pava_path.resolve())]:
            raise ValueError(f"PAVA lock mismatch: {pava_path}")
        raw = checked_frame(src, ["p_vb"]).p_vb.to_numpy(float)
        pava = checked_frame(pava_path, ["raw_pred", "pava_pred"])
        if not np.allclose(raw, pava.raw_pred.to_numpy(float), rtol=0, atol=1e-10):
            raise ValueError(f"F0 raw predictions changed: {seed}")
        base = pava.pava_pred.to_numpy(float)
        if np.any(np.diff(base) < -1e-10):
            raise ValueError(f"PAVA baseline declines: {seed}")
        corrected = base * factor
        if not np.isfinite(corrected).all():
            raise ValueError(f"Nonfinite OOR-PGA prediction: {seed}")
        tables.append((seed, pd.DataFrame({"cut_index": CUTS, "y_phys_pava": base,
            "oor_score": rate, "oor_flag_count_48": count, "oor_gate": gate,
            "pga_factor": factor, "cumulative_correction": corrected - base,
            "y_new": corrected})))
        sources[str(seed)] = {"f0_prediction": str(src.resolve()), "f0_sha256": sha(src),
                              "pava_prediction": str(pava_path.resolve()), "pava_sha256": sha(pava_path)}
        checkpoints[str(seed)] = sha(checkpoint)
    mean_base = np.mean([table.y_phys_pava.to_numpy() for _, table in tables], axis=0)
    scored_baseline = [checked_frame(PAVA / "per_seed_scored" / f"c1_to_c6_seed_{seed}.csv",
                                     ["pava_pred"]).pava_pred.to_numpy(float) for seed in SEEDS]
    if not np.allclose(mean_base, np.mean(scored_baseline, axis=0), rtol=0, atol=1e-12):
        raise ValueError("Original red-curve baseline differs pointwise")
    mean_new = mean_base * factor
    tables.append(("mean", pd.DataFrame({"cut_index": CUTS, "y_phys_pava": mean_base,
        "oor_score": rate, "oor_flag_count_48": count, "oor_gate": gate,
        "pga_factor": factor, "cumulative_correction": mean_new - mean_base,
        "y_new": mean_new})))
    OUT.mkdir(parents=True)
    (OUT / "predictions_unlabeled").mkdir()
    support.to_csv(OUT / "source_48_feature_support.csv", index=False, float_format="%.17g")
    files = {}
    for seed, table in tables:
        path = OUT / "predictions_unlabeled" / f"c1_to_c6_{seed}.csv"
        table.to_csv(path, index=False, float_format="%.17g")
        files[path.name] = sha(path)
    write_json(OUT / "prediction_lock_before_c6_labels.json", {
        "formula": "multiplicative: y_new=y_phys_pava*factor; factor=1 through first tau>0 full-gate cut, then (tau/t_oor)^m",
        "cumulative_correction_definition": "y_new-y_phys_pava; not an additive cumulative sum",
        "source": "c1", "target": "c6", "seeds": list(SEEDS), "cuts": [1, 315],
        "TL": TL, "TH": TH, "m": power["m"], "m_fit": power["fit"],
        "trigger_cut": trigger + 1 if trigger is not None else None,
        "first_changed_cut": trigger + 2 if trigger is not None and trigger < 314 else None,
        "source_wear_sha256": source_hash, "source_feature_sha256": sha(CACHE / "c1_stft.npy"),
        "target_feature_sha256": sha(CACHE / "c6_stft.npy"),
        "f0_lock_sha256": sha(f0_lock_path), "pava_lock_sha256": sha(pava_lock_path),
        "original_figure_sha256": plot_record["sha256"], "original_figure_ylim": plot_record["ylim"],
        "baseline_max_abs_difference_vb": float(np.max(np.abs(mean_base - np.mean(scored_baseline, axis=0)))),
        "oor_rate_max_abs_difference": float(np.max(np.abs(rate - old_score.oor_rate_48.to_numpy()))),
        "factor_min_step": float(np.diff(factor).min()), "sources": sources,
        "checkpoint_sha256": checkpoints, "unlabeled_prediction_sha256": files,
        "target_label_reads": 0})
    print(f"Saved five seed predictions and their plotted mean before C6 label read: {OUT}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {"R2": float(1 - np.square(err).sum() / np.square(y - y.mean()).sum()),
            "RMSE": float(np.sqrt(np.mean(np.square(err)))), "MAE": float(np.mean(np.abs(err))),
            "mean_bias": float(err.mean()), "decline_count": int(np.count_nonzero(np.diff(p) < -1e-10))}


def evaluate() -> None:
    lock_path = OUT / "prediction_lock_before_c6_labels.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (OUT / "comparison_metrics.csv").exists():
        raise FileExistsError("Evaluation already exists")
    for name, digest in lock["unlabeled_prediction_sha256"].items():
        if sha(OUT / "predictions_unlabeled" / name) != digest:
            raise ValueError(f"Prediction lock mismatch: {name}")
    if sha(TARGET_WEAR) != "7a80b692fd60a75c71ee4edbba9115cec3052c7d0205376b0ee876ddcc0fee7f":
        raise ValueError("C6 wear file changed from original evaluation")
    from run_single_source_pairs import wear_labels
    truth = wear_labels(TARGET_WEAR.parent, "c6").astype(float)
    if len(truth) != 315 or not np.isfinite(truth).all():
        raise ValueError("Invalid C6 labels")
    (OUT / "evaluated").mkdir()
    records = []
    for seed in [*SEEDS, "mean"]:
        name = f"c1_to_c6_{seed}.csv"
        frame = checked_frame(OUT / "predictions_unlabeled" / name,
                              ["y_phys_pava", "oor_score", "oor_gate", "pga_factor",
                               "cumulative_correction", "y_new"])
        evaluated = frame.copy()
        evaluated.insert(1, "true_vb", truth)
        evaluated.to_csv(OUT / "evaluated" / name, index=False, float_format="%.17g")
        for method, col in (("F0_phys_PAVA", "y_phys_pava"), ("F0_phys_PAVA_OOR_PGA", "y_new")):
            for segment, start in (("full_1_315", 0), ("last_30_percent_221_315", 220)):
                records.append({"seed": seed, "method": method, "segment": segment,
                                "cut_first": start + 1, "cut_last": 315, "n": 315 - start,
                                **metrics(truth[start:], frame[col].to_numpy(float)[start:])})
    pd.DataFrame(records).to_csv(OUT / "comparison_metrics.csv", index=False, float_format="%.17g")
    mean = checked_frame(OUT / "predictions_unlabeled/c1_to_c6_mean.csv", ["y_new"])
    ylim = lock["original_figure_ylim"]
    fig, ax = plt.subplots(figsize=(9, 6.6))
    ax.set_box_aspect(0.70)
    ax.plot(CUTS, truth, color="black", linewidth=2, label="True VB")
    ax.plot(CUTS, mean.y_new, color="#d04b3f", linewidth=1.55, label="F0 + physics + PAVA + OOR-PGA")
    ax.set_xlim(1, 315)
    ax.set_ylim(*ylim)
    ax.set_title("C1→C6")
    ax.set_xlabel("Cut index")
    ax.set_ylabel("VB")
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.subplots_adjust(left=.11, right=.97, bottom=.12, top=.88)
    fig.canvas.draw()
    box = ax.get_window_extent(fig.canvas.get_renderer())
    ratio = float(box.width / box.height)
    if abs(ratio - 1.4285714285714288) > 1e-10:
        raise ValueError("Figure axes box differs from original")
    plot_path = OUT / "C1_to_C6_F0_phys_PAVA_OOR_PGA.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    write_json(OUT / "evaluation_manifest.json", {
        "prediction_lock_sha256": sha(lock_path), "target_wear_sha256": sha(TARGET_WEAR),
        "figure_sha256": sha(plot_path), "metrics_sha256": sha(OUT / "comparison_metrics.csv"),
        "xlim": [1, 315], "ylim": ylim, "axes_width_height_ratio": ratio,
        "max_y_new": float(mean.y_new.max()), "figure_clips_prediction": bool(mean.y_new.max() > ylim[1])})
    print(pd.DataFrame(records).query("seed == 'mean'").to_string(index=False))
    print(f"Saved: {plot_path}")


def compare_plot() -> None:
    """Plot the locked five-seed mean before and after OOR-PGA with C6 truth."""
    lock_path = OUT / "prediction_lock_before_c6_labels.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    evaluated_path = OUT / "evaluated/c1_to_c6_mean.csv"
    prediction_path = OUT / "predictions_unlabeled/c1_to_c6_mean.csv"
    if sha(prediction_path) != lock["unlabeled_prediction_sha256"][prediction_path.name]:
        raise ValueError("Five-seed mean prediction lock mismatch")
    frame = checked_frame(evaluated_path, ["true_vb", "y_phys_pava", "y_new"])
    frozen = checked_frame(prediction_path, ["y_phys_pava", "y_new"])
    for column in ("y_phys_pava", "y_new"):
        if not np.allclose(frame[column], frozen[column], rtol=0, atol=1e-12):
            raise ValueError(f"Evaluated prediction differs from locked output: {column}")
    ylim = lock["original_figure_ylim"]
    fig, ax = plt.subplots(figsize=(9, 6.6))
    ax.set_box_aspect(0.70)
    ax.plot(CUTS, frame.true_vb, color="black", linewidth=2, label="True VB")
    ax.plot(CUTS, frame.y_phys_pava, color="#d04b3f", linewidth=1.55,
            label="F0 + physics + PAVA")
    ax.plot(CUTS, frame.y_new, color="#2563a6", linewidth=1.55,
            label="+ OOR-PGA")
    ax.set_xlim(1, 315)
    ax.set_ylim(*ylim)
    ax.set_title("C1→C6: before and after OOR-PGA")
    ax.set_xlabel("Cut index")
    ax.set_ylabel("VB")
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
    fig.subplots_adjust(left=.11, right=.97, bottom=.12, top=.88)
    fig.canvas.draw()
    box = ax.get_window_extent(fig.canvas.get_renderer())
    ratio = float(box.width / box.height)
    if abs(ratio - 1.4285714285714288) > 1e-10:
        raise ValueError("Figure axes box differs from original")
    path = OUT / "C1_to_C6_before_after_OOR_PGA.png"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    write_json(OUT / "before_after_figure_manifest.json", {
        "plot_sha256": sha(path), "prediction_lock_sha256": sha(lock_path),
        "evaluated_mean_sha256": sha(evaluated_path), "xlim": [1, 315], "ylim": ylim,
        "axes_width_height_ratio": ratio,
        "curves": ["true_vb", "y_phys_pava", "y_new"]})
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "compare_plot"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "evaluate":
        evaluate()
    else:
        compare_plot()
