"""Frozen C1->C6 DARE-GRAM plus causal five-point quadratic recursion.

Run predict and score separately. predict reads C6 wear CSV only through row 5;
score is the sole function permitted to read the remaining target labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "artifacts/selected_checkpoint_feature_audit_20260926/predictions_per_cut.csv"
CHECKPOINT = ROOT / "artifacts/multi_window_25_50_75_20260926/c1_to_c6/seed_42/daregram/final.pth"
EXPECTED_CHECKPOINT_SHA256 = "d7e8d545e2069d64ab5f4edcfccf9d9bff86dcb284cf42dcf725d9f46a6b23ed"
DEFAULT_OUTPUT = ROOT / "artifacts/c6_five_point_model_fusion_20260926"
CUTS = list(range(6, 316))
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
DELTAS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
SOURCE_STARTS = (1, 31, 61, 91, 121, 151, 181, 211)
SEGMENTS = (("all_6_315", 6, 315), ("early_6_105", 6, 105),
            ("middle_106_210", 106, 210), ("late_211_315", 211, 315))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wear_value(row: dict, expected_cut: int) -> float:
    if set(row) != {"cut", "flute_1", "flute_2", "flute_3"} or int(row["cut"]) != expected_cut:
        raise ValueError(f"Invalid wear row at cut {expected_cut}")
    vb = float(np.mean([float(row[f"flute_{i}"]) for i in (1, 2, 3)]))
    if not np.isfinite(vb):
        raise ValueError(f"Nonfinite wear at cut {expected_cut}")
    return vb


def read_initial_five(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["cut", "flute_1", "flute_2", "flute_3"]:
            raise ValueError("Unexpected wear schema")
        return {cut: {"vb": wear_value(next(reader), cut)} for cut in range(1, 6)}


def read_all_wear_after_predictions(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["cut", "flute_1", "flute_2", "flute_3"]:
            raise ValueError("Unexpected wear schema")
        rows = list(reader)
    if len(rows) != 315:
        raise ValueError("Expected 315 wear rows")
    return {cut: wear_value(rows[cut - 1], cut) for cut in range(1, 316)}


def fit_one_step(input_cuts: list[int], input_vb: list[float], prediction_cut: int):
    x = np.asarray(input_cuts, dtype=float)
    u = x - x[-1]
    a2, a1, a0 = np.linalg.lstsq(np.column_stack((u * u, u, np.ones(5))),
                                np.asarray(input_vb, dtype=float), rcond=None)[0]
    return float(a2), float(a1), float(a0), float(a2 + a1 + a0)


def model_predictions() -> tuple[np.ndarray, np.ndarray]:
    # This audited export contains predictions only, unlike the original
    # predictions.csv, whose true_vb column would violate prediction isolation.
    frame = pd.read_csv(PREDICTIONS)
    expected = {"direction", "model_type", "seed", "domain", "tool_id", "cut_index",
                "pred_w25", "pred_center", "pred_w75", "pred_mean"}
    if set(frame) != expected:
        raise ValueError("Unexpected audited prediction schema")
    subset = frame.loc[(frame.direction == "c1_to_c6") &
                       (frame.model_type == "daregram") & (frame.seed == 42)]
    result = []
    for domain, tool in (("source", "c1"), ("target", "c6")):
        part = subset.loc[(subset.domain == domain) & (subset.tool_id == tool)]
        if part.cut_index.tolist() != list(range(1, 316)):
            raise ValueError(f"Missing/unsorted model predictions: {domain}")
        windows = part[["pred_w25", "pred_center", "pred_w75"]].to_numpy(float)
        mean = part.pred_mean.to_numpy(float)
        if not np.isfinite(windows).all() or not np.isfinite(mean).all() or \
                not np.allclose(windows.mean(axis=1), mean, atol=1e-10, rtol=0):
            raise ValueError(f"Invalid three-window mean: {domain}")
        result.append(mean)
    return result[0], result[1]


def recurse(initial: dict[int, float], model: np.ndarray, alpha: float,
            delta: float, last_cut: int = 315) -> pd.DataFrame:
    """Never accept future labels: initial has exactly five consecutive cuts."""
    first = min(initial)
    if list(initial) != list(range(first, first + 5)) or not np.isfinite(list(initial.values())).all():
        raise ValueError("Recursion requires exactly five consecutive initial labels")
    history = dict(initial)
    rows = []
    for cut in range(first + 5, last_cut + 1):
        window_cuts = list(range(cut - 5, cut))
        values = [history[x] for x in window_cuts]
        a2, a1, a0, q = fit_one_step(window_cuts, values, cut)
        b = float(model[cut - 1])
        correction = float(np.clip(q - b, -delta, delta))
        y = float(b + alpha * correction)
        if not np.isfinite([q, b, y]).all():
            raise ValueError(f"Nonfinite forecast at cut {cut}")
        row = {"cut_index": cut, "input_cut_start": cut - 5, "input_cut_end": cut - 1,
               "input_true_count": sum(x < first + 5 for x in window_cuts),
               "input_predicted_count": sum(x >= first + 5 for x in window_cuts),
               "coef_u2": a2, "coef_u1": a1, "coef_u0": a0,
               "q_t": q, "b_t": b, "alpha_t": alpha, "delta": delta,
               "clipped_q_minus_b": correction, "pred_vb": y}
        for j, (input_cut, value) in enumerate(zip(window_cuts, values), 1):
            row[f"input_cut_{j}"] = input_cut
            row[f"input_vb_{j}"] = value
            row[f"input_source_{j}"] = "true_initial" if input_cut < first + 5 else "predicted"
        rows.append(row)
        history[cut] = y
    return pd.DataFrame(rows)


def source_calibration(source_true: dict[int, float], source_model: np.ndarray) -> tuple[pd.DataFrame, dict]:
    records = []
    for alpha in ALPHAS:
        for delta in DELTAS:
            if alpha == 0 and delta != DELTAS[0]:
                continue
            errors = []
            for first in SOURCE_STARTS:
                initial = {cut: source_true[cut] for cut in range(first, first + 5)}
                trace = recurse(initial, source_model, alpha, delta)
                truth = np.asarray([source_true[cut] for cut in trace.cut_index], dtype=float)
                errors.append(float(np.mean(np.abs(trace.pred_vb.to_numpy() - truth))))
            records.append({"alpha": alpha, "delta": delta,
                            "mean_task_mae": float(np.mean(errors)),
                            "start_1_mae": errors[0],
                            **{f"start_{first}_mae": error
                               for first, error in zip(SOURCE_STARTS, errors)}})
    grid = pd.DataFrame(records).sort_values(["mean_task_mae", "alpha", "delta"],
                                               kind="stable").reset_index(drop=True)
    selected = grid.iloc[0]
    parameters = {"alpha": float(selected.alpha), "delta": float(selected.delta),
                  "selection_metric": "unweighted mean of source-task MAE",
                  "source_task_initial_label_starts": list(SOURCE_STARTS),
                  "source_task_end_cut": 315,
                  "source_label_scope": "C1 full 1..315; each simulated recursion sees only its first five labels",
                  "target_suffix_labels_used_for_selection": False,
                  "source_model_prediction_limitation": "C1 model outputs are in-sample for the frozen checkpoint"}
    return grid, parameters


def predict(args: argparse.Namespace) -> None:
    out = args.out_root
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    if not CHECKPOINT.is_file() or not PREDICTIONS.is_file():
        raise FileNotFoundError("Frozen checkpoint or audited model predictions missing")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from audited seed-42 DARE-GRAM")
    source_model, target_model = model_predictions()
    # C1 is the source domain and may supply all training/simulation labels.
    source_true = read_all_wear_after_predictions(args.source_wear)
    grid, parameters = source_calibration(source_true, source_model)
    # This function consumes only the header and rows 1..5 of C6 wear.
    initial = read_initial_five(args.target_wear)
    initial_vb = {cut: row["vb"] for cut, row in initial.items()}
    fused = recurse(initial_vb, target_model, parameters["alpha"], parameters["delta"])
    curve = recurse(initial_vb, target_model, 1.0, float("inf"))
    assert fused.cut_index.tolist() == CUTS and curve.cut_index.tolist() == CUTS
    assert (fused.input_true_count.to_numpy() == np.clip(11 - np.asarray(CUTS), 0, 5)).all()
    out.mkdir(parents=True)
    grid.to_csv(out / "source_parameter_grid.csv", index=False, float_format="%.17g")
    write_json(out / "selected_parameters.json", parameters)
    pd.DataFrame({"cut_index": list(initial_vb), "true_vb_initial": list(initial_vb.values())}).to_csv(
        out / "initial_five_true_vb.csv", index=False, float_format="%.17g")
    fused.to_csv(out / "fusion_trace_unscored.csv", index=False, float_format="%.17g")
    curve.to_csv(out / "curve_only_trace_unscored.csv", index=False, float_format="%.17g")
    write_json(out / "prediction_lock.json", {
        "prediction_complete": True, "cuts": CUTS, "C6_true_VB_read": [1, 2, 3, 4, 5],
        "C6_suffix_true_VB_read_before_prediction_complete": [],
        "source_parameter_grid_sha256": sha256(out / "source_parameter_grid.csv"),
        "selected_parameters_sha256": sha256(out / "selected_parameters.json"),
        "initial_five_sha256": sha256(out / "initial_five_true_vb.csv"),
        "fusion_trace_sha256": sha256(out / "fusion_trace_unscored.csv"),
        "curve_only_trace_sha256": sha256(out / "curve_only_trace_unscored.csv"),
        "audited_model_predictions_sha256": sha256(PREDICTIONS),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "source_wear_sha256": sha256(args.source_wear),
        "target_wear_sha256": None})
    print(json.dumps({"selected": parameters, "prediction_count": len(fused)}, ensure_ascii=False))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"n": len(y), "R2": float(r2_score(y, p)),
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "mean_signed_error": float(np.mean(p - y))}


def score(args: argparse.Namespace) -> None:
    out = args.out_root
    if (out / "comparison_metrics.csv").exists():
        raise FileExistsError("Scored output already exists")
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    paths = {"source_parameter_grid_sha256": out / "source_parameter_grid.csv",
             "selected_parameters_sha256": out / "selected_parameters.json",
             "initial_five_sha256": out / "initial_five_true_vb.csv",
             "fusion_trace_sha256": out / "fusion_trace_unscored.csv",
             "curve_only_trace_sha256": out / "curve_only_trace_unscored.csv",
             "audited_model_predictions_sha256": PREDICTIONS,
             "checkpoint_sha256": CHECKPOINT,
             "source_wear_sha256": args.source_wear}
    if not lock["prediction_complete"] or lock["cuts"] != CUTS or \
            lock["C6_suffix_true_VB_read_before_prediction_complete"]:
        raise ValueError("Prediction lock is incomplete")
    for key, path in paths.items():
        if sha256(path) != lock[key]:
            raise ValueError(f"Locked input changed: {path}")
    fused = pd.read_csv(out / "fusion_trace_unscored.csv")
    curve = pd.read_csv(out / "curve_only_trace_unscored.csv")
    _, target_model = model_predictions()
    if fused.cut_index.tolist() != CUTS or curve.cut_index.tolist() != CUTS or \
            not np.isfinite(fused.pred_vb).all() or not np.isfinite(curve.pred_vb).all() or \
            not np.allclose(fused.b_t, target_model[5:], atol=1e-12, rtol=0):
        raise ValueError("Frozen traces or model baseline are invalid")
    # The first read of C6 cut 6..315 occurs only after lock verification.
    target_true = read_all_wear_after_predictions(args.target_wear)
    initial = pd.read_csv(out / "initial_five_true_vb.csv")
    if initial.cut_index.tolist() != list(range(1, 6)) or any(
            not np.isclose(row.true_vb_initial, target_true[row.cut_index], atol=1e-11)
            for row in initial.itertuples(index=False)):
        raise ValueError("Initial target labels differ from locked predictions")
    frame = pd.DataFrame({"cut_index": CUTS,
                          "true_vb": [target_true[cut] for cut in CUTS],
                          "daregram_b_t": target_model[5:],
                          "curve_only_q_t": curve.pred_vb,
                          "fusion_pred_vb": fused.pred_vb})
    frame.to_csv(out / "predictions_scored.csv", index=False, float_format="%.17g")
    records = []
    for method, column in (("daregram", "daregram_b_t"),
                           ("five_point_recursive", "curve_only_q_t"),
                           ("fusion", "fusion_pred_vb")):
        for segment, start, end in SEGMENTS:
            part = frame.loc[frame.cut_index.between(start, end)]
            records.append({"method": method, "segment": segment,
                            **metrics(part.true_vb.to_numpy(float), part[column].to_numpy(float))})
    pd.DataFrame(records).to_csv(out / "comparison_metrics.csv", index=False, float_format="%.17g")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(frame.cut_index, frame.true_vb, color="black", label="C6 true VB")
    ax.plot(frame.cut_index, frame.daregram_b_t, lw=1.2, label="DARE-GRAM")
    ax.plot(frame.cut_index, frame.fusion_pred_vb, lw=1.2, label="five-point + model")
    ax.set(xlabel="C6 cut", ylabel="VB", title="C1 to C6: frozen DARE-GRAM and source-calibrated fusion")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prediction_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(frame.cut_index, frame.true_vb, color="black", label="C6 true VB")
    ax.plot(frame.cut_index, frame.curve_only_q_t, label="five-point recursive only")
    ax.set(xlabel="C6 cut", ylabel="VB", title="Unclipped five-point quadratic recursion")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curve_only_comparison.png", dpi=180)
    plt.close(fig)
    write_json(out / "score_audit.json", {
        "C6_suffix_first_read": "score stage after all 310 predictions and SHA256 lock",
        "C6_wear_sha256_after_prediction_complete": sha256(args.target_wear),
        "C6_suffix_labels_used_in_parameter_selection_or_recursion": False,
        "signed_error": "prediction minus true VB",
        "prior_new_confirmed_t0": "cut 210 OOR gate: uses unlabeled C6 STFT/OOR and C1 source labels; no extra C6 true VB, but exploratory and excluded from primary comparison"})
    print((out / "comparison_metrics.csv").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score"))
    parser.add_argument("--source-wear", type=Path,
                        default=Path(r"E:\QLP\source\source_mill\c1_wear.csv"))
    parser.add_argument("--target-wear", type=Path,
                        default=Path(r"E:\QLP\source\source_mill\c6_wear.csv"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
