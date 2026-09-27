"""C6 five-point local quadratic, one-step, recursive VB forecast.

Only C6 cuts 1..5 are read during prediction. Scoring is a separate stage
that is enabled only after all 310 predictions are frozen and hash-locked.
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


ALL_CUTS = list(range(1, 316))
PREDICTION_CUTS = list(range(6, 316))
SEGMENTS = (("early_6_105", 6, 105), ("middle_106_210", 106, 210),
            ("late_211_315", 211, 315))
EXPLOSION_ABS_VB = 1_000.0


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score", "all"), default="all")
    parser.add_argument("--wear-csv", type=Path, default=Path(r"E:\QLP\source\source_mill\c6_wear.csv"))
    parser.add_argument("--out-root", type=Path,
                        default=Path("artifacts/c6_recursive_local_quadratic_20260926_final"))
    return parser.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_wear_row(row, expected_cut):
    assert set(row) == {"cut", "flute_1", "flute_2", "flute_3"}
    cut = int(row["cut"])
    if cut != expected_cut:
        raise ValueError(f"Expected cut {expected_cut}, found {cut}")
    vb = float(np.mean([float(row[f"flute_{i}"]) for i in (1, 2, 3)], dtype=np.float64))
    if not np.isfinite(vb):
        raise ValueError(f"Nonfinite true VB at cut {cut}")
    return vb


def read_initial_five(path):
    """Consume only the CSV header and five data rows; do not inspect suffix."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["cut", "flute_1", "flute_2", "flute_3"]:
            raise ValueError(f"Unexpected wear CSV schema: {reader.fieldnames}")
        result = {}
        for cut in range(1, 6):
            row = next(reader, None)
            if row is None:
                raise ValueError(f"Missing initial cut {cut}")
            result[cut] = {"vb": parse_wear_row(row, cut), "source": "true_initial"}
        return result


def read_all_wear_after_predictions(path):
    """Only call in score(), after a completed 310-prediction trace exists."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["cut", "flute_1", "flute_2", "flute_3"]:
            raise ValueError(f"Unexpected wear CSV schema: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 315:
        raise ValueError(f"Expected 315 wear rows, found {len(rows)}")
    return {cut: parse_wear_row(rows[cut - 1], cut) for cut in ALL_CUTS}


def fit_one_step(input_cuts, input_vb, prediction_cut):
    """OLS quadratic on actual cut indices, centered at the latest input cut."""
    xs = np.asarray(input_cuts, dtype=np.float64)
    ys = np.asarray(input_vb, dtype=np.float64)
    assert xs.shape == ys.shape == (5,)
    assert np.array_equal(xs, np.arange(prediction_cut - 5, prediction_cut))
    origin = float(xs[-1])
    u = xs - origin
    design = np.column_stack((u * u, u, np.ones(5, dtype=np.float64)))
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic, linear, constant = np.linalg.lstsq(design, ys, rcond=None)[0]
        step = float(prediction_cut - origin)
        prediction = float(quadratic * step * step + linear * step + constant)
    assert step == 1.0
    return float(quadratic), float(linear), float(constant), prediction


def plot_unscored(initial, trace, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x_initial = list(initial)
    y_initial = [initial[cut]["vb"] for cut in x_initial]
    x_pred = [row["prediction_cut"] for row in trace if np.isfinite(row["pred_vb"])]
    y_pred = [row["pred_vb"] for row in trace if np.isfinite(row["pred_vb"])]
    axes[0].plot(x_initial, y_initial, "ko-", markersize=3, label="initial true VB, cuts 1–5")
    axes[0].plot(x_pred, y_pred, color="tab:blue", marker=".", markersize=2,
                 label="recursive prediction")
    axes[0].set(ylabel="VB", title="C6 five-point quadratic recursive forecast (unscored)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(x_pred, np.abs(y_pred), color="tab:blue", marker=".", markersize=2)
    axes[1].axhline(EXPLOSION_ABS_VB, color="tab:red", linestyle="--", label="explosion threshold")
    axes[1].set(xlabel="C6 prediction cut", ylabel="|predicted VB| (log scale)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_early_detail(scored, path):
    early = scored.loc[scored.prediction_cut.between(6, 40)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(early.prediction_cut, early.true_vb, color="black", label="C6 true VB")
    ax.plot(early.prediction_cut, early.pred_vb, label="recursive prediction")
    ax.axhline(-EXPLOSION_ABS_VB, color="tab:red", linestyle="--",
               label="−1,000 diagnostic threshold")
    ax.set(xlabel="C6 prediction cut", ylabel="VB",
           title="Recursive forecast, cuts 6–40: onset of divergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def predict(args):
    out = args.out_root
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    # No full-file hash or dataframe read here: future C6 labels remain unread.
    initial = read_initial_five(args.wear_csv)
    assert len(initial) == 5 and list(initial) == list(range(1, 6))
    history = dict(initial)
    trace = []
    failure = None
    first_explosion = None
    first_negative = None
    for prediction_cut in PREDICTION_CUTS:
        input_cuts = list(range(prediction_cut - 5, prediction_cut))
        input_rows = [history[cut] for cut in input_cuts]
        input_vb = [row["vb"] for row in input_rows]
        input_sources = [row["source"] for row in input_rows]
        if prediction_cut == 6:
            assert input_cuts == [1, 2, 3, 4, 5]
            assert input_sources == ["true_initial"] * 5
        if prediction_cut == 7:
            assert input_sources == ["true_initial"] * 4 + ["predicted"]
        if prediction_cut >= 11:
            assert input_sources == ["predicted"] * 5
        quadratic, linear, constant, pred = fit_one_step(input_cuts, input_vb, prediction_cut)
        row = {"prediction_cut": prediction_cut,
               "input_cut_start": input_cuts[0], "input_cut_end": input_cuts[-1],
               "basis_origin_cut": input_cuts[-1]}
        for j, (cut, vb, source) in enumerate(zip(input_cuts, input_vb, input_sources), 1):
            row[f"input_cut_{j}"] = cut
            row[f"input_vb_{j}"] = vb
            row[f"input_source_{j}"] = source
        row.update({"coef_u2": quadratic, "coef_u1": linear, "coef_u0": constant,
                    "pred_vb": pred})
        if not np.isfinite(pred):
            failure = {"kind": "nonfinite_prediction", "first_failure_cut": prediction_cut,
                       "raw_prediction_repr": repr(pred)}
        elif abs(pred) > EXPLOSION_ABS_VB and first_explosion is None:
            first_explosion = {"first_explosion_cut": prediction_cut,
                               "raw_prediction_repr": repr(pred),
                               "explosion_abs_vb_threshold": EXPLOSION_ABS_VB}
        if np.isfinite(pred) and pred < 0 and first_negative is None:
            first_negative = {"first_negative_cut": prediction_cut,
                              "raw_prediction_repr": repr(pred)}
        row["status"] = "nonfinite_stop" if failure else (
            "finite_explosion" if abs(pred) > EXPLOSION_ABS_VB else "predicted")
        trace.append(row)
        if failure:
            break
        # A successful forecast immediately becomes history for the next fit.
        history[prediction_cut] = {"vb": pred, "source": "predicted"}
    out.mkdir(parents=True)
    pd.DataFrame({"cut_index": list(initial),
                  "true_vb_initial": [initial[cut]["vb"] for cut in initial]}).to_csv(
                      out / "initial_five_true_vb.csv", index=False, float_format="%.17g")
    trace_path = out / "recursive_trace_unscored.csv"
    pd.DataFrame(trace).to_csv(trace_path, index=False, float_format="%.17g")
    status = {"status": ("completed_with_explosion" if first_explosion else "completed")
              if failure is None else "stopped_nonfinite",
              "initial_label_cuts_read": [1, 2, 3, 4, 5],
              "future_label_cuts_read_before_prediction_complete": [],
              "prediction_cuts_generated": [row["prediction_cut"] for row in trace],
              "successful_prediction_count": len(history) - 5,
              "trace_sha256": file_hash(trace_path),
              "explosion_abs_vb_threshold": EXPLOSION_ABS_VB,
              "coefficient_basis": "VB(x) = coef_u2*(x-basis_origin_cut)^2 + coef_u1*(x-basis_origin_cut) + coef_u0",
              "fitting": "ordinary least squares, five latest history points, refit every prediction cut",
              "first_negative": first_negative, "first_explosion": first_explosion,
              "failure": failure}
    if failure is None:
        assert len(trace) == 310 and len(history) == 315
    else:
        assert trace[-1]["prediction_cut"] == failure["first_failure_cut"]
    write_json(out / "prediction_status.json", status)
    plot_unscored(initial, trace, out / "recursive_curve_unscored.png")
    print(json.dumps({"status": status["status"], "successful_prediction_count": status["successful_prediction_count"],
                      "first_explosion": first_explosion, "failure": failure}, ensure_ascii=False, indent=2))
    return failure is None


def score(args):
    out = args.out_root
    if (out / "predictions_scored.csv").exists() or (out / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite scored results in {out}")
    status = json.loads((out / "prediction_status.json").read_text(encoding="utf-8"))
    if status["status"] not in ("completed", "completed_with_explosion") or status["successful_prediction_count"] != 310:
        raise RuntimeError("Recursion did not complete 310 finite predictions; future C6 labels must remain unread")
    trace_path = out / "recursive_trace_unscored.csv"
    if file_hash(trace_path) != status["trace_sha256"]:
        raise ValueError("Unscored prediction trace changed after it was fixed")
    trace = pd.read_csv(trace_path)
    if trace.prediction_cut.tolist() != PREDICTION_CUTS or not np.isfinite(trace.pred_vb).all():
        raise ValueError("Expected 310 ordered, finite predictions")
    initial = pd.read_csv(out / "initial_five_true_vb.csv")
    assert initial.cut_index.tolist() == [1, 2, 3, 4, 5]
    # First suffix-label read occurs here, after all forecasts are complete.
    wear = read_all_wear_after_predictions(args.wear_csv)
    for row in initial.itertuples(index=False):
        assert np.isclose(row.true_vb_initial, wear[row.cut_index], atol=1e-12)
    trace["true_vb"] = [wear[cut] for cut in PREDICTION_CUTS]
    trace.to_csv(out / "predictions_scored.csv", index=False, float_format="%.17g")
    scored = pd.read_csv(out / "predictions_scored.csv")
    pred = scored.pred_vb.to_numpy(dtype=np.float64)
    truth = scored.true_vb.to_numpy(dtype=np.float64)
    signed = pred - truth
    metrics = {"cuts": "6..315", "n": 310,
               "R2": float(r2_score(truth, pred)),
               "MAE": float(mean_absolute_error(truth, pred)),
               "RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
               "mean_signed_error": float(signed.mean()),
               "segments": {name: {"cuts": f"{start}..{end}",
                                   "n": int(scored.prediction_cut.between(start, end).sum()),
                                   "mean_signed_error": float(signed[scored.prediction_cut.between(start, end)].mean())}
                            for name, start, end in SEGMENTS}}
    write_json(out / "metrics.json", metrics)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(scored.prediction_cut, truth, color="black", label="C6 true VB")
    ax.plot(scored.prediction_cut, pred, label="recursive five-point quadratic forecast")
    ax.set(xlabel="C6 prediction cut", ylabel="VB", title="C6 recursive local quadratic forecast")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prediction_vs_true.png", dpi=200)
    plt.close(fig)
    plot_early_detail(scored, out / "prediction_vs_true_early.png")
    write_json(out / "score_audit.json", {
        "target_suffix_label_first_read": "after all 310 predictions were fixed",
        "target_suffix_labels_used_in_recursion": False,
        "wear_csv_sha256_after_prediction_complete": file_hash(args.wear_csv),
        "signed_error_definition": "prediction minus true VB",
        "segment_boundaries": {name: [start, end] for name, start, end in SEGMENTS},
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    args = arguments()
    completed = True
    if args.stage in ("predict", "all"):
        completed = predict(args)
    if args.stage == "score" or (args.stage == "all" and completed):
        score(args)


if __name__ == "__main__":
    main()
