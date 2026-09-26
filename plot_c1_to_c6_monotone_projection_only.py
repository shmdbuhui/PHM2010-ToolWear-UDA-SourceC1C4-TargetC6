"""Redraw C1->C6 five-seed PAVA curves from saved predictions; never train."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "artifacts/monotone_pava_delta0_full_1_315_20260926"
PAIR_DIR = ROOT / "c1_to_c6"
OUTPUT = ROOT / "c1_to_c6_monotone_projection_only.png"
CUTS = np.arange(1, 316, dtype=np.int64)
SEEDS = range(42, 47)
METHODS = {"source_only": "source-only + PAVA", "daregram": "DARE-GRAM + PAVA"}


def read_projection(path):
    # usecols prevents raw_pred from being loaded or passed to plotting code.
    frame = pd.read_csv(path, usecols=["cut_index", "true_vb", "monotone_pred"])
    if (len(frame) != len(CUTS) or frame.cut_index.isna().any() or
        frame.cut_index.duplicated().any() or
        not np.array_equal(np.sort(frame.cut_index.to_numpy()), CUTS)):
        raise ValueError(f"Missing, repeated, or unexpected cut in {path}")
    frame = frame.sort_values("cut_index")
    values = frame[["true_vb", "monotone_pred"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite VB value in {path}")
    return values[:, 0], values[:, 1]


def check_final_curve(label, cuts, values):
    if (len(cuts) != len(CUTS) or len(np.unique(cuts)) != len(CUTS) or
        not np.array_equal(cuts, CUTS) or values.shape != CUTS.shape or
        not np.isfinite(values).all()):
        raise ValueError(f"Final cut or prediction array invalid: {label}")
    steps = np.diff(values)
    declines = np.flatnonzero(steps < 0)
    print(f"{label}: min adjacent difference = {steps.min():.12g} VB; decline count = {len(declines)}")
    if len(declines):
        locations = [(int(cuts[i]), int(cuts[i + 1]), float(steps[i])) for i in declines]
        raise ValueError(f"Declining final curve {label}; (cut_i, cut_i+1, step) = {locations}")


def main():
    curves = {}
    truth = None
    used = []
    for method, label in METHODS.items():
        predictions = []
        for seed in SEEDS:
            path = PAIR_DIR / f"seed_{seed}" / method / "predictions.csv"
            true_vb, monotone_pred = read_projection(path)
            if truth is None:
                truth = true_vb
            elif not np.allclose(truth, true_vb, rtol=0, atol=1e-10):
                raise ValueError(f"True VB disagrees across saved CSVs: {path}")
            predictions.append(monotone_pred)
            used.append(path)
        curves[label] = np.mean(np.stack(predictions), axis=0)

    # Validate the exact arrays passed to ax.plot before creating or saving a figure.
    for label, values in curves.items():
        check_final_curve(label, CUTS, values)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(CUTS, truth, color="#1b1b1b", linewidth=2, label="True VB")
    ax.plot(CUTS, curves["source-only + PAVA"], color="#2563a6", linewidth=1.8,
            label="source-only + PAVA")
    ax.plot(CUTS, curves["DARE-GRAM + PAVA"], color="#b12d58", linewidth=1.8,
            label="DARE-GRAM + PAVA")
    ax.set(xlim=(1, 315), xlabel="Target cut", ylabel="VB",
           title="C1→C6: five-seed mean of monotone-projected predictions")
    ax.set_xticks([1, 50, 100, 150, 200, 250, 315])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
    print("Loaded columns: cut_index, true_vb, monotone_pred (raw_pred was not loaded)")
    for path in used:
        print(f"  {path}")


if __name__ == "__main__":
    main()
