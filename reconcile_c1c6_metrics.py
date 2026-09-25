"""Read-only reconciliation of C1->C6 suffix and full-lifecycle reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("artifacts")
OLD = ROOT / "five_seed_paired" / "c1_to_c6"
FULL = ROOT / "norm_comparison_20260925" / "zscore" / "c1_to_c6"
DIAGNOSTIC = ROOT / "five_seed_feature_distribution_20260925" / "paired_seed_metrics.csv"
OUTPUT = ROOT / "c1c6_metric_reconciliation_20260925"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_predictions(path: Path, cuts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [int(row["cut_index"]) for row in rows] != cuts:
        raise ValueError(f"Unexpected evaluation cuts: {path}")
    y = np.array([float(row["true_vb"]) for row in rows])
    p = np.array([float(row["pred_vb"]) for row in rows])
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError(f"Nonfinite prediction: {path}")
    return y, p


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    diagnostic = {(int(row["seed"]), row["source"], row["target"]): row
                  for row in csv.DictReader(DIAGNOSTIC.open(newline="", encoding="utf-8"))}
    rows = []
    for seed in range(42, 47):
        for method in ("source_only", "daregram"):
            old = OLD / f"seed_{seed}" / method
            full = FULL / f"seed_{seed}" / method
            old_config = json.loads((old / "config.json").read_text(encoding="utf-8"))
            full_config = json.loads((full / "config.json").read_text(encoding="utf-8"))
            if ((old_config["source"], old_config["target"], old_config["seed"], old_config["method"]) !=
                    ("c1", "c6", seed, method)):
                raise ValueError(f"Old config identity mismatch: {old}")
            if ((full_config["source"], full_config["target"], full_config["seed"], full_config["method"],
                 full_config["norm_method"]) != ("c1", "c6", seed, method, "zscore")):
                raise ValueError(f"Full config identity mismatch: {full}")
            if old_config["evaluation_cuts"] != list(range(95, 316)) or full_config["evaluation_cuts"] != list(range(1, 316)):
                raise ValueError(f"Unexpected config cut lists: seed {seed} {method}")
            old_hash, full_hash = sha(old / "final.pth"), sha(full / "final.pth")
            if old_hash != full_hash:
                raise ValueError(f"Different model checkpoints: seed {seed} {method}")
            if (old_config["source_normalization_mean"] != full_config["source_normalization_parameters"]["mean"] or
                old_config["source_normalization_std"] != full_config["source_normalization_parameters"]["std"]):
                raise ValueError(f"Different normalization: seed {seed} {method}")
            y_old, p_old = read_predictions(old / "predictions.csv", list(range(95, 316)))
            y_full, p_full = read_predictions(full / "predictions.csv", list(range(1, 316)))
            if not np.array_equal(y_old, y_full[94:]):
                raise ValueError(f"Different target labels on overlapping cuts: seed {seed} {method}")
            suffix_rmse = float(np.sqrt(np.mean((y_old - p_old) ** 2)))
            full_rmse = float(np.sqrt(np.mean((y_full - p_full) ** 2)))
            old_recorded = json.loads((old / "metrics.json").read_text(encoding="utf-8"))["RMSE"]
            full_recorded = json.loads((full / "metrics.json").read_text(encoding="utf-8"))["RMSE"]
            if not np.isclose(suffix_rmse, old_recorded, rtol=0, atol=1e-10) or not np.isclose(full_rmse, full_recorded, rtol=0, atol=1e-10):
                raise ValueError(f"RMSE does not match prediction file: seed {seed} {method}")
            diagnostic_rmse = float(diagnostic[seed, "c1", "c6"][f"{method}_rmse"])
            if not np.isclose(diagnostic_rmse, suffix_rmse, rtol=0, atol=1e-10):
                raise ValueError(f"Feature diagnostic used a different RMSE: seed {seed} {method}")
            rows.append({"seed": seed, "method": method,
                         "suffix_checkpoint": str(old / "final.pth"), "suffix_checkpoint_sha256": old_hash,
                         "full_checkpoint": str(full / "final.pth"), "full_checkpoint_sha256": full_hash,
                         "suffix_prediction_file": str(old / "predictions.csv"),
                         "full_prediction_file": str(full / "predictions.csv"),
                         "suffix_cut_first": 95, "suffix_cut_last": 315, "suffix_count": 221,
                         "full_cut_first": 1, "full_cut_last": 315, "full_count": 315,
                         "suffix_rmse_recomputed": suffix_rmse, "full_rmse_recomputed": full_rmse,
                         "diagnostic_rmse": diagnostic_rmse,
                         "overlap_prediction_max_abs_difference": float(np.max(np.abs(p_old - p_full[94:])))})
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for scope in ("suffix", "full"):
        s = np.array([r[f"{scope}_rmse_recomputed"] for r in rows if r["method"] == "source_only"])
        d = np.array([r[f"{scope}_rmse_recomputed"] for r in rows if r["method"] == "daregram"])
        summary[scope] = {"evaluation_cuts": "95..315" if scope == "suffix" else "1..315",
                          "source_only_rmse_mean": float(s.mean()), "source_only_rmse_sd": float(s.std(ddof=1)),
                          "daregram_rmse_mean": float(d.mean()), "daregram_rmse_sd": float(d.std(ddof=1)),
                          "delta_rmse_by_seed": (d - s).tolist(),
                          "delta_rmse_mean": float((d - s).mean()),
                          "delta_rmse_sd": float((d - s).std(ddof=1)),
                          "aggregation": "arithmetic mean of five seed RMSE values; sample SD ddof=1"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
