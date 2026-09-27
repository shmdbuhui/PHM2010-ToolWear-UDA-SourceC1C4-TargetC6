"""Compare frozen feature distribution distances with full-cut paired errors."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    args = parser.parse_args()
    root = args.audit_root
    index = pd.read_csv(root / "feature_index.csv")
    scores = pd.read_csv(root / "seed_summary.csv")
    columns = [f"feature_{i:03d}" for i in range(512)]
    results = []
    for record in index.to_dict("records"):
        s = pd.read_csv(record["resnet18_512_source"], usecols=columns).to_numpy(dtype=np.float64)
        t = pd.read_csv(record["resnet18_512_target"], usecols=columns).to_numpy(dtype=np.float64)
        if s.shape != (315, 512) or t.shape != (315, 512):
            raise ValueError("Feature shape mismatch")
        sigma = np.maximum(s.std(axis=0), 1e-8)
        zs = (s - s.mean(axis=0)) / sigma
        zt = (t - s.mean(axis=0)) / sigma
        mean_distance = np.linalg.norm(zs.mean(axis=0) - zt.mean(axis=0)) / np.sqrt(512)
        cov_distance = np.linalg.norm(np.cov(zs, rowvar=False) - np.cov(zt, rowvar=False), "fro") / 512
        results.append({k: record[k] for k in ("source", "target", "seed", "method")}
                       | {"feature_mean_distance": mean_distance, "feature_covariance_distance": cov_distance,
                          "checkpoint_sha256": record["checkpoint_sha256"]})
    detail = pd.DataFrame(results)
    detail.to_csv(root / "feature_alignment.csv", index=False)
    pairs = []
    for (source, target, seed), block in detail.groupby(["source", "target", "seed"]):
        if set(block.method) != {"source_only", "daregram"}:
            raise ValueError("Incomplete pair")
        score = scores[(scores.source == source) & (scores.target == target) & (scores.seed == seed)].iloc[0]
        a, b = [block[block.method == m].iloc[0] for m in ("source_only", "daregram")]
        pairs.append({"source": source, "target": target, "seed": seed,
                      "delta_mean_distance": b.feature_mean_distance - a.feature_mean_distance,
                      "delta_covariance_distance": b.feature_covariance_distance - a.feature_covariance_distance,
                      "delta_full_MAE": score.delta_MAE, "delta_full_RMSE": score.delta_RMSE,
                      "closer_mean_worse_MAE": bool(b.feature_mean_distance < a.feature_mean_distance and score.delta_MAE > 0),
                      "closer_covariance_worse_MAE": bool(b.feature_covariance_distance < a.feature_covariance_distance and score.delta_MAE > 0)})
    pd.DataFrame(pairs).to_csv(root / "paired_alignment_vs_error.csv", index=False)
    print(f"Saved 60 feature distance rows and {len(pairs)} paired full-cut comparisons")


if __name__ == "__main__":
    main()
