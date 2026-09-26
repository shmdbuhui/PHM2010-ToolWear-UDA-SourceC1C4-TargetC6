"""In-sample source-domain diagnostic for frozen A and trained M checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_multi_window_ablation as multi
import run_single_source_pairs as base


def predict_a(checkpoint: Path, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
              seed: int, device: torch.device) -> np.ndarray:
    model, _ = base.build_model(seed, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["epoch"] != 50:
        raise ValueError(f"A checkpoint is not final epoch: {checkpoint}")
    model.load_state_dict(state["model"])
    model.eval()
    pred = np.empty(315, dtype=np.float64)
    with torch.no_grad():
        for offset in range(0, 315, multi.BATCH_SIZE):
            block = np.asarray(x[offset:offset + multi.BATCH_SIZE])
            block = (block - mean[None, :, None, None]) / std[None, :, None, None]
            xb = torch.from_numpy(np.asarray(block, dtype=np.float32)).to(device)
            values = model.regressor(model.feature_extractor(xb)).cpu().numpy().reshape(-1)
            pred[offset:offset + len(values)] = values
    return pred


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"MAE": float(np.mean(np.abs(pred - y))),
            "RMSE": float(np.sqrt(np.mean(np.square(pred - y)))),
            "R2": float(1 - np.sum(np.square(pred - y)) / np.sum(np.square(y - y.mean()))),
            "signed_error": float(np.mean(pred - y))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-root", type=Path, default=Path("artifacts/full_1_315_baseline_zscore_20260925"))
    parser.add_argument("--m-root", type=Path, default=multi.DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = args.m_root / "source_domain_audit"
    out.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.a_root / "checkpoint_audit_full_1_315.csv")
    rows = []
    device = torch.device(args.device)
    for source, target in multi.PAIRS:
        source_y = base.wear_labels(args.raw_root, source).astype(np.float64)
        x_m = multi.cache_view(args.m_root, source)
        x_a = np.load(Path("artifacts/five_seed_paired/feature_cache") / f"{source}_stft.npy",
                      mmap_mode="r", allow_pickle=False)
        for seed in multi.SEEDS:
            m_folder = args.m_root / f"{source}_to_{target}" / f"seed_{seed}"
            config = json.loads((m_folder / "config.json").read_text(encoding="utf-8"))
            mean_m = np.asarray(config["source_zscore_mean"], dtype=np.float32)
            std_m = np.asarray(config["source_zscore_std"], dtype=np.float32)
            for method in multi.METHODS:
                a_row = audit[(audit.source == source) & (audit.target == target) &
                              (audit.seed == seed) & (audit.method == method)]
                if len(a_row) != 1:
                    raise ValueError(f"Missing A provenance: {source} {target} {seed} {method}")
                a_row = a_row.iloc[0]
                if base.file_hash(Path(a_row.checkpoint)) != a_row.checkpoint_sha256:
                    raise ValueError("Frozen A checkpoint hash mismatch")
                mean_a = np.asarray(json.loads(a_row.source_zscore_mean), dtype=np.float32)
                std_a = np.asarray(json.loads(a_row.source_zscore_std), dtype=np.float32)
                a_pred = predict_a(Path(a_row.checkpoint), x_a, mean_a, std_a, seed, device)
                m_pred = multi.predict_windows(method, x_m, mean_m, std_m, seed, m_folder, device)
                preds = {"A": a_pred, "M_center": m_pred[:, 1], "M_mean": m_pred.mean(axis=1)}
                for variant, pred in preds.items():
                    rows.append({"source": source, "target": target, "seed": seed,
                                 "method": method, "variant": variant, **metrics(source_y, pred)})
                pd.DataFrame({"cut_index": base.ALL_CUTS, "true_vb": source_y,
                              "A": a_pred, "M_center": m_pred[:, 1],
                              "M_mean": m_pred.mean(axis=1)}).to_csv(
                    out / f"{source}_to_{target}_seed_{seed}_{method}.csv", index=False,
                    float_format="%.17g")
                print(f"Source domain {source}->{target} seed={seed} {method}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_seed_metrics.csv", index=False)
    grouped = frame.groupby(["source", "target", "method", "variant"], sort=True)
    summary = grouped.agg(n_seeds=("seed", "size"),
                          **{f"{metric}_{stat}": (metric, "mean" if stat == "mean" else "std")
                             for metric in ("R2", "MAE", "RMSE", "signed_error")
                             for stat in ("mean", "sd")}).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(f"Saved source-domain in-sample audit: {out}")


if __name__ == "__main__":
    main()
