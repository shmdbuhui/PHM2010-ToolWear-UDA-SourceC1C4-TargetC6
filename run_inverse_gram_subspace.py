"""Six-direction project-defined inverse Gram subspace comparison.

This is not a reproduction of Kim et al.'s IGSM.  Training reads source wear
only; `prepare` freezes all unlabeled predictions before `evaluate` reads
target wear. Existing source-only, DARE-GRAM and OOR-PGA files are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import analyze_zscore_stage_errors as stage
import run_five_seed_pairs as historical
import run_full_1_315_baseline_protocol as baseline
import run_oor_pga_full_1_315 as oor
import run_single_source_pairs as base
from inverse_gram_subspace import ENERGY, EPS_RELATIVE, inverse_gram_subspace_loss


METHOD = "inverse_gram_subspace"
DEFAULT_ROOT = Path("artifacts/inverse_gram_subspace_20260926")
FROZEN = Path("artifacts/full_1_315_baseline_zscore_20260925")
OOR = Path("artifacts/oor_pga_full_1_315_zscore_20260925")
CACHE = Path("artifacts/five_seed_paired/feature_cache")
HISTORICAL = Path("artifacts/five_seed_paired")
RAW = Path(r"E:\QLP\source\source_mill")
CUTS = list(range(1, 316))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_path(root: Path, source: str, target: str, seed: int) -> Path:
    return root / f"{source}_to_{target}" / f"seed_{seed}"


def verify_frozen() -> None:
    oor.check_baseline_lock(FROZEN)
    lock = stage.json_read(OOR / "prediction_lock_before_target_labels.json")
    if stage.sha(FROZEN / "protocol_lock_full_1_315.json") != lock["baseline_lock_sha256"]:
        raise ValueError("Frozen baseline/OOR lock mismatch")
    for name, digest in lock["oor_score_sha256"].items():
        if stage.sha(OOR / "oor_scores" / name) != digest:
            raise ValueError(f"OOR score changed: {name}")
    if stage.sha(CACHE / "manifest.json") != lock["STFT_cache_manifest_sha256"]:
        raise ValueError("STFT cache changed")


def load_features(source: str, target: str):
    manifest = stage.json_read(CACHE / "manifest.json")
    arrays = {}
    for tool in (source, target):
        path = CACHE / f"{tool}_stft.npy"
        if stage.sha(path) != manifest["tools"][tool]["feature_sha256"]:
            raise ValueError(f"STFT hash mismatch: {tool}")
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if x.shape != (315, 6, 128, 128) or x.dtype != np.float32:
            raise ValueError(f"STFT shape/dtype mismatch: {tool}")
        arrays[tool] = x
    mean = arrays[source].mean(axis=(0, 2, 3)).astype(np.float32)
    std = (arrays[source].std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    xs = ((arrays[source] - mean[None, :, None, None]) /
          (std[None, :, None, None] + 1e-8)).astype(np.float32)
    xt = ((arrays[target] - mean[None, :, None, None]) /
          (std[None, :, None, None] + 1e-8)).astype(np.float32)
    if not np.isfinite(xs).all() or not np.isfinite(xt).all():
        raise ValueError("Nonfinite normalized STFT")
    return xs, xt, mean, std, manifest


def train_one(root: Path, source: str, target: str, seed: int, device: torch.device) -> None:
    folder = run_path(root, source, target, seed)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "final.pth"
    raw_path = folder / "raw_predictions.csv"
    if checkpoint.exists() and raw_path.exists():
        print(f"Reuse {source}->{target} seed {seed}", flush=True)
        return
    if checkpoint.exists() != raw_path.exists():
        raise RuntimeError(f"Partial checkpoint/prediction: {folder}")
    xs, xt, mean, std, manifest = load_features(source, target)
    y = base.wear_labels(RAW, source)  # source labels only
    old_cfg = stage.json_read(HISTORICAL / f"{source}_to_{target}" /
                              f"seed_{seed}" / "daregram" / "config.json")
    if (old_cfg["epochs"], old_cfg["batch_size"], old_cfg["lr"],
        old_cfg["checkpoint_selection"], old_cfg["actual_unlabeled_target_cuts"]) != (
            historical.EPOCHS, historical.BATCH_SIZE, historical.LR, "final epoch", CUTS):
        raise ValueError(f"Historical training protocol mismatch: {folder}")
    if (not np.array_equal(np.asarray(old_cfg["source_normalization_mean"], np.float32), mean) or
        not np.array_equal(np.asarray(old_cfg["source_normalization_std"], np.float32), std)):
        raise ValueError(f"Normalization mismatch: {folder}")
    model, init_hash = base.build_model(seed, device)
    if init_hash != old_cfg["initial_model_sha256"]:
        raise ValueError(f"Initialization mismatch: {folder}")
    src = DataLoader(TensorDataset(torch.from_numpy(xs), torch.from_numpy(y),
                                   torch.arange(1, 316)), batch_size=historical.BATCH_SIZE,
                     shuffle=True, drop_last=False,
                     generator=torch.Generator().manual_seed(seed))
    tgt = DataLoader(TensorDataset(torch.from_numpy(xt), torch.arange(1, 316)),
                     batch_size=historical.BATCH_SIZE, shuffle=True, drop_last=False,
                     generator=torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=historical.LR)
    batch_rows, epoch_rows, source_hashes, target_seen = [], [], [], set()
    preflight = None
    print(f"Train {source}->{target} seed {seed} on {device}", flush=True)
    for epoch in range(1, historical.EPOCHS + 1):
        model.train()
        target_iter = iter(tgt)
        source_order, epoch_target = [], set()
        weight = 2 / (1 + math.exp(-10 * (epoch - 1) / (historical.EPOCHS - 1))) - 1
        for batch_index, ((xb, yb, sc), (tb, tc)) in enumerate(zip(src, target_iter), 1):
            source_order.extend(sc.tolist())
            epoch_target.update(tc.tolist())
            context = f"{source}->{target} seed={seed} epoch={epoch} batch={batch_index}"
            detail = {}
            try:
                xb, yb, tb = xb.to(device), yb.to(device).unsqueeze(1), tb.to(device)
                optimizer.zero_grad()
                hs = model.feature_extractor(xb)
                ht = model.feature_extractor(tb)
                pred = model.regressor(hs)
                mse = F.mse_loss(pred, yb)
                align, detail = inverse_gram_subspace_loss(hs, ht)
                weighted = weight * align
                total = mse + weighted
                if not torch.isfinite(torch.stack((mse, align, weighted, total))).all():
                    raise FloatingPointError("Nonfinite loss")
                alignment_feature_grad_norm = None
                if preflight is None:
                    alignment_feature_grad = torch.autograd.grad(align, hs, retain_graph=True)[0]
                    if not torch.isfinite(alignment_feature_grad).all():
                        raise FloatingPointError("Nonfinite alignment feature gradient")
                    alignment_feature_grad_norm = float(torch.linalg.vector_norm(alignment_feature_grad))
                total.backward()
                gradient_sq = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Nonfinite parameter gradient")
                        gradient_sq += float(parameter.grad.double().square().sum())
                grad_norm = math.sqrt(gradient_sq)
                if not math.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                before = model.feature_extractor.backbone.conv1.weight.detach().clone() if preflight is None else None
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite parameter after optimizer step")
                if preflight is None:
                    update_norm = float(torch.linalg.vector_norm(
                        model.feature_extractor.backbone.conv1.weight.detach() - before))
                    if not math.isfinite(update_norm) or update_norm <= 0:
                        raise FloatingPointError("First-batch parameter update missing/nonfinite")
                    preflight = {"context": context, "source_mse": float(mse.detach()),
                                 "alignment_loss": float(align.detach()), "grad_norm": grad_norm,
                                 "alignment_feature_grad_norm": alignment_feature_grad_norm,
                                 "conv1_update_norm": update_norm, **detail}
                    write_json(folder / "real_batch_preflight.json", preflight)
                batch_rows.append({"epoch": epoch, "batch": batch_index,
                                   "source_mse": float(mse.detach()),
                                   "alignment_loss": float(align.detach()),
                                   "weighted_alignment_loss": float(weighted.detach()),
                                   "alignment_weight": weight, "total_loss": float(total.detach()),
                                   "gradient_norm": grad_norm, **detail})
            except Exception as exc:
                spectra = {}
                if "hs" in locals() and "ht" in locals():
                    with torch.no_grad():
                        for name, features in (("source", hs), ("target", ht)):
                            try:
                                z = torch.cat((torch.ones_like(features[:, :1]), features), dim=1).double()
                                spectra[name] = torch.linalg.svdvals(z).cpu().tolist()
                            except Exception as spectrum_exc:
                                spectra[name] = repr(spectrum_exc)
                failure = {"context": context, "error": repr(exc), "singular_values": spectra,
                           "singular_value_summary": detail,
                           "traceback": traceback.format_exc()}
                write_json(folder / "failure.json", failure)
                raise RuntimeError(f"Numerical failure at {context}; {detail}; {exc}") from exc
        if source_order != [] and sorted(source_order) == CUTS and epoch_target == set(CUTS):
            target_seen.update(epoch_target)
        else:
            raise ValueError(f"Incomplete source/target cut coverage epoch {epoch}: {folder}")
        digest = hashlib.sha256(np.asarray(source_order, dtype=np.int32).tobytes()).hexdigest()
        source_hashes.append(digest)
        if digest != old_cfg["source_order_sha256_by_epoch"][epoch - 1]:
            raise ValueError(f"Source batch order mismatch at epoch {epoch}: {folder}")
        frame = pd.DataFrame(batch_rows[-len(src):])
        row = {"epoch": epoch, "source_order_sha256": digest, "target_unique": len(epoch_target)}
        for key in ("source_mse", "alignment_loss", "weighted_alignment_loss", "alignment_weight",
                    "total_loss", "gradient_norm", "k", "rank_source", "rank_target",
                    "sigma_source_max", "sigma_source_min", "sigma_source_kept_min",
                    "sigma_target_max", "sigma_target_min", "sigma_target_kept_min",
                    "epsilon_source", "epsilon_target"):
            row[key + "_mean"] = float(frame[key].mean())
            row[key + "_min"] = float(frame[key].min())
            row[key + "_max"] = float(frame[key].max())
        epoch_rows.append(row)
        pd.DataFrame(batch_rows).to_csv(folder / "batch_diagnostics.csv", index=False)
        pd.DataFrame(epoch_rows).to_csv(folder / "epoch_diagnostics.csv", index=False)
        print(f"{source}->{target} seed {seed} epoch {epoch}/{historical.EPOCHS}: "
              f"MSE={row['source_mse_mean']:.4f} align={row['alignment_loss_mean']:.5f} "
              f"k={row['k_mean']:.1f} grad={row['gradient_norm_mean']:.3g}", flush=True)
    torch.save({"model": model.state_dict(), "epoch": historical.EPOCHS}, checkpoint)
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, 315, historical.BATCH_SIZE):
            xb = torch.from_numpy(xt[offset:offset + historical.BATCH_SIZE]).to(device)
            predictions.extend(model.regressor(model.feature_extractor(xb)).cpu().numpy().ravel().tolist())
    if len(predictions) != 315 or not np.isfinite(predictions).all():
        raise FloatingPointError(f"Invalid raw predictions: {folder}")
    pd.DataFrame({"cut_index": CUTS, "pred_vb": predictions}).to_csv(
        raw_path, index=False, float_format="%.17g")
    write_json(folder / "config.json", {
        "method": METHOD, "definition": "project-defined comparison, not Kim et al. IGSM",
        "source": source, "target": target, "seed": seed,
        "source_cuts": CUTS, "unlabeled_target_cuts": sorted(target_seen),
        "evaluation_cuts": CUTS, "target_label_reads_during_training": 0,
        "input": "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128",
        "backbone": "ResNet18", "regressor": "Linear(512, 1)", "feature_layer": "512-d post-pool",
        "source_loss": "MSE", "optimizer": "Adam", "lr": historical.LR,
        "epochs": historical.EPOCHS, "batch_size": historical.BATCH_SIZE,
        "checkpoint_selection": "final epoch", "energy": ENERGY,
        "epsilon_relative": EPS_RELATIVE,
        "epsilon_formula": "max(sigma_max^2,1)*1e-8 per domain, detached",
        "alignment_weight": "2/(1+exp(-10*(epoch-1)/(50-1)))-1",
        "source_normalization_mean": mean.tolist(), "source_normalization_std": std.tolist(),
        "stft_code_sha256": manifest["stft_code_sha256"],
        "source_feature_sha256": manifest["tools"][source]["feature_sha256"],
        "target_feature_sha256": manifest["tools"][target]["feature_sha256"],
        "initial_model_sha256": init_hash,
        "source_order_sha256_by_epoch": source_hashes,
        "checkpoint_sha256": stage.sha(checkpoint), "raw_prediction_sha256": stage.sha(raw_path)})


def selected(args):
    for source, target in baseline.PAIRS:
        if args.direction and args.direction != f"{source}_to_{target}":
            continue
        for seed in baseline.SEEDS:
            if args.seed is None or seed == args.seed:
                yield source, target, seed


def prepare(root: Path) -> None:
    verify_frozen()
    expected = [(s, t, seed) for s, t in baseline.PAIRS for seed in baseline.SEEDS]
    for source, target, seed in expected:
        folder = run_path(root, source, target, seed)
        cfg = stage.json_read(folder / "config.json")
        if (cfg["checkpoint_sha256"] != stage.sha(folder / "final.pth") or
            cfg["raw_prediction_sha256"] != stage.sha(folder / "raw_predictions.csv") or
            cfg["evaluation_cuts"] != CUTS):
            raise ValueError(f"Incomplete/changed run: {folder}")
    hashes = {}
    for source, target, seed in expected:
        folder = run_path(root, source, target, seed)
        raw = pd.read_csv(folder / "raw_predictions.csv")
        if raw.cut_index.tolist() != CUTS or not np.isfinite(raw.pred_vb).all():
            raise ValueError(f"Invalid raw prediction: {folder}")
        score = pd.read_csv(OOR / "oor_scores" /
                            f"{source}_to_{target}_oor_score_full_1_315.csv")
        if score.cut_index.tolist() != CUTS:
            raise ValueError(f"OOR cut mismatch: {folder}")
        # The fixed factor was already produced by unchanged run_oor_pga_full_1_315.py.
        # Read its saved output instead of refitting or retuning on IGSM predictions.
        factor = score.multiplicative_factor.to_numpy(float)
        historical_oor = pd.read_csv(OOR / "unlabeled_predictions" /
                                     f"{source}_to_{target}_seed_{seed}_full_1_315.csv",
                                     usecols=["cut_index", "daregram_pred_vb", "daregram_oor_pga_pred_vb"])
        if (historical_oor.cut_index.tolist() != CUTS or
            not np.allclose(historical_oor.daregram_pred_vb.to_numpy(float) * factor,
                            historical_oor.daregram_oor_pga_pred_vb.to_numpy(float),
                            rtol=1e-12, atol=1e-12)):
            raise ValueError(f"Frozen OOR factor does not reproduce existing correction: {folder}")
        corrected = raw.pred_vb.to_numpy(float) * factor
        if not np.isfinite(corrected).all():
            raise FloatingPointError(f"Nonfinite OOR-PGA predictions: {folder}")
        table = pd.DataFrame({"cut_index": CUTS, "raw_pred_vb": raw.pred_vb,
                              "oor_rate_48": score.oor_rate_48, "oor_flag_count_48": score.oor_flag_count_48,
                              "gate": score.gate, "trigger_cut": score.trigger_cut,
                              "source_pga_m": score.source_pga_m,
                              "multiplicative_factor": factor,
                              "correction_delta": corrected - raw.pred_vb.to_numpy(float),
                              "oor_pga_pred_vb": corrected})
        path = folder / "oor_pga_predictions.csv"
        table.to_csv(path, index=False, float_format="%.17g")
        hashes[f"{source}_to_{target}_seed_{seed}"] = {
            "raw_sha256": stage.sha(folder / "raw_predictions.csv"),
            "corrected_sha256": stage.sha(path)}
    write_json(root / "prediction_lock_before_target_labels.json", {
        "method": METHOD, "scope": "full_1_315", "target_label_reads": 0,
        "frozen_baseline_lock_sha256": stage.sha(FROZEN / "protocol_lock_full_1_315.json"),
        "frozen_oor_lock_sha256": stage.sha(OOR / "prediction_lock_before_target_labels.json"),
        "predictions": hashes})
    print("Prepared and locked 30 raw + 30 OOR-PGA predictions before target label reads", flush=True)


def measure(y, p):
    error = p - y
    return {"R2": float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)),
            "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error**2)))}


def evaluate(root: Path) -> None:
    verify_frozen()
    lock = stage.json_read(root / "prediction_lock_before_target_labels.json")
    if (lock["frozen_baseline_lock_sha256"] != stage.sha(FROZEN / "protocol_lock_full_1_315.json") or
        lock["frozen_oor_lock_sha256"] != stage.sha(OOR / "prediction_lock_before_target_labels.json") or
        len(lock["predictions"]) != 30):
        raise ValueError("Prediction lock incomplete or changed")
    for source, target in baseline.PAIRS:
        for seed in baseline.SEEDS:
            folder = run_path(root, source, target, seed)
            hashes = lock["predictions"][f"{source}_to_{target}_seed_{seed}"]
            if (stage.sha(folder / "raw_predictions.csv") != hashes["raw_sha256"] or
                stage.sha(folder / "oor_pga_predictions.csv") != hashes["corrected_sha256"]):
                raise ValueError(f"Predictions changed since label lock: {folder}")
    truth = {tool: stage.labels(RAW, tool)[0].astype(float) for tool in base.TOOLS}
    old_raw = pd.read_csv(FROZEN / "per_seed_metrics_full_1_315.csv")
    old_oor = pd.read_csv(OOR / "per_seed_metrics_full_1_315.csv")
    rows, deltas = [], []
    for source, target in baseline.PAIRS:
        for seed in baseline.SEEDS:
            folder = run_path(root, source, target, seed)
            y = truth[target]
            table = pd.read_csv(folder / "oor_pga_predictions.csv")
            if table.cut_index.tolist() != CUTS:
                raise ValueError(f"Cut ordering mismatch: {folder}")
            table.insert(1, "true_vb", y)
            table.to_csv(folder / "evaluated_per_cut.csv", index=False, float_format="%.17g")
            for method, old, values in (
                (METHOD, None, table.raw_pred_vb.to_numpy(float)),
                (METHOD + "_oor_pga", None, table.oor_pga_pred_vb.to_numpy(float))):
                rows.append({"source": source, "target": target, "seed": seed,
                             "method": method, "scope": "full_1_315", **measure(y, values)})
            for method in ("source_only", "daregram"):
                for suffix, source_table in (("", old_raw), ("_oor_pga", old_oor)):
                    old = source_table[(source_table.source == source) &
                                       (source_table.target == target) &
                                       (source_table.seed == seed) &
                                       (source_table.method == method + suffix) &
                                       ((source_table.segment == "full") if "segment" in source_table else True)]
                    if len(old) != 1 or int(old.iloc[0].n) != 315:
                        raise ValueError(f"Baseline protocol mismatch: {source}->{target} {seed} {method+suffix}")
                    rows.append({"source": source, "target": target, "seed": seed,
                                 "method": method + suffix, "scope": "full_1_315",
                                 **{m: float(old.iloc[0][m]) for m in ("R2", "MAE", "RMSE")}})
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(CUTS, y, color="black", label="True VB")
            ax.plot(CUTS, table.raw_pred_vb, label=METHOD + " raw")
            ax.plot(CUTS, table.oor_pga_pred_vb, label=METHOD + " + OOR-PGA")
            ax.set(xlabel="Target cut", ylabel="VB", title=f"{source.upper()}→{target.upper()} seed {seed}")
            ax.legend()
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(folder / "ordered_curve.png", dpi=160)
            plt.close(fig)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(root / "six_direction_per_seed_metrics.csv", index=False)
    summary = per_seed.groupby(["source", "target", "method"], sort=True).agg(
        n_seeds=("seed", "count"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std")).reset_index()
    summary.to_csv(root / "six_direction_five_seed_summary.csv", index=False)
    for (source, target, seed), group in per_seed.groupby(["source", "target", "seed"]):
        indexed = group.set_index("method")
        for stage_name, suffix in (("raw", ""), ("oor_pga", "_oor_pga")):
            for reference in ("source_only", "daregram"):
                a = indexed.loc[METHOD + suffix]
                b = indexed.loc[reference + suffix]
                deltas.append({"source": source, "target": target, "seed": seed,
                               "stage": stage_name, "reference": reference,
                               **{f"delta_{metric}": float(a[metric] - b[metric])
                                  for metric in ("R2", "MAE", "RMSE")}})
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(root / "paired_per_seed_deltas.csv", index=False)
    delta_frame.groupby(["source", "target", "stage", "reference"], sort=True).agg(
        **{f"delta_{m}_{s}": (f"delta_{m}", op)
           for m in ("R2", "MAE", "RMSE") for s, op in (("mean", "mean"), ("sd", "std"))}
    ).reset_index().to_csv(root / "paired_five_seed_deltas.csv", index=False)
    write_json(root / "evaluation_manifest.json", {
        "prediction_lock_sha256": stage.sha(root / "prediction_lock_before_target_labels.json"),
        "target_labels_first_read": "evaluate, after all 30 raw and corrected prediction pairs were locked",
        "baseline_retrained": False, "scope": "full_1_315",
        "per_seed_metrics_sha256": stage.sha(root / "six_direction_per_seed_metrics.csv")})
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "prepare", "evaluate", "all"), default="all")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--direction", choices=[f"{s}_to_{t}" for s, t in baseline.PAIRS])
    parser.add_argument("--seed", type=int, choices=list(baseline.SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.stage in ("prepare", "evaluate") and (args.direction or args.seed is not None):
        parser.error("prepare/evaluate require all 30 runs")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("train", "all"):
        verify_frozen()
        for source, target, seed in selected(args):
            train_one(args.out_root, source, target, seed, torch.device(args.device))
    if args.stage in ("prepare", "all") and not (args.direction or args.seed is not None):
        prepare(args.out_root)
    if args.stage in ("evaluate", "all") and not (args.direction or args.seed is not None):
        evaluate(args.out_root)


if __name__ == "__main__":
    main()
