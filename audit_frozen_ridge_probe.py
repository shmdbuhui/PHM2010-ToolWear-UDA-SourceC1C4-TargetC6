"""Frozen 512-D Ridge probe: prepare all predictions before scoring target wear."""
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
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_single_source_pairs as base

TOOLS = ("c1", "c4", "c6")
SEEDS = range(42, 47)
METHODS = ("source_only", "daregram")
CUTS = list(range(1, 316))
COLS = [f"feature_{i:03d}" for i in range(512)]
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BLOCKS = [np.arange(i * 63, (i + 1) * 63) for i in range(5)]
SEGMENTS = {"full": (1, 315), "early": (1, 105), "middle": (106, 210), "late": (211, 315)}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def measure(y, p):
    error = p - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": float(np.mean(abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "R2": float(1 - np.sum(error ** 2) / ss) if ss > 1e-8 else np.nan,
            "signed_error": float(error.mean()), "underestimate_fraction": float(np.mean(error < 0)),
            "true_vb_std": float(np.std(y))}


def checked_features(path, tool):
    frame = pd.read_csv(path)
    if (frame.columns.tolist() != ["tool_id", "cut_index", *COLS] or len(frame) != 315
            or frame.tool_id.tolist() != [tool] * 315 or frame.cut_index.tolist() != CUTS):
        raise ValueError(f"Feature/cut mapping invalid: {path}")
    x = frame[COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite feature: {path}")
    return x


def fit_probe(x, y, indices, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="auto"))
    model.fit(x[indices], y[indices])
    return model


def select_alpha(x, y, outer_train_blocks):
    # Inner folds are fixed contiguous original-cut blocks; source labels only.
    rows = []
    for alpha in ALPHAS:
        rmse = []
        for val_block in outer_train_blocks:
            train_indices = np.concatenate([BLOCKS[j] for j in outer_train_blocks if j != val_block])
            val_indices = BLOCKS[val_block]
            model = fit_probe(x, y, train_indices, alpha)
            p = model.predict(x[val_indices])
            rmse.append(measure(y[val_indices], p)["RMSE"])
        rows.append({"alpha": alpha, "inner_rmse_mean": float(np.mean(rmse)),
                     "inner_rmse_sd": float(np.std(rmse, ddof=1)),
                     "inner_rmse_by_block": json.dumps(rmse)})
    chosen = min(rows, key=lambda r: (r["inner_rmse_mean"], r["alpha"]))["alpha"]
    return chosen, rows


def verify_inputs(prior_root, mode_root):
    provenance = pd.read_csv(prior_root / "provenance.csv")
    feature_index = pd.read_csv(prior_root / "feature_index.csv")
    modes = pd.read_csv(mode_root / "mode_metrics.csv")
    if len(provenance) != 60 or len(feature_index) != 60 or len(modes) != 60:
        raise ValueError("Expected 60 completed checkpoint records")
    records = provenance.merge(feature_index, on=["source", "target", "seed", "method", "checkpoint_sha256"], validate="one_to_one")
    records = records.merge(modes[["source", "target", "seed", "method", "checkpoint_sha256"]],
                            on=["source", "target", "seed", "method", "checkpoint_sha256"], validate="one_to_one")
    if len(records) != 60:
        raise ValueError("Provenance/mode audit mismatch")
    for row in records.itertuples():
        if sha(Path(row.checkpoint)) != row.checkpoint_sha256:
            raise ValueError(f"Checkpoint changed: {row.checkpoint}")
        fa = json.loads(Path(row.feature_audit).read_text(encoding="utf-8"))
        for role, tool in (("source", row.source), ("target", row.target)):
            path = Path(getattr(row, f"{role}_features"))
            if sha(path) != fa["feature_stats"][role]["file_sha256"]:
                raise ValueError(f"Feature hash mismatch: {path}")
            checked_features(path, tool)
        reg = pd.read_csv(row.regressor_input_512)
        if (len(reg) != 630 or reg.domain.tolist() != ["source"] * 315 + ["target"] * 315
                or reg.cut_index.tolist() != CUTS + CUTS or reg.window_index.tolist() != [0] * 630):
            raise ValueError(f"Regressor input mapping mismatch: {row.regressor_input_512}")
        source = checked_features(Path(row.source_features), row.source)
        target = checked_features(Path(row.target_features), row.target)
        mode_eval = mode_root / "mode_features" / f"{row.source}_to_{row.target}_seed_{row.seed}_{row.method}_eval.csv"
        if not np.array_equal(source, checked_features(mode_eval, row.source)):
            raise ValueError(f"Prior source features differ from fresh eval forward: {mode_eval}")
        if not np.array_equal(reg[COLS].to_numpy(dtype=np.float64), np.vstack([source, target])):
            raise ValueError(f"ResNet/Linear interface mismatch: {row.regressor_input_512}")
    return records


def prepare(args):
    if args.out_root.exists():
        raise FileExistsError("Fresh output directory required")
    records = verify_inputs(args.prior_root, args.mode_root)
    # Labels are used only for the source role of each independent direction.
    # A tool can be a source in one direction and a target in another.
    out = args.out_root
    out.mkdir(parents=True)
    (out / "probe_models").mkdir()
    fold_rows, alpha_rows, source_rows, target_rows, model_rows, source_scores = [], [], [], [], [], []
    for row in records.itertuples():
        identity = {k: getattr(row, k) for k in ("source", "target", "seed", "method")}
        xs = checked_features(Path(row.source_features), row.source)
        xt = checked_features(Path(row.target_features), row.target)
        y = base.wear_labels(args.raw_root, row.source).astype(np.float64)
        ckpt = torch.load(row.checkpoint, map_location="cpu", weights_only=True)["model"]
        weight = ckpt["regressor.0.weight"].numpy().ravel().astype(np.float64)
        bias = float(ckpt["regressor.0.bias"].item())
        original_source = xs @ weight + bias
        original_target = xt @ weight + bias
        fold_predictions = np.full(315, np.nan)
        for outer in range(5):
            train_blocks = [i for i in range(5) if i != outer]
            chosen, candidates = select_alpha(xs, y, train_blocks)
            for candidate in candidates:
                alpha_rows.append(dict(identity, scope="outer", outer_fold=outer + 1,
                                       outer_validation_cuts=f"{outer*63+1}-{(outer+1)*63}",
                                       inner_blocks=",".join(str(i + 1) for i in train_blocks),
                                       chosen_alpha=chosen, **candidate))
            train_indices = np.concatenate([BLOCKS[i] for i in train_blocks])
            val_indices = BLOCKS[outer]
            model = fit_probe(xs, y, train_indices, chosen)
            probe_pred = model.predict(xs[val_indices])
            fold_predictions[val_indices] = probe_pred
            for head, values in (("original", original_source[val_indices]), ("probe", probe_pred)):
                fold_rows.append(dict(identity, outer_fold=outer + 1, validation_first_cut=int(val_indices[0] + 1),
                                      validation_last_cut=int(val_indices[-1] + 1), head=head, alpha=chosen if head == "probe" else np.nan,
                                      **measure(y[val_indices], values)))
        if not np.isfinite(fold_predictions).all():
            raise ValueError("Missing outer-fold predictions")
        selected, candidates = select_alpha(xs, y, list(range(5)))
        for candidate in candidates:
            alpha_rows.append(dict(identity, scope="full_source", outer_fold=0,
                                   outer_validation_cuts="none", inner_blocks="1,2,3,4,5",
                                   chosen_alpha=selected, **candidate))
        model = fit_probe(xs, y, np.arange(315), selected)
        probe_source = model.predict(xs)
        probe_target = model.predict(xt)
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        name = f"{row.source}_to_{row.target}_seed_{row.seed}_{row.method}.npz"
        model_path = out / "probe_models" / name
        np.savez(model_path, source_feature_mean=scaler.mean_, source_feature_scale=scaler.scale_,
                 coefficient=ridge.coef_, intercept=np.float64(ridge.intercept_), alpha=np.float64(selected),
                 checkpoint_sha256=np.array(row.checkpoint_sha256))
        model_rows.append(dict(identity, alpha=selected, checkpoint=row.checkpoint,
                               checkpoint_sha256=row.checkpoint_sha256,
                               source_features=row.source_features, target_features=row.target_features,
                               source_feature_sha256=sha(Path(row.source_features)),
                               target_feature_sha256=sha(Path(row.target_features)),
                               probe_model=str(model_path.resolve()), probe_model_sha256=sha(model_path),
                               source_cut_count=315, target_cut_count=315))
        for head, values in (("original", original_source), ("probe", probe_source), ("probe_oof", fold_predictions)):
            source_scores.append(dict(identity, head=head, evaluation="source_full_in_sample" if head != "probe_oof" else "source_contiguous_5fold_oof",
                                      **measure(y, values)))
        for cut in CUTS:
            i = cut - 1
            source_rows.append(dict(identity, cut_index=cut, true_vb=y[i],
                                    original_pred_vb=original_source[i], probe_pred_vb=probe_source[i],
                                    probe_outer_fold_pred_vb=fold_predictions[i],
                                    checkpoint_sha256=row.checkpoint_sha256))
            target_rows.append(dict(identity, cut_index=cut, original_pred_vb=original_target[i],
                                    probe_pred_vb=probe_target[i], checkpoint_sha256=row.checkpoint_sha256))
        print(f"probe prepare {row.source}->{row.target} seed={row.seed} {row.method}, alpha={selected}", flush=True)
    pd.DataFrame(fold_rows).to_csv(out / "source_outer_fold_metrics.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(out / "source_inner_alpha_selection.csv", index=False)
    pd.DataFrame(source_scores).to_csv(out / "source_head_probe_metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(out / "source_predictions.csv", index=False)
    pd.DataFrame(target_rows).to_csv(out / "target_predictions_unscored.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out / "probe_model_index.csv", index=False)
    locked = ["source_outer_fold_metrics.csv", "source_inner_alpha_selection.csv", "source_head_probe_metrics.csv",
              "source_predictions.csv", "target_predictions_unscored.csv", "probe_model_index.csv"]
    manifest = {"status": "predictions_locked_before_target_role_scoring", "target_role_label_uses_in_prepare": 0,
                "checkpoint_count": 60, "target_predictions": 60 * 315, "source_cut_blocks": [f"{i*63+1}-{(i+1)*63}" for i in range(5)],
                "alpha_candidates": ALPHAS, "alpha_selection": "mean RMSE across inner source validation blocks; lower alpha breaks ties",
                "files_sha256": {name: sha(out / name) for name in locked},
                "probe_model_sha256": {Path(r["probe_model"]).name: r["probe_model_sha256"] for r in model_rows}}
    (out / "prediction_lock.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("All 60 probe models and 315-cut target predictions locked before target label read")


def score(args):
    out = args.out_root
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "predictions_locked_before_target_role_scoring" or lock["target_role_label_uses_in_prepare"] != 0:
        raise ValueError("Target prediction lock invalid")
    for name, digest in lock["files_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Locked file changed: {name}")
    for name, digest in lock["probe_model_sha256"].items():
        if sha(out / "probe_models" / name) != digest:
            raise ValueError(f"Locked probe changed: {name}")
    if (out / "direction_summary.csv").exists():
        raise FileExistsError("Target already scored")
    target_pred = pd.read_csv(out / "target_predictions_unscored.csv")
    if len(target_pred) != 60 * 315:
        raise ValueError("Missing target predictions")
    # First target-role use of wear labels occurs here, after the lock.
    truth = {tool: base.wear_labels(args.raw_root, tool).astype(np.float64) for tool in TOOLS}
    scored, metrics_rows, segment_rows = [], [], []
    for (source, target, seed, method), frame in target_pred.groupby(["source", "target", "seed", "method"], sort=True):
        frame = frame.sort_values("cut_index")
        if frame.cut_index.tolist() != CUTS:
            raise ValueError("Target cut mapping invalid")
        y = truth[target]
        for cut, original, probe in zip(CUTS, frame.original_pred_vb, frame.probe_pred_vb):
            scored.append({"source": source, "target": target, "seed": seed, "method": method,
                           "cut_index": cut, "true_vb": y[cut - 1], "original_pred_vb": original,
                           "probe_pred_vb": probe, "original_signed_error": original - y[cut - 1],
                           "probe_signed_error": probe - y[cut - 1]})
        for head in ("original", "probe"):
            p = frame[f"{head}_pred_vb"].to_numpy()
            metrics_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                                 "head": head, **measure(y, p)})
            for name, (first, last) in SEGMENTS.items():
                segment_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                                     "head": head, "segment": name, "first_cut": first, "last_cut": last,
                                     **measure(y[first - 1:last], p[first - 1:last])})
    pd.DataFrame(scored).to_csv(out / "target_predictions_scored.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(out / "target_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(out / "target_segment_metrics.csv", index=False)
    source_metrics = pd.read_csv(out / "source_head_probe_metrics.csv")
    target_metrics = pd.DataFrame(metrics_rows)
    folds = pd.read_csv(out / "source_outer_fold_metrics.csv")
    seed_rows = []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            for seed in SEEDS:
                row = {"source": source, "target": target, "seed": seed}
                for method in METHODS:
                    for head in ("original", "probe"):
                        sm = source_metrics[(source_metrics.source == source) & (source_metrics.target == target) &
                                            (source_metrics.seed == seed) & (source_metrics.method == method) &
                                            (source_metrics["head"] == head)].iloc[0]
                        tm = target_metrics[(target_metrics.source == source) & (target_metrics.target == target) &
                                            (target_metrics.seed == seed) & (target_metrics.method == method) &
                                            (target_metrics["head"] == head)].iloc[0]
                        for metric in ("MAE", "RMSE", "R2"):
                            row[f"{method}_{head}_source_{metric}"] = sm[metric]
                            row[f"{method}_{head}_target_{metric}"] = tm[metric]
                    for metric in ("MAE", "RMSE", "R2"):
                        rows = folds[(folds.source == source) & (folds.target == target) & (folds.seed == seed) &
                                     (folds.method == method) & (folds["head"] == "probe")]
                        row[f"{method}_probe_cv_{metric}_fold_mean"] = float(rows[metric].mean())
                        oof = source_metrics[(source_metrics.source == source) & (source_metrics.target == target) &
                                             (source_metrics.seed == seed) & (source_metrics.method == method) &
                                             (source_metrics["head"] == "probe_oof")].iloc[0]
                        row[f"{method}_probe_cv_{metric}_pooled"] = oof[metric]
                for head in ("original", "probe"):
                    for domain in ("source", "target"):
                        for metric in ("MAE", "RMSE", "R2"):
                            key = f"{head}_{domain}_{metric}"
                            row[f"dare_minus_source_only_{key}"] = row[f"daregram_{key}"] - row[f"source_only_{key}"]
                for method in METHODS:
                    for domain in ("source", "target"):
                        for metric in ("MAE", "RMSE", "R2"):
                            row[f"{method}_probe_minus_original_{domain}_{metric}"] = (
                                row[f"{method}_probe_{domain}_{metric}"] - row[f"{method}_original_{domain}_{metric}"])
                seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(out / "paired_seed_metrics.csv", index=False)
    summary = []
    for (source, target), block in seed_frame.groupby(["source", "target"]):
        row = {"source": source, "target": target, "n_seeds": 5}
        for col in block.columns:
            if col in ("source", "target", "seed"):
                continue
            row[f"{col}_mean"] = float(block[col].mean())
            row[f"{col}_sd"] = float(block[col].std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "direction_summary.csv", index=False)
    plots = out / "plots"
    plots.mkdir()
    scored_frame = pd.DataFrame(scored)
    for source in ("c1", "c4"):
        target = "c6"
        for seed in SEEDS:
            subset = scored_frame[(scored_frame.source == source) & (scored_frame.target == target) &
                                  (scored_frame.seed == seed)]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(CUTS, truth[target], color="black", label="true VB", linewidth=2)
            for method, color in (("source_only", "tab:blue"), ("daregram", "tab:orange")):
                block = subset[subset.method == method].sort_values("cut_index")
                ax.plot(CUTS, block.original_pred_vb, color=color, linestyle="--", label=f"{method} original")
                ax.plot(CUTS, block.probe_pred_vb, color=color, label=f"{method} probe")
            ax.set(xlabel="C6 cut", ylabel="VB", xlim=(1, 315), title=f"{source.upper()} to C6 | seed {seed}")
            ax.legend(ncol=2, fontsize=8)
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(plots / f"{source}_to_c6_seed_{seed}.png", dpi=160)
            plt.close(fig)
    print("Target wear scored after lock: 60 models, 315 cuts each, 10 C6 plots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--prior-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    parser.add_argument("--mode-root", type=Path, default=Path("artifacts/representation_probe_audit_20260926"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/frozen_ridge_probe_audit_20260926"))
    args = parser.parse_args()
    prepare(args) if args.phase == "prepare" else score(args)


if __name__ == "__main__":
    main()
