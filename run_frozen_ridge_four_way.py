"""Independent four-way frozen ResNet18 / source Ridge comparison.

Run prepare, then score. prepare never reads target-role wear or scored CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import audit_frozen_ridge_probe as prior_code
import run_single_source_pairs as base

CUTS = list(range(1, 316))
METHODS = {("source_only", "original"): "A", ("source_only", "probe"): "B",
           ("daregram", "original"): "C", ("daregram", "probe"): "D"}
METRICS = ("R2", "MAE", "RMSE")
LOCKED = ("predictions_unscored.csv", "ridge_model_index.csv", "alpha_selection.csv",
          "linear_replay_audit.csv", "source_max_vb.csv")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked_cuts(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if len(frame) != 315 or frame.cut_index.isna().any() or frame.cut_index.duplicated().any():
        raise ValueError(f"315 unique cuts required: {key}")
    result = frame.sort_values("cut_index")
    if result.cut_index.astype(int).tolist() != CUTS:
        raise ValueError(f"Missing/out-of-range cut: {key}")
    return result


def prepare(args):
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    old = args.prior_root
    lock = json.loads((old / "prediction_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "predictions_locked_before_target_role_scoring" or lock["checkpoint_count"] != 60:
        raise ValueError("Prior prediction lock invalid")
    for name, digest in lock["files_sha256"].items():
        if sha(old / name) != digest:
            raise ValueError(f"Prior locked file changed: {name}")
    for name, digest in lock["probe_model_sha256"].items():
        if sha(old / "probe_models" / name) != digest:
            raise ValueError(f"Prior Ridge model changed: {name}")
    index = pd.read_csv(old / "probe_model_index.csv")
    old_pred = pd.read_csv(old / "target_predictions_unscored.csv")
    old_source = pd.read_csv(old / "source_predictions.csv")
    alpha = pd.read_csv(old / "source_inner_alpha_selection.csv")
    provenance = pd.read_csv(args.provenance_root / "provenance.csv")
    if len(index) != 60 or len(provenance) != 60 or len(old_pred) != 18900:
        raise ValueError("Expected 60 checkpoint records and 18,900 target rows")
    records = index.merge(provenance[["source", "target", "seed", "method", "checkpoint_sha256",
                                      "prediction_csv", "feature_audit"]],
                          on=["source", "target", "seed", "method", "checkpoint_sha256"], validate="one_to_one")
    if len(records) != 60:
        raise ValueError("Checkpoint provenance mismatch")
    args.out_root.mkdir(parents=True)
    models_dir = args.out_root / "ridge_models"
    models_dir.mkdir()
    pred_rows, model_rows, replay_rows, source_max = [], [], [], {}
    for row in records.itertuples(index=False):
        key = (row.source, row.target, row.seed, row.method)
        identity = dict(zip(("source", "target", "seed", "feature_method"), key))
        checkpoint = Path(row.checkpoint)
        if sha(checkpoint) != row.checkpoint_sha256:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
        audit = json.loads(Path(row.feature_audit).read_text(encoding="utf-8"))
        xs = prior_code.checked_features(Path(row.source_features), row.source)
        xt = prior_code.checked_features(Path(row.target_features), row.target)
        for role, path in (("source", Path(row.source_features)), ("target", Path(row.target_features))):
            if sha(path) != audit["feature_stats"][role]["file_sha256"]:
                raise ValueError(f"Feature hash changed: {path}")
        y_source = base.wear_labels(args.raw_root, row.source).astype(np.float64)
        source_max[row.source] = float(y_source.max())
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        w = state["regressor.0.weight"].numpy().reshape(-1).astype(np.float64)
        b = float(state["regressor.0.bias"].item())
        linear = xt @ w + b
        historical = checked_cuts(pd.read_csv(row.prediction_csv, usecols=["cut_index", "pred_vb"]), str(key))
        op = checked_cuts(old_pred.loc[(old_pred.source == row.source) & (old_pred.target == row.target) &
                                       (old_pred.seed == row.seed) & (old_pred.method == row.method)], str(key))
        sp = checked_cuts(old_source.loc[(old_source.source == row.source) & (old_source.target == row.target) &
                                         (old_source.seed == row.seed) & (old_source.method == row.method)], str(key))
        if not np.allclose(sp.true_vb.to_numpy(), y_source, rtol=0, atol=1e-5):
            raise ValueError(f"Source label mismatch: {key}")
        if not np.allclose(sp.original_pred_vb.to_numpy(), xs @ w + b, rtol=0, atol=1e-8):
            raise ValueError(f"Source Linear replay mismatch: {key}")
        if not np.allclose(op.original_pred_vb.to_numpy(), linear, rtol=0, atol=1e-8):
            raise ValueError(f"Prior target Linear replay mismatch: {key}")
        delta = np.abs(historical.pred_vb.to_numpy() - linear)
        if delta.max() > 0.005:
            raise ValueError(f"Historical versus feature replay exceeds 0.005 VB: {key}, {delta.max()}")
        replay_rows.append(dict(identity, checkpoint_sha256=row.checkpoint_sha256,
                                historical_prediction_sha256=sha(Path(row.prediction_csv)),
                                max_abs_vb_difference=float(delta.max()), mean_abs_vb_difference=float(delta.mean())))
        model_path = Path(row.probe_model)
        if sha(model_path) != row.probe_model_sha256:
            raise ValueError(f"Ridge model hash changed: {key}")
        model = np.load(model_path)
        selected = alpha.loc[(alpha.source == row.source) & (alpha.target == row.target) &
                             (alpha.seed == row.seed) & (alpha.method == row.method) &
                             (alpha.scope == "full_source")]
        if len(selected) != 6 or sorted(selected.alpha.tolist()) != list(prior_code.ALPHAS):
            raise ValueError(f"Alpha grid differs: {key}")
        choice = selected.sort_values(["inner_rmse_mean", "alpha"]).iloc[0]
        if not (float(choice.alpha) == float(row.alpha) == float(model["alpha"])):
            raise ValueError(f"Alpha selection mismatch: {key}")
        # The locked model contains the full-source StandardScaler and Ridge parameters.
        if not np.allclose(model["source_feature_mean"], xs.mean(axis=0), rtol=1e-10, atol=1e-10):
            raise ValueError(f"Scaler not fitted on full source: {key}")
        ridge_target = ((xt - model["source_feature_mean"]) / model["source_feature_scale"]) @ model["coefficient"] + model["intercept"]
        ridge_source = ((xs - model["source_feature_mean"]) / model["source_feature_scale"]) @ model["coefficient"] + model["intercept"]
        if not np.allclose(ridge_target, op.probe_pred_vb, rtol=0, atol=1e-8) or not np.allclose(ridge_source, sp.probe_pred_vb, rtol=0, atol=1e-8):
            raise ValueError(f"Locked Ridge replay mismatch: {key}")
        copied = models_dir / model_path.name
        shutil.copyfile(model_path, copied)
        model_rows.append(dict(identity, checkpoint=str(checkpoint), checkpoint_sha256=row.checkpoint_sha256,
                               source_features=row.source_features, source_feature_sha256=row.source_feature_sha256,
                               target_features=row.target_features, target_feature_sha256=row.target_feature_sha256,
                               ridge_model=str(copied.resolve()), ridge_model_sha256=sha(copied), ridge_alpha=row.alpha))
        for method, values in ((METHODS[(row.method, "original")], historical.pred_vb.to_numpy()),
                               (METHODS[(row.method, "probe")], op.probe_pred_vb.to_numpy())):
            for cut, pred in zip(CUTS, values):
                pred_rows.append(dict(source=row.source, target=row.target, seed=row.seed, method=method,
                                      cut_index=cut, pred_vb=float(pred), checkpoint_sha256=row.checkpoint_sha256,
                                      ridge_alpha=float(row.alpha) if method in ("B", "D") else np.nan))
    output = pd.DataFrame(pred_rows).sort_values(["source", "target", "seed", "method", "cut_index"])
    if len(output) != 6 * 5 * 4 * 315 or output.duplicated(["source", "target", "seed", "method", "cut_index"]).any():
        raise ValueError("Four-way prediction count or key error")
    for key, frame in output.groupby(["source", "target", "seed", "method"]):
        checked_cuts(frame, str(key))
    output.to_csv(args.out_root / "predictions_unscored.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.out_root / "ridge_model_index.csv", index=False)
    pd.DataFrame(replay_rows).to_csv(args.out_root / "linear_replay_audit.csv", index=False)
    pd.DataFrame([{"source": k, "source_max_vb": v} for k, v in sorted(source_max.items())]).to_csv(args.out_root / "source_max_vb.csv", index=False)
    shutil.copyfile(old / "source_inner_alpha_selection.csv", args.out_root / "alpha_selection.csv")
    manifest = {"status": "four_way_predictions_locked_before_target_role_scoring", "target_role_label_uses_in_prepare": 0,
                "methods": {"A": "source-only Linear", "B": "source-only Ridge", "C": "DARE-GRAM Linear", "D": "DARE-GRAM Ridge"},
                "prior_prediction_lock_sha256": sha(old / "prediction_lock.json"),
                "files_sha256": {name: sha(args.out_root / name) for name in LOCKED},
                "ridge_model_sha256": {p.name: sha(p) for p in models_dir.glob("*.npz")},
                "count": len(output), "source_blocks": [f"{i*63+1}-{(i+1)*63}" for i in range(5)],
                "alpha_grid": list(prior_code.ALPHAS), "alpha_rule": "mean source-block RMSE, lower alpha on ties"}
    (args.out_root / "prediction_lock.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Locked {len(output)} target predictions before target-role labels; max Linear replay gap {max(r['max_abs_vb_difference'] for r in replay_rows):.6g} VB")


def score(args):
    out = args.out_root
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "four_way_predictions_locked_before_target_role_scoring" or lock["target_role_label_uses_in_prepare"] != 0:
        raise ValueError("Prediction lock invalid")
    for name, digest in lock["files_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Locked file changed: {name}")
    for name, digest in lock["ridge_model_sha256"].items():
        if sha(out / "ridge_models" / name) != digest:
            raise ValueError(f"Locked Ridge changed: {name}")
    if (out / "predictions_per_cut.csv").exists():
        raise FileExistsError("Already scored")
    pred = pd.read_csv(out / "predictions_unscored.csv")
    if len(pred) != lock["count"] or pred.duplicated(["source", "target", "seed", "method", "cut_index"]).any():
        raise ValueError("Prediction keys invalid")
    for key, frame in pred.groupby(["source", "target", "seed", "method"]):
        checked_cuts(frame, str(key))
    # Target-role truth first enters this independent experiment after the lock check.
    truth = {tool: base.wear_labels(args.raw_root, tool).astype(np.float64) for tool in prior_code.TOOLS}
    maxima = dict(zip(pd.read_csv(out / "source_max_vb.csv").source,
                      pd.read_csv(out / "source_max_vb.csv").source_max_vb))
    pred["true_vb"] = [truth[t][int(c)-1] for t, c in zip(pred.target, pred.cut_index)]
    pred["signed_error"] = pred.pred_vb - pred.true_vb
    pred = pred[["source", "target", "seed", "method", "cut_index", "true_vb", "pred_vb",
                 "signed_error", "checkpoint_sha256", "ridge_alpha"]]
    pred.to_csv(out / "predictions_per_cut.csv", index=False)
    seed_rows, segment_rows = [], []
    for (source, target, seed), block in pred.groupby(["source", "target", "seed"], sort=True):
        row = {"source": source, "target": target, "seed": seed}
        for method in "ABCD":
            part = checked_cuts(block.loc[block.method == method], f"{source}-{target}-{seed}-{method}")
            y, p = part.true_vb.to_numpy(), part.pred_vb.to_numpy()
            for metric, value in prior_code.measure(y, p).items():
                if metric in METRICS:
                    row[f"{method}_{metric}"] = value
            masks = {"early_1_105": part.cut_index.le(105).to_numpy(),
                     "middle_106_210": part.cut_index.between(106, 210).to_numpy(),
                     "late_211_315": part.cut_index.ge(211).to_numpy(),
                     "above_source_max_vb": y > maxima[source]}
            for name, mask in masks.items():
                stats = prior_code.measure(y[mask], p[mask]) if mask.any() else {k: np.nan for k in ("MAE", "RMSE", "R2", "signed_error")}
                segment_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                                     "segment": name, "n_cuts": int(mask.sum()), "source_max_vb": maxima[source],
                                     **{k: stats[k] for k in ("MAE", "RMSE", "R2", "signed_error")}})
        for label, left, right in (("B_minus_A", "B", "A"), ("D_minus_C", "D", "C"), ("D_minus_B", "D", "B")):
            for metric in METRICS:
                row[f"{label}_{metric}"] = row[f"{left}_{metric}"] - row[f"{right}_{metric}"]
        seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(out / "seed_summary.csv", index=False)
    directions = []
    for (source, target), block in seeds.groupby(["source", "target"], sort=True):
        row = {"source": source, "target": target, "n_seeds": len(block)}
        for col in block.columns.difference(["source", "target", "seed"]):
            row[f"{col}_mean"] = float(block[col].mean())
            row[f"{col}_sd"] = float(block[col].std(ddof=1))
        for label in ("B_minus_A", "D_minus_C", "D_minus_B"):
            row[f"{label}_improved_seeds_MAE"] = int((block[f"{label}_MAE"] < 0).sum())
            row[f"{label}_improved_seeds_RMSE"] = int((block[f"{label}_RMSE"] < 0).sum())
            row[f"{label}_improved_seeds_R2"] = int((block[f"{label}_R2"] > 0).sum())
        directions.append(row)
    pd.DataFrame(directions).to_csv(out / "direction_summary.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(out / "segment_summary.csv", index=False)
    plots = out / "plots"
    plots.mkdir()
    colors = {"A": "#1f77b4", "B": "#17becf", "C": "#ff7f0e", "D": "#d62728"}
    for (source, target), block in pred.groupby(["source", "target"], sort=True):
        fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True, sharey=True)
        for ax, seed in zip(axes, range(42, 47)):
            subset = block.loc[block.seed == seed]
            ax.plot(CUTS, truth[target], color="black", linewidth=1.6, label="True VB")
            for method in "ABCD":
                part = checked_cuts(subset.loc[subset.method == method], f"plot-{seed}-{method}")
                ax.plot(CUTS, part.pred_vb, color=colors[method], linewidth=1, label=method)
            ax.set(ylabel=f"seed {seed}\nVB", xlim=(1, 315))
            ax.grid(alpha=.2)
        axes[0].legend(ncol=5, fontsize=9)
        axes[-1].set_xlabel("Cut index")
        fig.suptitle(f"{source.upper()} to {target.upper()}: full 315-cut four-way comparison")
        fig.tight_layout()
        fig.savefig(plots / f"{source}_to_{target}_all_seeds.png", dpi=160)
        plt.close(fig)
        if target == "c6" and source in ("c1", "c4"):
            for seed in range(42, 47):
                fig, ax = plt.subplots(figsize=(12, 5))
                subset = block.loc[block.seed == seed]
                ax.plot(CUTS, truth[target], color="black", linewidth=1.8, label="True VB")
                for method in "ABCD":
                    part = checked_cuts(subset.loc[subset.method == method], f"detail-{seed}-{method}")
                    ax.plot(CUTS, part.pred_vb, color=colors[method], linewidth=1.1, label=method)
                ax.set(xlabel="Cut index", ylabel="VB", xlim=(1, 315), title=f"{source.upper()} to C6 | seed {seed}")
                ax.legend(ncol=5)
                ax.grid(alpha=.2)
                fig.tight_layout()
                fig.savefig(plots / f"{source}_to_c6_seed_{seed}.png", dpi=160)
                plt.close(fig)
    print("Scored 120 groups x 315 cuts from the locked CSV; generated 6 direction and 10 C6 seed plots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--prior-root", type=Path, default=Path("artifacts/frozen_ridge_probe_audit_20260926"))
    parser.add_argument("--provenance-root", type=Path, default=Path("artifacts/target_uda_failure_audit_20260926"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/frozen_ridge_four_way_20260926"))
    args = parser.parse_args()
    prepare(args) if args.phase == "prepare" else score(args)


if __name__ == "__main__":
    main()
