"""Independent held-out-source DEV selection for six PHM2010 UDA candidates.

Run from upstream-reproduction. Target wear is accessed only in evaluate(),
after selected_models.csv and selection_frozen.json have been serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset

import run_single_source_pairs as base
import run_dann_mmd_joint as joint
import utils
from models.DANN import Discriminator
from models.DAREGRAM import Trainset as DAREGRAM

METHODS = ("source_only", "dann", "mmd", "daregram", "daregram_mmd", "daregram_dann")
DECAYS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
CUTS = np.arange(1, 316)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=Path("artifacts/dev_model_selection_20260926"))
    p.add_argument("--baseline-root", type=Path, default=Path("artifacts/five_seed_paired"))
    p.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    p.add_argument("--pairs", nargs="+", choices=[f"{s}_to_{t}" for s, t in joint.PAIRS])
    p.add_argument("--seeds", nargs="+", type=int, choices=joint.SEEDS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-aggregate", action="store_true", help="Skip shared root CSV writes during disjoint parallel runs")
    p.add_argument("--aggregate-only", action="store_true", help="Merge completed run files without training")
    a = p.parse_args()
    if a.out_root.resolve() == a.baseline_root.resolve():
        p.error("DEV output root must be independent of the original runs")
    return a


def split(source, target, seed):
    """Seven nonoverlapping 3-cut blocks in each chronological wear stage."""
    digest = hashlib.sha256(f"DEV-v1:{source}:{target}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    rows = []
    for stage, start in (("early", 1), ("middle", 106), ("late", 211)):
        groups = rng.choice(35, size=7, replace=False)
        validation = set((start + 3 * int(g) + k) for g in groups for k in range(3))
        for cut in range(start, start + 105):
            rows.append({"source": source, "target": target, "seed": seed, "cut": cut,
                         "wear_stage": stage, "group_of_three": (cut - start) // 3,
                         "split": "val" if cut in validation else "train"})
    frame = pd.DataFrame(rows)
    assert frame.split.value_counts().to_dict() == {"train": 252, "val": 63}
    assert frame.groupby(["wear_stage", "split"]).size().to_dict() == {
        (stage, part): n for stage in ("early", "middle", "late")
        for part, n in (("train", 84), ("val", 21))}
    return frame


def cached_features(a, manifest, tool):
    path = a.baseline_root / "feature_cache" / f"{tool}_stft.npy"
    if base.file_hash(path) != manifest["tools"][tool]["feature_sha256"]:
        raise ValueError(f"STFT cache hash mismatch: {path}")
    x = np.load(path, allow_pickle=False)
    if x.shape != (315, 6, 128, 128) or not np.isfinite(x).all():
        raise ValueError(f"Invalid cached STFT: {path}")
    return x


def prepared(a, manifest, source, target, train_cuts):
    raw_s = cached_features(a, manifest, source)
    raw_t = cached_features(a, manifest, target)
    fit = raw_s[train_cuts - 1]
    mean = fit.mean(axis=(0, 2, 3)).astype(np.float32)
    std = (fit.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    xs = ((raw_s - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    xt = ((raw_t - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    return xs, xt, mean, std


def train(a, method, source, target, seed, train_cuts, xs, ys, xt, folder):
    device = torch.device(a.device)
    model, init_hash = base.build_model(seed, device)
    use_g, use_m, use_d = method.startswith("daregram"), "mmd" in method, "dann" in method
    discriminator = Discriminator().to(device) if use_d else None
    source_loader = DataLoader(TensorDataset(torch.from_numpy(xs[train_cuts - 1]),
                                             torch.from_numpy(ys[train_cuts - 1]),
                                             torch.from_numpy(train_cuts)),
                               batch_size=joint.BATCH_SIZE, shuffle=True,
                               generator=torch.Generator().manual_seed(seed))
    target_loader = None
    if method != "source_only":
        target_loader = DataLoader(TensorDataset(torch.from_numpy(xt), torch.from_numpy(CUTS)),
                                   batch_size=joint.BATCH_SIZE, shuffle=True,
                                   generator=torch.Generator().manual_seed(seed + 1))
    steps = joint.EPOCHS * len(source_loader)
    domain_adv = utils.DomainAdversarialLoss(discriminator, max_iters=steps) if use_d else None
    optimizer = torch.optim.Adam(list(model.parameters()) +
                                 (list(discriminator.parameters()) if use_d else []), lr=joint.LR)
    logs, orders, target_seen = [], [], set()
    for epoch in range(1, joint.EPOCHS + 1):
        model.train()
        if use_d:
            discriminator.train()
        source_order, target_order = [], []
        totals = dict.fromkeys(("wear", "gram", "mmd", "dann", "total"), 0.0)
        factor = joint.ramp(epoch, "exp")
        batches = zip(source_loader, target_loader) if target_loader is not None else ((s, None) for s in source_loader)
        for (xb, yb, sc), target_batch in batches:
            source_order.extend(sc.tolist())
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            fs = model.feature_extractor(xb)
            wear = F.mse_loss(model.regressor(fs), yb)
            gram = mmd = dann = wear.new_zeros(())
            if target_batch is not None:
                tx, tc = target_batch
                target_order.extend(tc.tolist())
                ft = model.feature_extractor(tx.to(device))
                if fs.shape != ft.shape or fs.shape[1] != 512:
                    raise AssertionError("Paired feature batch shape mismatch")
                if use_g:
                    gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), fs, ft)
                if use_m:
                    mmd = joint.mmd_loss(fs, ft)
                if use_d:
                    dann, _ = domain_adv(fs, ft)
            loss = wear + factor * (gram + 0.1 * mmd + 0.1 * dann)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite training loss {folder} epoch {epoch}")
            loss.backward()
            optimizer.step()
            for key, tensor in (("wear", wear), ("gram", gram), ("mmd", mmd), ("dann", dann), ("total", loss)):
                totals[key] += float(tensor.detach())
        if sorted(source_order) != train_cuts.tolist() or set(source_order) & (set(CUTS) - set(train_cuts)):
            raise AssertionError("Source validation cut entered model training")
        if target_order and (len(target_order) != len(train_cuts) or len(set(target_order)) != len(train_cuts)):
            raise AssertionError("Unexpected target batch coverage")
        target_seen.update(target_order)
        orders.append(hashlib.sha256(np.asarray(source_order, np.int32).tobytes()).hexdigest())
        logs.append({"epoch": epoch, "source_cuts": len(source_order), "unlabeled_target_cuts": len(target_order),
                     "source_order_sha256": orders[-1], "ramp": factor,
                     **{k: v / len(source_loader) for k, v in totals.items()}})
        if epoch == 1 or epoch % 10 == 0:
            logging.info("%s %s->%s seed %s epoch %s/%s loss %.4f", method, source, target,
                         seed, epoch, joint.EPOCHS, logs[-1]["total"])
    if method != "source_only" and target_seen != set(CUTS):
        raise AssertionError("Unlabeled target cuts not covered across epochs")
    checkpoint = folder / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": joint.EPOCHS}, checkpoint)
    pd.DataFrame(logs).to_csv(folder / "epoch_losses.csv", index=False)
    return {"initial_model_sha256": init_hash, "source_order_sha256_by_epoch": orders,
            "source_train_cuts": train_cuts.tolist(), "target_unlabeled_cuts_seen": sorted(target_seen),
            "checkpoint_sha256": base.file_hash(checkpoint), "epochs": joint.EPOCHS,
            "batch_size": joint.BATCH_SIZE, "lr": joint.LR,
            "lambda_gram": int(use_g), "lambda_mmd": 0.1 if use_m else 0,
            "lambda_dann": 0.1 if use_d else 0, "checkpoint_rule": "final epoch"}


def export(a, source, target, method, seed, xs, xt, ys, train_cuts, val_cuts, folder):
    model, _ = base.build_model(seed, torch.device(a.device))
    ckpt = torch.load(folder / "final.pth", map_location=a.device, weights_only=True)
    if ckpt["epoch"] != joint.EPOCHS:
        raise AssertionError("Checkpoint is not final epoch")
    model.load_state_dict(ckpt["model"])
    model.eval()
    def forward(x):
        features, predictions = [], []
        with torch.no_grad():
            for offset in range(0, len(x), joint.BATCH_SIZE):
                f = model.feature_extractor(torch.from_numpy(x[offset:offset + joint.BATCH_SIZE]).to(a.device))
                features.append(f.cpu().numpy())
                predictions.append(model.regressor(f).cpu().numpy().ravel())
        return np.concatenate(features).astype(np.float32), np.concatenate(predictions).astype(np.float64)
    fst, _ = forward(xs[train_cuts - 1])
    fsv, yhat_sv = forward(xs[val_cuts - 1])
    ft, yhat_t = forward(xt)
    if any(not np.isfinite(x).all() for x in (fst, fsv, ft, yhat_sv, yhat_t)):
        raise FloatingPointError("Nonfinite exported feature or prediction")
    np.savez_compressed(folder / "dev_arrays.npz", source=np.asarray(source),
                        target=np.asarray(target), seed=np.asarray(seed), method=np.asarray(method),
                        source_train_cut=train_cuts,
                        F_s_train=fst, source_val_cut=val_cuts, F_s_val=fsv,
                        yhat_s_val=yhat_sv, y_s_val=ys[val_cuts - 1], target_cut=CUTS,
                        F_t=ft, yhat_t=yhat_t)
    return fst, fsv, ft, yhat_sv, ys[val_cuts - 1], yhat_t


def dev_score(fst, fsv, ft, pred_val, y_val, seed):
    """Official get_weight MLP and get_dev_risk formula with explicit safeguards."""
    ns, nt = len(fst), len(ft)
    if fst.shape[1] != 512 or fsv.shape[1] != 512 or ft.shape[1] != 512:
        raise AssertionError("DEV expects candidate-specific 512-dimensional features")
    domain_x = np.concatenate((fst, ft))
    domain_y = np.r_[np.ones(ns, np.int32), np.zeros(nt, np.int32)]
    xtr, xte, ytr, yte = train_test_split(domain_x, domain_y, train_size=0.8,
                                          stratify=domain_y, random_state=seed)
    scaler = StandardScaler().fit(xtr)
    xtr = scaler.transform(xtr)
    xte = scaler.transform(xte)
    fsv_scaled = scaler.transform(fsv)
    scores, models = [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            for decay in DECAYS:
                clf = MLPClassifier(hidden_layer_sizes=(512, 512, 2), activation="relu",
                                    alpha=decay, random_state=seed)
                clf.fit(xtr, ytr)
                scores.append(float(accuracy_score(yte, clf.predict(xte))))
                models.append(clf)
    best_idx = int(np.argmax(scores))
    clf = models[best_idx]
    proba = clf.predict_proba(fsv_scaled)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weight = proba[:, 0] / proba[:, 1] * (ns / nt)
    error = np.abs(pred_val.astype(np.float64) - y_val.astype(np.float64))
    flags = []
    if not np.isfinite(weight).all() or (weight < 0).any():
        flags.append("nonfinite_or_negative_weight")
    valid = np.isfinite(weight).all() and (weight >= 0).all()
    var_w = float(np.var(weight, ddof=1)) if valid else None
    if valid and var_w <= 1e-12:
        flags.append("near_zero_weight_variance")
        valid = False
    score = eta = None
    if valid:
        weighted = weight * error
        cov = float(np.cov(np.column_stack((weighted, weight)), rowvar=False)[0, 1])
        eta = -cov / var_w
        score = float(np.mean(weighted) + eta * (np.mean(weight) - 1))
        if not math.isfinite(score):
            flags.append("nonfinite_dev_score")
            score = eta = None
    quant = np.quantile(weight, [0, .01, .05, .25, .5, .75, .95, .99, 1]) if np.isfinite(weight).all() else [None] * 9
    sumsq = float(np.sum(weight**2)) if np.isfinite(weight).all() else 0.0
    diag = {"domain_train_source_n": int(sum(ytr)), "domain_train_target_n": int(len(ytr) - sum(ytr)),
            "domain_holdout_source_n": int(sum(yte)), "domain_holdout_target_n": int(len(yte) - sum(yte)),
            "domain_val_accuracy": scores[best_idx],
            "domain_val_balanced_accuracy": float(balanced_accuracy_score(yte, clf.predict(xte))),
            "domain_val_auc": float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1])),
            "domain_alpha": DECAYS[best_idx], "domain_alpha_accuracy_grid": json.dumps(scores),
            "mlp_warning_count": len(caught), "weight_prior_factor": ns / nt,
            "domain_feature_scaling": "StandardScaler fit on domain classifier training subset only",
            "weight_var": var_w, "weight_max": float(np.max(weight)) if np.isfinite(weight).all() else None,
            "weight_ess": float(np.sum(weight)**2 / sumsq) if sumsq > 0 else None,
            "weight_mean": float(np.mean(weight)) if np.isfinite(weight).all() else None,
            "eta": eta, "dev_score": score, "status": "ok" if score is not None else "invalid",
            "anomalies": ";".join(flags)}
    diag.update({f"weight_q{int(q*100):02d}": float(v) if v is not None else None
                 for q, v in zip((0, .01, .05, .25, .5, .75, .95, .99, 1), quant)})
    # DEV-guard: a simpler regularized domain model supplies a comparable
    # fallback score for *all* six candidates if any raw MLP risk is invalid.
    with threadpool_limits(limits=1):
        guard_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(xtr, ytr)
    guard_proba = guard_clf.predict_proba(fsv_scaled)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        guard_weight = guard_proba[:, 0] / guard_proba[:, 1] * (ns / nt)
    guard_var = float(np.var(guard_weight, ddof=1)) if np.isfinite(guard_weight).all() else None
    guard_score_value = None
    guard_status = "invalid"
    if guard_var is not None and guard_var > 1e-12 and (guard_weight >= 0).all():
        guard_loss = guard_weight * error
        guard_eta = -float(np.cov(np.column_stack((guard_loss, guard_weight)), rowvar=False)[0, 1]) / guard_var
        guard_score_value = float(np.mean(guard_loss) + guard_eta * (np.mean(guard_weight) - 1))
        if math.isfinite(guard_score_value):
            guard_status = "ok"
        else:
            guard_score_value = None
    guard_sumsq = float(np.sum(guard_weight**2)) if np.isfinite(guard_weight).all() else 0.0
    diag.update({"dev_guard_score": guard_score_value, "dev_guard_status": guard_status,
                 "dev_guard_domain_val_accuracy": float(accuracy_score(yte, guard_clf.predict(xte))),
                 "dev_guard_weight_ess": float(np.sum(guard_weight)**2 / guard_sumsq) if guard_sumsq > 0 else None,
                 "dev_guard_weight_max": float(np.max(guard_weight)) if np.isfinite(guard_weight).all() else None})
    return diag, weight, guard_weight, error


def one(a, manifest, source, target, seed):
    root = a.out_root / "runs" / f"{source}_to_{target}" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    split_frame = split(source, target, seed)
    train_cuts = split_frame.loc[split_frame.split == "train", "cut"].to_numpy(np.int64)
    val_cuts = split_frame.loc[split_frame.split == "val", "cut"].to_numpy(np.int64)
    split_path = root / "source_split.csv"
    if split_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(split_path), split_frame)
    else:
        split_frame.to_csv(split_path, index=False)
    xs, xt, mean, std = prepared(a, manifest, source, target, train_cuts)
    ys = base.wear_labels(a.raw_root, source)
    score_rows, pred_rows = [], []
    for method in METHODS:
        folder = root / method
        folder.mkdir(exist_ok=True)
        cfg_path = folder / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg["checkpoint_sha256"] != base.file_hash(folder / "final.pth") or cfg["source_train_cuts"] != train_cuts.tolist():
                raise AssertionError(f"Existing run provenance changed: {folder}")
        else:
            if (folder / "final.pth").exists():
                raise FileExistsError(f"Incomplete run requires inspection: {folder}")
            cfg = train(a, method, source, target, seed, train_cuts, xs, ys, xt, folder)
            cfg.update({"source": source, "target": target, "seed": seed, "method": method,
                        "source_val_cuts": val_cuts.tolist(), "normalization_fit_cuts": train_cuts.tolist(),
                        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
                        "target_label_reads_before_selection": 0})
            write_json(cfg_path, cfg)
        fst, fsv, ft, pv, yv, pt = export(a, source, target, method, seed,
                                        xs, xt, ys, train_cuts, val_cuts, folder)
        diag, weight, guard_weight, error = dev_score(fst, fsv, ft, pv, yv, seed)
        score_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                           "checkpoint_sha256": cfg["checkpoint_sha256"], **diag})
        pd.DataFrame({"source": source, "target": target, "seed": seed, "method": method,
                      "cut": val_cuts, "weight": weight, "dev_guard_weight": guard_weight,
                      "absolute_error_vb": error}).to_csv(
                          folder / "dev_validation_per_cut.csv", index=False, float_format="%.17g")
        pred_rows.extend({"source": source, "target": target, "seed": seed, "method": method,
                          "cut": int(c), "pred_vb": float(p)} for c, p in zip(CUTS, pt))
        logging.info("DEV %s %s->%s seed %s score %s ESS %s", method, source, target, seed,
                     diag["dev_score"], diag["weight_ess"])
    scores = pd.DataFrame(score_rows)
    scores.to_csv(root / "dev_scores.csv", index=False, float_format="%.17g")
    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(root / "candidate_predictions_before_target_labels.csv", index=False, float_format="%.17g")
    variant = "DEV" if (scores.status == "ok").all() else "DEV-guard"
    score_column = "dev_score" if variant == "DEV" else "dev_guard_score"
    status_column = "status" if variant == "DEV" else "dev_guard_status"
    eligible = scores.loc[scores[status_column] == "ok"].sort_values([score_column, "method"])
    if eligible.empty:
        raise RuntimeError(f"No finite {variant} score for {source}->{target} seed {seed}")
    selected = eligible.iloc[0]
    selection = {"source": source, "target": target, "seed": seed, "selected_method": selected.method,
                 "selection_variant": variant, "selected_score": float(selected[score_column]),
                 "selected_raw_dev_score": float(selected.dev_score) if pd.notna(selected.dev_score) else None,
                 "selected_checkpoint_sha256": selected.checkpoint_sha256,
                 "candidate_count": len(METHODS), "valid_dev_count": len(eligible),
                 "ranking": ";".join(eligible.method.tolist()), "selection_uses_target_vb": False}
    pd.DataFrame([selection]).to_csv(root / "selected_models.csv", index=False)
    write_json(root / "selection_frozen.json", selection)
    evaluate(a, root, selection, scores, predictions)
    return split_frame


def evaluate(a, root, selection, scores, predictions):
    """This is the first and only target wear access in the selection path."""
    source, target, seed = selection["source"], selection["target"], selection["seed"]
    frozen = json.loads((root / "selection_frozen.json").read_text(encoding="utf-8"))
    if frozen != selection or not (root / "selected_models.csv").exists():
        raise AssertionError("Target labels requested before selection was frozen")
    if predictions.groupby(["method", "cut"]).size().ne(1).any() or len(predictions) != len(METHODS) * 315:
        raise AssertionError("Candidate predictions must have exactly one row per method and target cut")
    y = base.wear_labels(a.raw_root, target).astype(np.float64)
    metrics_rows = []
    for method in METHODS:
        p = predictions.loc[predictions.method == method].sort_values("cut").pred_vb.to_numpy()
        metrics_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                             **{k: v for k, v in base.metrics(y, p).items() if k in ("R2", "MAE", "RMSE")}})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(root / "candidate_target_metrics.csv", index=False, float_format="%.17g")
    oracle = metrics.sort_values(["MAE", "method"]).iloc[0]
    chosen = selection["selected_method"]
    final = predictions.loc[predictions.method == chosen].sort_values("cut").copy()
    if final.cut.tolist() != CUTS.tolist():
        raise AssertionError("Each target cut must have exactly one final prediction")
    final["true_vb"] = y
    final.to_csv(root / "predictions_per_cut.csv", index=False, float_format="%.17g")
    comparison = {"source": source, "target": target, "seed": seed, "selected_method": chosen,
                  "oracle_method_diagnostic_only": oracle.method,
                  "oracle_MAE": float(oracle.MAE),
                  "worst_method": metrics.sort_values(["MAE", "method"]).iloc[-1].method}
    for label, method in (("selected", chosen), ("daregram", "daregram"),
                          ("best", oracle.method), ("worst", comparison["worst_method"])):
        row = metrics.set_index("method").loc[method]
        comparison.update({f"{label}_{key}": float(row[key]) for key in ("R2", "MAE", "RMSE")})
    pd.DataFrame([comparison]).to_csv(root / "selected_vs_daregram_best_worst.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(CUTS, y, label="True VB", linewidth=1.6)
    ax.plot(CUTS, final.pred_vb, label=f"DEV selected: {chosen}", linewidth=1.3)
    ax.set(xlabel=f"{target.upper()} cut", ylabel="VB", title=f"{source.upper()} to {target.upper()}, seed {seed}")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(root / "selected_prediction_curve.png", dpi=180)
    plt.close(fig)


def aggregate(a):
    roots = sorted((a.out_root / "runs").glob("*_to_*/seed_*"))
    files = ("source_split.csv", "dev_scores.csv", "selected_models.csv", "predictions_per_cut.csv",
             "candidate_target_metrics.csv", "selected_vs_daregram_best_worst.csv")
    for file in files:
        paths = [root / file for root in roots if (root / file).exists()]
        if paths:
            pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
                a.out_root / file, index=False, float_format="%.17g")
    if not (a.out_root / "candidate_target_metrics.csv").exists():
        return
    metrics = pd.read_csv(a.out_root / "candidate_target_metrics.csv")
    selected = pd.read_csv(a.out_root / "selected_models.csv")
    rows = metrics.merge(selected[["source", "target", "seed", "selected_method"]],
                         on=["source", "target", "seed"], validate="many_to_one")
    rows["role"] = np.where(rows.method == rows.selected_method, "selected", "candidate")
    summary = rows.groupby(["source", "target", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"))
    selected_rows = rows.loc[rows.role == "selected"].groupby(["source", "target"], as_index=False).agg(
        seeds=("seed", "nunique"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"))
    selected_rows["method"] = "DEV_selected"
    pd.concat((summary, selected_rows), ignore_index=True).to_csv(
        a.out_root / "six_direction_summary.csv", index=False, float_format="%.17g")


def main():
    a = arguments()
    a.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(a.out_root / "run.log", encoding="utf-8"), logging.StreamHandler()])
    manifest = json.loads((a.baseline_root / "feature_cache" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["stft_code_sha256"] != base.file_hash(Path(base.sampling.__file__)):
        raise AssertionError("STFT implementation differs from feature cache")
    if a.aggregate_only:
        aggregate(a)
        return
    pairs = [p for p in joint.PAIRS if a.pairs is None or f"{p[0]}_to_{p[1]}" in a.pairs]
    seeds = joint.SEEDS if a.seeds is None else a.seeds
    for source, target in pairs:
        for seed in seeds:
            logging.info("Starting DEV %s->%s seed %d", source, target, seed)
            one(a, manifest, source, target, seed)
            if not a.no_aggregate:
                aggregate(a)
    logging.info("Complete: %s", a.out_root)


if __name__ == "__main__":
    main()
