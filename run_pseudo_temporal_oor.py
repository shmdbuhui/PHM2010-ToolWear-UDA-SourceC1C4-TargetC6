"""Source-labelled pseudo/temporal ablation from the locked DARE+Ridge model.

Phases: train (resumable), lock (target-label-free), score (target labels only).
The OOR-PGA multiplier is imported from the repository's locked implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

import audit_frozen_ridge_probe as ridge_code
import run_frozen_ridge_four_way as four
import run_single_source_pairs as base

PAIRS = [(s, t) for s in ("c1", "c4", "c6") for t in ("c1", "c4", "c6") if s != t]
SEEDS = range(42, 47)
BRANCHES = ("E0", "E1", "E2", "E3")
EPOCHS = 2
BATCH = 63
LR_ENCODER = 1e-5
LR_HEAD = 1e-4
PSEUDO_WEIGHT = 0.1
TEMPORAL_WEIGHT = 1.0
CONFIDENCE_QUANTILE = 0.90
MAX_REPLAY_GAP = 0.01
CUTS = list(range(1, 316))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_locked_four(root):
    lock = json.loads((root / "prediction_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "four_way_predictions_locked_before_target_role_scoring":
        raise ValueError("Four-way prediction lock invalid")
    for name, digest in lock["files_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Four-way locked file changed: {name}")
    for name, digest in lock["ridge_model_sha256"].items():
        if sha(root / "ridge_models" / name) != digest:
            raise ValueError(f"Four-way Ridge changed: {name}")
    return pd.read_csv(root / "predictions_unscored.csv"), pd.read_csv(root / "ridge_model_index.csv")


def fold_ridge(model_file):
    item = np.load(model_file)
    coef = np.asarray(item["coefficient"], dtype=np.float64).reshape(-1)
    mean = np.asarray(item["source_feature_mean"], dtype=np.float64)
    scale = np.asarray(item["source_feature_scale"], dtype=np.float64)
    weight = coef / scale
    bias = float(item["intercept"] - np.dot(weight, mean))
    return weight, bias, float(item["alpha"])


def checked_315(frame, name):
    if len(frame) != 315 or frame.cut_index.duplicated().any() or frame.sort_values("cut_index").cut_index.tolist() != CUTS:
        raise ValueError(f"Expected exactly cuts 1..315: {name}")
    return frame.sort_values("cut_index")


def features(path, tool):
    return ridge_code.checked_features(Path(path), tool)


def initial_model(row, seed, device, xs, xt):
    if sha(row.checkpoint) != row.checkpoint_sha256 or sha(row.ridge_model) != row.ridge_model_sha256:
        raise ValueError("DARE checkpoint or Ridge model changed")
    model, _ = base.build_model(seed, device)
    model.load_state_dict(torch.load(row.checkpoint, map_location="cpu", weights_only=True)["model"])
    weight, bias, alpha = fold_ridge(row.ridge_model)
    direct = np.load(row.ridge_model)
    for x in (xs, xt):
        pipeline = ((x - direct["source_feature_mean"]) / direct["source_feature_scale"]) @ direct["coefficient"] + direct["intercept"]
        folded = x @ weight + bias
        if np.max(np.abs(pipeline - folded)) > 1e-7:
            raise ValueError("Ridge-to-Linear fold mismatch")
    with torch.no_grad():
        model.regressor[0].weight.copy_(torch.from_numpy(weight.astype(np.float32)).to(device).view(1, -1))
        model.regressor[0].bias.copy_(torch.tensor([bias], dtype=torch.float32, device=device))
    model.eval()
    return model, alpha


def raw_cache(cache, tool, manifest):
    entry = manifest["tools"][tool]
    path = cache / f"{tool}_stft.npy"
    if sha(path) != entry["feature_sha256"]:
        raise ValueError(f"STFT cache changed: {path}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (315, 6, 128, 128) or x.dtype != np.float32:
        raise ValueError("STFT shape or dtype changed")
    return x


def norm_batch(raw, indices, mean, std, device):
    x = torch.from_numpy(np.asarray(raw[indices], dtype=np.float32).copy()).to(device)
    return (x - mean) / std


def infer(model, raw, mean, std, device, with_features=False):
    model.eval()  # BN running statistics are frozen in both training and inference.
    ps, fs = [], []
    with torch.no_grad():
        for start in range(0, 315, BATCH):
            ids = np.arange(start, min(start + BATCH, 315))
            xb = norm_batch(raw, ids, mean, std, device)
            f = model.feature_extractor(xb)
            ps.append(model.regressor(f).cpu().numpy().reshape(-1))
            if with_features:
                fs.append(f.cpu().numpy())
    pred = np.concatenate(ps).astype(np.float64)
    return (pred, np.concatenate(fs).astype(np.float64)) if with_features else pred


def pseudo_rule(xs, xt, y_source, teacher_pred, ridge_file):
    saved = np.load(ridge_file)
    mean, scale = saved["source_feature_mean"], saved["source_feature_scale"]
    zs, zt = (xs - mean) / scale, (xt - mean) / scale
    ss = cdist(zs, zs)
    np.fill_diagonal(ss, np.inf)
    threshold = float(np.quantile(ss.min(axis=1), CONFIDENCE_QUANTILE))
    distance = cdist(zt, zs).min(axis=1)
    selected = (distance <= threshold) & (teacher_pred >= y_source.min()) & (teacher_pred <= y_source.max())
    weight = np.where(selected, np.maximum(0.1, 1.0 - distance / threshold), 0.0)
    late_reference = float(np.median(y_source[210:]))
    return pd.DataFrame({"cut_index": CUTS, "teacher_pseudo_vb": teacher_pred,
                         "nearest_source_feature_distance": distance, "source_distance_q90": threshold,
                         "selected": selected, "confidence_weight": weight,
                         "source_label_min": float(y_source.min()), "source_label_max": float(y_source.max()),
                         "source_late_median_vb": late_reference,
                         "late_below_source_median_flag": (np.arange(315) >= 210) & (teacher_pred < late_reference)})


def run_one(source, target, seed, args, pred_four, model_index, cache_manifest, device):
    folder = args.out_root / "runs" / f"{source}_to_{target}" / f"seed_{seed}"
    if (folder / "complete.json").exists():
        return "skip"
    if folder.exists():
        raise FileExistsError(f"Incomplete run requires inspection: {folder}")
    row = model_index.loc[(model_index.source == source) & (model_index.target == target) &
                          (model_index.seed == seed) & (model_index.feature_method == "daregram")]
    if len(row) != 1:
        raise ValueError("Missing DARE+Ridge starting model")
    row = row.iloc[0]
    cfg = json.loads((Path(row.checkpoint).parent / "config.json").read_text(encoding="utf-8"))
    source_mean = np.asarray(cfg["source_normalization_mean"], dtype=np.float32)
    source_std = np.asarray(cfg["source_normalization_std"], dtype=np.float32)
    if cfg["source"] != source or cfg["target"] != target or cfg["seed"] != seed:
        raise ValueError("Checkpoint config identity mismatch")
    if cfg["input"] != "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128":
        raise ValueError("Input preprocessing changed")
    raw_s = raw_cache(args.cache_root, source, cache_manifest)
    raw_t = raw_cache(args.cache_root, target, cache_manifest)
    xs, xt = features(row.source_features, source), features(row.target_features, target)
    if sha(row.source_features) != row.source_feature_sha256 or sha(row.target_features) != row.target_feature_sha256:
        raise ValueError("Stored 512-D feature hash changed")
    y_source = base.wear_labels(args.raw_root, source).astype(np.float64)
    initial, old_alpha = initial_model(row, seed, device, xs, xt)
    mean = torch.as_tensor(source_mean, device=device).view(1, 6, 1, 1)
    std = torch.as_tensor(source_std, device=device).view(1, 6, 1, 1)
    teacher = copy.deepcopy(initial).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_pred = infer(teacher, raw_t, mean, std, device)
    old = checked_315(pred_four.loc[(pred_four.source == source) & (pred_four.target == target) &
                                    (pred_four.seed == seed) & (pred_four.method == "D")], "locked D")
    replay_gap = float(np.max(np.abs(teacher_pred - old.pred_vb.to_numpy())))
    if replay_gap > MAX_REPLAY_GAP:
        raise ValueError(f"Teacher D replay differs from locked D: {source}->{target} seed {seed}, gap={replay_gap}")
    pseudo = pseudo_rule(xs, xt, y_source, teacher_pred, row.ridge_model)
    folder.mkdir(parents=True)
    pseudo.to_csv(folder / "pseudo_labels.csv", index=False)
    initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    pseudo_y = torch.from_numpy(teacher_pred.astype(np.float32)).to(device)
    pseudo_w = torch.from_numpy(pseudo.confidence_weight.to_numpy(np.float32)).to(device)
    training_rows, prediction_rows, branch_rows = [], [], []
    for branch in BRANCHES:
        use_pseudo = branch in ("E1", "E3")
        use_temporal = branch in ("E2", "E3")
        base.seed_everything(seed)
        student, _ = base.build_model(seed, device)
        student.load_state_dict(initial_state)
        # eval() retains gradients but fixes every BN running buffer and batch semantics.
        student.eval()
        optimizer = torch.optim.Adam([{"params": student.feature_extractor.parameters(), "lr": LR_ENCODER},
                                      {"params": student.regressor.parameters(), "lr": LR_HEAD}])
        for epoch in range(1, EPOCHS + 1):
            rng = np.random.default_rng(seed + epoch * 1009)
            source_order = rng.permutation(315)
            sums = {"source_mse": 0.0, "pseudo_mse": 0.0, "temporal_mse": 0.0, "total": 0.0}
            for step in range(5):
                source_ids = source_order[step * BATCH:(step + 1) * BATCH]
                target_first = step * BATCH
                target_last = min((step + 1) * BATCH, 315)
                source_x = norm_batch(raw_s, source_ids, mean, std, device)
                target_ids = np.arange(target_first, min(target_last + int(use_temporal and target_last < 315), 315))
                optimizer.zero_grad(set_to_none=True)
                source_pred = student.regressor(student.feature_extractor(source_x)).reshape(-1)
                source_loss = F.mse_loss(source_pred, torch.from_numpy(y_source[source_ids].astype(np.float32)).to(device))
                loss = source_loss
                pseudo_loss = source_loss.new_zeros(())
                temporal_loss = source_loss.new_zeros(())
                if use_pseudo or use_temporal:
                    target_x = norm_batch(raw_t, target_ids, mean, std, device)
                    target_pred = student.regressor(student.feature_extractor(target_x)).reshape(-1)
                    if use_pseudo:
                        ww = pseudo_w[target_first:target_last]
                        if ww.sum() > 0:
                            pseudo_loss = (ww * (target_pred[:BATCH] - pseudo_y[target_first:target_last]).square()).sum() / ww.sum()
                        loss = loss + PSEUDO_WEIGHT * pseudo_loss
                    if use_temporal:
                        # The 64th overlapping point covers the four block boundaries.
                        # Every pair here is exactly (t, t+1) of the same target tool.
                        temporal_loss = F.relu(target_pred[:-1] - target_pred[1:]).square().mean()
                        loss = loss + TEMPORAL_WEIGHT * temporal_loss
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                optimizer.step()
                for name, value in (("source_mse", source_loss), ("pseudo_mse", pseudo_loss),
                                    ("temporal_mse", temporal_loss), ("total", loss)):
                    sums[name] += float(value.item()) / 5
            training_rows.append({"source": source, "target": target, "seed": seed, "branch": branch,
                                  "epoch": epoch, "source_order_sha256": hashlib.sha256(source_order.tobytes()).hexdigest(),
                                  "bn_policy": "all_modules_eval_running_buffers_frozen", **sums})
        p_target, f_target = infer(student, raw_t, mean, std, device, with_features=True)
        _, f_source = infer(student, raw_s, mean, std, device, with_features=True)
        selected_alpha, candidates = ridge_code.select_alpha(f_source, y_source, list(range(5)))
        refit = ridge_code.fit_probe(f_source, y_source, np.arange(315), selected_alpha)
        p_refit = refit.predict(f_target)
        scaler, ridge = refit.steps[0][1], refit.steps[1][1]
        branch_dir = folder / branch
        branch_dir.mkdir()
        checkpoint_path = branch_dir / "student.pth"
        torch.save({"model": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                    "source": source, "target": target, "seed": seed, "branch": branch,
                    "epochs": EPOCHS, "starting_checkpoint_sha256": row.checkpoint_sha256,
                    "starting_ridge_sha256": row.ridge_model_sha256}, checkpoint_path)
        refit_path = branch_dir / "refit_ridge.npz"
        np.savez(refit_path, source_feature_mean=scaler.mean_, source_feature_scale=scaler.scale_,
                 coefficient=ridge.coef_, intercept=np.float64(ridge.intercept_), alpha=np.float64(selected_alpha))
        for cut in CUTS:
            i = cut - 1
            prediction_rows.append({"source": source, "target": target, "seed": seed, "method": branch + "_linear",
                                    "cut_index": cut, "raw_pred_vb": float(p_target[i])})
            prediction_rows.append({"source": source, "target": target, "seed": seed, "method": branch + "_refit_ridge",
                                    "cut_index": cut, "raw_pred_vb": float(p_refit[i])})
        branch_rows.append({"source": source, "target": target, "seed": seed, "branch": branch,
                            "student_checkpoint": str(checkpoint_path.resolve()), "student_sha256": sha(checkpoint_path),
                            "refit_ridge": str(refit_path.resolve()), "refit_ridge_sha256": sha(refit_path),
                            "refit_alpha": selected_alpha, "starting_alpha": old_alpha,
                            "encoder_changed": any(not torch.equal(student.state_dict()[k].cpu(), initial_state[k])
                                                   for k in initial_state if k.startswith("feature_extractor.")),
                            "alpha_candidates_source_rmse": json.dumps(candidates, ensure_ascii=False)})
        del student, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(training_rows).to_csv(folder / "training_log.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(folder / "predictions_unscored.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(folder / "model_index.csv", index=False)
    record = {"status": "complete_before_target_labels", "source": source, "target": target, "seed": seed,
              "starting_checkpoint_sha256": row.checkpoint_sha256, "starting_ridge_sha256": row.ridge_model_sha256,
              "teacher_replay_max_abs_vb": replay_gap, "pseudo_selected_cuts": int(pseudo.selected.sum()),
              "pseudo_selected_late_cuts": int(pseudo.loc[pseudo.cut_index.ge(211), "selected"].sum()),
              "files_sha256": {name: sha(folder / name) for name in ("pseudo_labels.csv", "training_log.csv", "predictions_unscored.csv", "model_index.csv")},
              "branch_files_sha256": {str(p.relative_to(folder)): sha(p) for p in folder.glob("E*/*") if p.is_file()}}
    (folder / "complete.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"{source}->{target} seed={seed} pseudo={record['pseudo_selected_cuts']}/315 late={record['pseudo_selected_late_cuts']}/105 replay={replay_gap:.6g}"


def train(args):
    pred, models = load_locked_four(args.four_root)
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["cuts"] != CUTS:
        raise ValueError("Feature cache cuts changed")
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    config_path = args.out_root / "training_config.json"
    config = {"starting_method": "D", "pairs": PAIRS, "seeds": list(SEEDS), "branches": BRANCHES,
              "epochs": EPOCHS, "batch_size": BATCH, "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD,
              "optimizer": "Adam", "gradient_clip_norm": 10.0, "pseudo_weight": PSEUDO_WEIGHT,
              "temporal_weight": TEMPORAL_WEIGHT, "confidence_rule": "target nearest source 512D standardized distance <= source leave-one-out q90 and teacher VB in source label range",
              "confidence_quantile": CONFIDENCE_QUANTILE, "bn_policy": "student and teacher eval throughout; BN running buffers frozen",
              "target_label_reads": 0, "four_way_lock_sha256": sha(args.four_root / "prediction_lock.json"),
              "cache_manifest_sha256": sha(args.cache_root / "manifest.json"),
              "oor_rule": "existing 48-feature STFT source-fitted OOR-PGA; threshold 3/48, 6/48; source Theil-Sen exponent; first full-gate trigger"}
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
            raise ValueError("Training config differs from existing run")
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for source, target in PAIRS:
        if args.source and source != args.source:
            continue
        if args.target and target != args.target:
            continue
        for seed in SEEDS:
            if args.seed and seed != args.seed:
                continue
            print(run_one(source, target, seed, args, pred, models, manifest, device), flush=True)


def lock_predictions(args):
    out = args.out_root
    if (out / "prediction_lock.json").exists():
        raise FileExistsError("Predictions already locked")
    prior_pred, _ = load_locked_four(args.four_root)
    oor_lock_file = args.oor_root / "prediction_lock_before_target_labels.json"
    oor_lock = json.loads(oor_lock_file.read_text(encoding="utf-8"))
    if sha(args.oor_root / "source_pga_exponents.csv") != oor_lock["source_pga_exponents_sha256"]:
        raise ValueError("Original OOR-PGA source exponent changed")
    if sha(args.oor_root / "triggers_and_factors_full_1_315.csv") != oor_lock["trigger_table_sha256"]:
        raise ValueError("Original OOR-PGA trigger table changed")
    prediction_rows, training_logs, pseudo_rows, model_rows, trigger_rows = [], [], [], [], []
    all_model_hashes = {}
    for source, target in PAIRS:
        score_file = args.oor_root / "oor_scores" / f"{source}_to_{target}_oor_score_full_1_315.csv"
        if sha(score_file) != oor_lock["oor_score_sha256"][score_file.name]:
            raise ValueError(f"Original OOR-PGA score changed: {score_file}")
        oor = checked_315(pd.read_csv(score_file), "OOR score")
        factor = oor.multiplicative_factor.to_numpy(np.float64)
        if not np.isfinite(factor).all() or np.any(factor < 1):
            raise ValueError("Invalid locked OOR-PGA factor")
        trigger_rows.append({"source": source, "target": target, "oor_score_file": str(score_file.resolve()),
                             "oor_score_sha256": sha(score_file), "trigger_cut": str(oor.trigger_cut.iloc[0]),
                             "trigger_events": int(oor.multiplicative_factor.gt(1).any()),
                             "fully_open_cuts": int(oor.fully_open.sum()),
                             "max_factor": float(factor.max()), "pga_parameter_provenance": "source_only_fixed_48_stft_port"})
        for seed in SEEDS:
            folder = out / "runs" / f"{source}_to_{target}" / f"seed_{seed}"
            complete_file = folder / "complete.json"
            if not complete_file.exists():
                raise ValueError(f"Missing completed training run: {folder}")
            complete = json.loads(complete_file.read_text(encoding="utf-8"))
            if complete["status"] != "complete_before_target_labels":
                raise ValueError(f"Training run not label-isolated: {folder}")
            for name, digest in complete["files_sha256"].items():
                if sha(folder / name) != digest:
                    raise ValueError(f"Completed run file changed: {folder / name}")
            for name, digest in complete["branch_files_sha256"].items():
                if sha(folder / name) != digest:
                    raise ValueError(f"Completed model changed: {folder / name}")
                all_model_hashes[str((folder / name).resolve())] = digest
            pseudo = checked_315(pd.read_csv(folder / "pseudo_labels.csv"), "pseudo coverage")
            pseudo.insert(0, "seed", seed)
            pseudo.insert(0, "target", target)
            pseudo.insert(0, "source", source)
            pseudo_rows.append(pseudo)
            log = pd.read_csv(folder / "training_log.csv")
            if len(log) != 4 * EPOCHS or sorted(log.branch.unique()) != list(BRANCHES):
                raise ValueError(f"Training budget incomplete: {folder}")
            training_logs.append(log)
            model_index = pd.read_csv(folder / "model_index.csv")
            if len(model_index) != 4 or sorted(model_index.branch.tolist()) != list(BRANCHES):
                raise ValueError(f"Model branches incomplete: {folder}")
            model_rows.append(model_index)
            new_pred = pd.read_csv(folder / "predictions_unscored.csv")
            for method in [f"{b}_{h}" for b in BRANCHES for h in ("linear", "refit_ridge")]:
                part = checked_315(new_pred.loc[new_pred.method == method], method)
                for cut, value in zip(CUTS, part.raw_pred_vb):
                    i = cut - 1
                    prediction_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                                            "cut_index": cut, "raw_pred_vb": value,
                                            "pga_pred_vb": value * factor[i], "oor_flag_count_48": int(oor.oor_flag_count_48.iloc[i]),
                                            "oor_fully_open": bool(oor.fully_open.iloc[i]),
                                            "pga_factor": factor[i]})
            for method in ("B", "D"):
                part = checked_315(prior_pred.loc[(prior_pred.source == source) & (prior_pred.target == target) &
                                                  (prior_pred.seed == seed) & (prior_pred.method == method)], method)
                for cut, value in zip(CUTS, part.pred_vb):
                    i = cut - 1
                    prediction_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                                            "cut_index": cut, "raw_pred_vb": value,
                                            "pga_pred_vb": value * factor[i], "oor_flag_count_48": int(oor.oor_flag_count_48.iloc[i]),
                                            "oor_fully_open": bool(oor.fully_open.iloc[i]),
                                            "pga_factor": factor[i]})
    pred = pd.DataFrame(prediction_rows).sort_values(["source", "target", "seed", "method", "cut_index"])
    if len(pred) != 30 * 10 * 315 or pred.duplicated(["source", "target", "seed", "method", "cut_index"]).any():
        raise ValueError("Prediction key count or duplicate error")
    for key, part in pred.groupby(["source", "target", "seed", "method"]):
        checked_315(part, str(key))
    pred.to_csv(out / "predictions_unscored.csv", index=False)
    pd.concat(training_logs, ignore_index=True).to_csv(out / "training_log.csv", index=False)
    pd.concat(pseudo_rows, ignore_index=True).to_csv(out / "pseudo_coverage.csv", index=False)
    pd.concat(model_rows, ignore_index=True).to_csv(out / "model_index.csv", index=False)
    pd.DataFrame(trigger_rows).to_csv(out / "oor_provenance.csv", index=False)
    locked_files = ("training_config.json", "predictions_unscored.csv", "training_log.csv", "pseudo_coverage.csv",
                    "model_index.csv", "oor_provenance.csv")
    lock = {"status": "student_models_pseudo_labels_predictions_locked_before_target_scoring",
            "target_role_label_uses_before_lock": 0, "n_prediction_rows": len(pred),
            "four_way_prediction_lock_sha256": sha(args.four_root / "prediction_lock.json"),
            "oor_original_lock_sha256": sha(oor_lock_file),
            "files_sha256": {name: sha(out / name) for name in locked_files},
            "model_files_sha256": all_model_hashes}
    (out / "prediction_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Locked {len(pred)} base-method cut predictions and their OOR-PGA corrections before target labels")


def score(args):
    out = args.out_root
    lock = json.loads((out / "prediction_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "student_models_pseudo_labels_predictions_locked_before_target_scoring" or lock["target_role_label_uses_before_lock"] != 0:
        raise ValueError("Prediction lock invalid")
    for name, digest in lock["files_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Locked file changed: {name}")
    for name, digest in lock["model_files_sha256"].items():
        if sha(name) != digest:
            raise ValueError(f"Locked model changed: {name}")
    if (out / "predictions_per_cut.csv").exists():
        raise FileExistsError("Already scored")
    pred = pd.read_csv(out / "predictions_unscored.csv")
    if len(pred) != lock["n_prediction_rows"] or pred.duplicated(["source", "target", "seed", "method", "cut_index"]).any():
        raise ValueError("Locked prediction keys invalid")
    for key, part in pred.groupby(["source", "target", "seed", "method"]):
        checked_315(part, str(key))
    # First target-role label read is here, after the model/prediction lock.
    truth = {tool: base.wear_labels(args.raw_root, tool).astype(np.float64) for tool in ("c1", "c4", "c6")}
    c1c6 = pred.loc[(pred.source == "c1") & (pred.target == "c6") & (pred.method == "D")]
    old_mae = np.mean([np.mean(np.abs(g.sort_values("cut_index").raw_pred_vb.to_numpy() - truth["c6"]))
                       for _, g in c1c6.groupby("seed")])
    if abs(old_mae - 9.753934) > 0.01:
        raise ValueError(f"Locked old D C1->C6 MAE failed 9.754 replay: {old_mae}")
    pseudo = pd.read_csv(out / "pseudo_coverage.csv")
    if len(pseudo) != 30 * 315 or pseudo.duplicated(["source", "target", "seed", "cut_index"]).any():
        raise ValueError("Pseudo-label coverage mapping invalid")
    for key, part in pseudo.groupby(["source", "target", "seed"]):
        checked_315(part, f"pseudo-{key}")
    pseudo["true_vb_posthoc"] = [truth[t][int(c) - 1] for t, c in zip(pseudo.target, pseudo.cut_index)]
    pseudo["teacher_signed_error_posthoc"] = pseudo.teacher_pseudo_vb - pseudo.true_vb_posthoc
    pseudo.to_csv(out / "pseudo_diagnostic_scored.csv", index=False)
    pseudo_summary = []
    for (source, target, seed), part in pseudo.groupby(["source", "target", "seed"], sort=True):
        late = part.loc[part.cut_index.ge(211)]
        selected_late = late.loc[late.selected]
        pseudo_summary.append({"source": source, "target": target, "seed": seed,
                               "selected_cuts": int(part.selected.sum()), "selected_late_cuts": len(selected_late),
                               "late_teacher_signed_error_posthoc": float(late.teacher_signed_error_posthoc.mean()),
                               "late_teacher_underestimated_cuts_posthoc": int(late.teacher_signed_error_posthoc.lt(0).sum()),
                               "selected_late_teacher_signed_error_posthoc": float(selected_late.teacher_signed_error_posthoc.mean()) if len(selected_late) else np.nan})
    pd.DataFrame(pseudo_summary).to_csv(out / "pseudo_diagnostic_summary.csv", index=False)
    pred["true_vb"] = [truth[t][int(c) - 1] for t, c in zip(pred.target, pred.cut_index)]
    pred["raw_signed_error"] = pred.raw_pred_vb - pred.true_vb
    pred["pga_signed_error"] = pred.pga_pred_vb - pred.true_vb
    pred.to_csv(out / "predictions_per_cut.csv", index=False)
    metric_rows = []
    for (source, target, seed, method), part in pred.groupby(["source", "target", "seed", "method"], sort=True):
        part = checked_315(part, f"{source}-{target}-{seed}-{method}")
        y = part.true_vb.to_numpy()
        for variant, col in (("raw", "raw_pred_vb"), ("pga", "pga_pred_vb")):
            p = part[col].to_numpy()
            full = ridge_code.measure(y, p)
            metric_rows.append({"source": source, "target": target, "seed": seed, "method": method,
                                "variant": variant, "R2": full["R2"], "MAE": full["MAE"], "RMSE": full["RMSE"],
                                "early_MAE": float(np.mean(np.abs(p[:105] - y[:105]))),
                                "middle_MAE": float(np.mean(np.abs(p[105:210] - y[105:210]))),
                                "late_MAE": float(np.mean(np.abs(p[210:] - y[210:]))),
                                "late_signed_error": float(np.mean(p[210:] - y[210:])),
                                "monotonic_violations": int(np.sum(p[:-1] > p[1:])),
                                "oor_trigger_events": int(part.pga_factor.gt(1).any()),
                                "oor_full_gate_cuts": int(part.oor_fully_open.sum())})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "seed_summary.csv", index=False)
    paired = []
    for (source, target, seed, variant), block in metrics.groupby(["source", "target", "seed", "variant"], sort=True):
        index = block.set_index("method")
        for head in ("linear", "refit_ridge"):
            baseline = index.loc[f"E0_{head}"]
            for branch in ("E1", "E2", "E3"):
                candidate = index.loc[f"{branch}_{head}"]
                paired.append({"source": source, "target": target, "seed": seed, "variant": variant,
                               "comparison": f"{branch}_minus_E0_{head}",
                               **{f"delta_{m}": float(candidate[m] - baseline[m]) for m in
                                  ("R2", "MAE", "RMSE", "early_MAE", "middle_MAE", "late_MAE", "late_signed_error", "monotonic_violations")}})
            for branch in BRANCHES:
                candidate = index.loc[f"{branch}_{head}"]
                reference = index.loc["B"]
                paired.append({"source": source, "target": target, "seed": seed, "variant": variant,
                               "comparison": f"{branch}_{head}_minus_B",
                               **{f"delta_{m}": float(candidate[m] - reference[m]) for m in
                                  ("R2", "MAE", "RMSE", "early_MAE", "middle_MAE", "late_MAE", "late_signed_error", "monotonic_violations")}})
    for (source, target, seed, method), block in metrics.groupby(["source", "target", "seed", "method"], sort=True):
        index = block.set_index("variant")
        paired.append({"source": source, "target": target, "seed": seed, "variant": "pga_minus_raw",
                       "comparison": method,
                       **{f"delta_{m}": float(index.loc["pga", m] - index.loc["raw", m]) for m in
                          ("R2", "MAE", "RMSE", "early_MAE", "middle_MAE", "late_MAE", "late_signed_error", "monotonic_violations")}})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired_seed_delta.csv", index=False)
    directions = []
    for (source, target, method, variant), block in metrics.groupby(["source", "target", "method", "variant"], sort=True):
        row = {"source": source, "target": target, "method": method, "variant": variant, "n_seeds": len(block)}
        for col in ("R2", "MAE", "RMSE", "early_MAE", "middle_MAE", "late_MAE", "late_signed_error", "monotonic_violations", "oor_trigger_events", "oor_full_gate_cuts"):
            row[f"{col}_mean"] = float(block[col].mean())
            row[f"{col}_sd"] = float(block[col].std(ddof=1))
        directions.append(row)
    pd.DataFrame(directions).to_csv(out / "direction_summary.csv", index=False)
    paired_directions = []
    for (source, target, variant, comparison), block in paired.groupby(["source", "target", "variant", "comparison"], sort=True):
        row = {"source": source, "target": target, "variant": variant, "comparison": comparison, "n_seeds": len(block)}
        for col in [c for c in block.columns if c.startswith("delta_")]:
            row[f"{col}_mean"] = float(block[col].mean())
            row[f"{col}_sd"] = float(block[col].std(ddof=1))
        row["improved_seeds_MAE"] = int((block.delta_MAE < 0).sum())
        row["improved_seeds_RMSE"] = int((block.delta_RMSE < 0).sum())
        row["improved_seeds_R2"] = int((block.delta_R2 > 0).sum())
        paired_directions.append(row)
    pd.DataFrame(paired_directions).to_csv(out / "paired_direction_summary.csv", index=False)
    plots = out / "plots"
    plots.mkdir()
    palette = {"B": "#888888", "D": "#111111", "E0_linear": "#1f77b4", "E1_linear": "#ff7f0e",
               "E2_linear": "#2ca02c", "E3_linear": "#d62728"}
    for source, target in PAIRS:
        subset = pred.loc[(pred.source == source) & (pred.target == target)]
        fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True, sharey=True)
        for ax, seed in zip(axes, SEEDS):
            block = subset.loc[subset.seed == seed]
            ax.plot(CUTS, truth[target], color="black", linewidth=1.8, label="True VB")
            for method, color in palette.items():
                part = checked_315(block.loc[block.method == method], f"plot-{seed}-{method}")
                ax.plot(CUTS, part.raw_pred_vb, color=color, linewidth=1, label=method)
            ax.set(ylabel=f"seed {seed}\nVB", xlim=(1, 315))
            ax.grid(alpha=.2)
        axes[0].legend(ncol=4, fontsize=8)
        axes[-1].set_xlabel("Cut index")
        fig.suptitle(f"{source.upper()} to {target.upper()}: raw students, D and B")
        fig.tight_layout()
        fig.savefig(plots / f"{source}_to_{target}_all_seeds_raw.png", dpi=140)
        plt.close(fig)
        if target == "c6":
            for seed in SEEDS:
                block = subset.loc[subset.seed == seed]
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(CUTS, truth[target], color="black", linewidth=1.8, label="True VB")
                for method in ("D", "E0_linear", "E1_linear", "E2_linear", "E3_linear"):
                    part = checked_315(block.loc[block.method == method], f"plot-detail-{seed}-{method}")
                    ax.plot(CUTS, part.raw_pred_vb, linewidth=1, label=method)
                for method in ("D", "E3_linear"):
                    part = checked_315(block.loc[block.method == method], f"plot-detail-pga-{seed}-{method}")
                    ax.plot(CUTS, part.pga_pred_vb, linestyle="--", linewidth=1.1, label=method + "+PGA")
                ax.set(xlabel="Cut index", ylabel="VB", xlim=(1, 315), title=f"{source.upper()} to C6 | seed {seed}")
                ax.legend(ncol=4, fontsize=8)
                ax.grid(alpha=.2)
                fig.tight_layout()
                fig.savefig(plots / f"{source}_to_c6_seed_{seed}_with_pga.png", dpi=160)
                plt.close(fig)
    print(f"Scored locked predictions; prior D C1->C6 MAE={old_mae:.6f}; generated 16 plots")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "lock", "score"))
    parser.add_argument("--four-root", type=Path, default=Path("artifacts/frozen_ridge_four_way_20260926"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/five_seed_paired/feature_cache"))
    parser.add_argument("--oor-root", type=Path, default=Path("artifacts/oor_pga_full_1_315_zscore_20260925"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/pseudo_temporal_oor_20260926"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source", choices=("c1", "c4", "c6"))
    parser.add_argument("--target", choices=("c1", "c4", "c6"))
    parser.add_argument("--seed", type=int, choices=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "train":
        train(args)
    elif args.phase == "lock":
        lock_predictions(args)
    elif args.phase == "score":
        score(args)
    else:
        raise NotImplementedError(args.phase)
