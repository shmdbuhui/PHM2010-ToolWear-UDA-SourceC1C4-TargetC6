"""Frozen-protocol STFT amplitude ablation for C1->C4, then C1->C6.

A is the audited original five-seed training. B is an independently trained
identity control: the original path already applies log1p to STFT magnitude.
C divides each cut's nonnegative STFT magnitudes by their common RMS before
the original log1p, resize and C1-channel Z-score.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.ndimage import zoom

import data_sampling
import diagnose_c1_c6_distribution as diag
import run_five_seed_pairs as five
import run_single_source_pairs as base


SEEDS = tuple(range(42, 47))
METHODS = ("source_only", "daregram")
SCHEMES = ("A", "B", "C")
STAGES = ((1, 105), (106, 210), (211, 315))
EPS = 1e-8
BASELINE = Path("artifacts/five_seed_paired")
FULL_C6 = Path("artifacts/norm_comparison_20260925/zscore")
FEATURE_AUDIT = Path("artifacts/five_seed_feature_audit_20260925")


def jwrite(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jread(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def check_baseline(root, tool):
    manifest = jread(root / "feature_cache/manifest.json")
    path = root / f"feature_cache/{tool}_stft.npy"
    if (manifest["cuts"] != base.ALL_CUTS or
        manifest["feature_shape"] != [315, 6, 128, 128] or
        manifest["stft_code_sha256"] != base.file_hash(Path(data_sampling.__file__)) or
        base.file_hash(path) != manifest["tools"][tool]["feature_sha256"]):
        raise ValueError(f"Original STFT cache provenance failed: {tool}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (315, 6, 128, 128) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"Invalid original STFT cache: {path}")
    return x, manifest


def build_energy_cache(raw_root, tool, path, original):
    """Reproduce original STFT exactly; alter only magnitude before log1p."""
    if path.exists():
        raise FileExistsError(path)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(315, 6, 128, 128))
    rms_rows = []
    for i, signal_path in enumerate(base.signal_files(raw_root, tool)):
        frame = data_sampling._read_pass_df(str(signal_path))
        crop = data_sampling._crop_fixed_center(frame, data_sampling.EXPECTED_INPUT_COLS, 4096)
        if crop is None or crop.shape != (6, 4096):
            raise ValueError(f"Bad center crop: {signal_path}")
        magnitudes = []
        for channel in crop:
            _, _, z = signal.stft(channel, fs=50_000.0, window="hann",
                                  nperseg=256, noverlap=224, detrend=False,
                                  boundary=None, padded=False)
            magnitudes.append(np.abs(z))
        mag = np.stack(magnitudes)
        if mag.shape != (6, 129, 121) or not np.isfinite(mag).all() or np.any(mag < 0):
            raise ValueError(f"Bad magnitude STFT: {signal_path}")
        # Check the independent STFT calculation before any new treatment.
        a = np.stack([zoom(np.log1p(m), (128/129, 128/121), order=1).astype(np.float32)
                      for m in mag])
        if not np.array_equal(a, original[i]):
            raise ValueError(f"A regeneration differs from original cache: {tool} cut {i+1}")
        energy = float(np.sqrt(np.mean(np.square(mag, dtype=np.float64))))
        divisor = max(energy, EPS)
        output[i] = np.stack([zoom(np.log1p(m/divisor), (128/129, 128/121), order=1).astype(np.float32)
                              for m in mag])
        rms_rows.append({"tool": tool, "cut_index": i+1, "magnitude_rms": energy,
                         "divisor": divisor, "eps": EPS})
        if (i+1) % 50 == 0 or i == 314:
            print(f"C energy STFT {tool}: {i+1}/315", flush=True)
    output.flush()
    del output
    return rms_rows


def normalized_inputs(root, tool, scheme, mean=None, std=None):
    file = (root / f"cache_C/{tool}_stft.npy" if scheme == "C" else
            BASELINE / f"feature_cache/{tool}_stft.npy")
    raw = np.load(file, mmap_mode="r", allow_pickle=False)
    if mean is None:
        mean = raw.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
    x = ((raw - mean[None, :, None, None]) /
         (std[None, :, None, None] + 1e-8)).astype(np.float32)
    return x, mean, std


def A_source(target, seed, method):
    folder = BASELINE / f"c1_to_{target}/seed_{seed}/{method}"
    pred = (FULL_C6 / f"c1_to_c6/seed_{seed}/{method}/predictions.csv"
            if target == "c6" else folder / "predictions.csv")
    return folder / "final.pth", folder / "config.json", pred


def model_features(checkpoint, seed, x, device):
    model, _ = base.build_model(seed, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["epoch"] != 50:
        raise ValueError(f"Unexpected checkpoint epoch: {checkpoint}")
    model.load_state_dict(state["model"])
    model.eval()
    features, predictions = [], []
    with torch.no_grad():
        for start in range(0, 315, 63):
            z = model.feature_extractor(torch.from_numpy(x[start:start+63]).to(device))
            features.append(z.cpu().numpy())
            predictions.append(model.regressor(z).cpu().numpy().reshape(-1))
    del model
    return np.concatenate(features), np.concatenate(predictions).astype(np.float64)


def check_A(target, seed, method, x_target, device, mean, std):
    checkpoint, config_path, prediction_path = A_source(target, seed, method)
    cfg = jread(config_path)
    if (cfg["source"] != "c1" or cfg["target"] != target or cfg["seed"] != seed or
        cfg["method"] != method or cfg["epochs"] != 50 or cfg["batch_size"] != 63 or
        cfg["lr"] != .001 or cfg["regressor"] != "Linear(512, 1)" or
        cfg["target_unlabeled_cuts"] != base.ALL_CUTS or
        cfg["actual_unlabeled_target_cuts"] != (base.ALL_CUTS if method == "daregram" else []) or
        cfg["target_labels_training_reads"] != 0):
        raise ValueError(f"A model protocol mismatch: {config_path}")
    if (not np.array_equal(np.asarray(cfg["source_normalization_mean"], dtype=np.float32), mean) or
        not np.array_equal(np.asarray(cfg["source_normalization_std"], dtype=np.float32), std) or
        cfg["source_feature_sha256"] != jread(BASELINE / "feature_cache/manifest.json")["tools"]["c1"]["feature_sha256"] or
        cfg["target_feature_sha256"] != jread(BASELINE / "feature_cache/manifest.json")["tools"][target]["feature_sha256"] or
        base.file_hash(checkpoint) != jread(BASELINE / f"c1_to_{target}/seed_{seed}/audit.json")["checkpoint_sha256"][method]):
        raise ValueError(f"A normalization/cache/checkpoint mismatch: {config_path}")
    # The historical file contains true_vb; deliberately read prediction columns only.
    old = pd.read_csv(prediction_path, usecols=["cut_index", "pred_vb"])
    if list(old.columns) != ["cut_index", "pred_vb"] or old.cut_index.tolist() != base.ALL_CUTS:
        raise ValueError(f"A prediction scope mismatch: {prediction_path}")
    feat, pred = model_features(checkpoint, seed, x_target, device)
    delta = float(np.max(np.abs(pred-old.pred_vb.to_numpy(np.float64))))
    if delta > 1e-4:
        raise ValueError(f"A replay failed, max prediction delta {delta}: {prediction_path}")
    return feat, pred, {"checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": base.file_hash(checkpoint),
                        "reference_prediction": str(prediction_path.resolve()),
                        "reference_prediction_sha256": base.file_hash(prediction_path),
                        "max_replay_delta_vb": delta,
                        "source_order_sha256_all_epochs": cfg["source_order_sha256_all_epochs"]}


def stage_metrics(y, pred):
    rows = []
    for lo, hi in STAGES:
        p, t = pred[lo-1:hi], y[lo-1:hi]
        rows.append({"stage": f"{lo}-{hi}", "MAE": float(np.mean(np.abs(p-t))),
                     "signed_error": float(np.mean(p-t)),
                     "pred_increment": float(p[-1]-p[0]),
                     "true_increment": float(t[-1]-t[0]),
                     "pred_max": float(np.max(p))})
    return rows


def metric_rows(target, scheme, seed, method, truth, pred):
    full = base.metrics(truth, pred)
    rows = [{"target": target, "scheme": scheme, "seed": seed, "method": method,
             "scope": "full_1_315", "stage": "all", "n": 315,
             "signed_error": float(np.mean(pred-truth)),
             "pred_increment": float(pred[-1]-pred[0]),
             "true_increment": float(truth[-1]-truth[0]), "pred_max": float(np.max(pred)), **full}]
    for item in stage_metrics(truth, pred):
        rows.append({"target": target, "scheme": scheme, "seed": seed, "method": method,
                     "scope": "stage", "stage": item["stage"], "n": 105,
                     "R2": np.nan, "RMSE": np.nan, "MAPE_percent": np.nan, **item})
    return rows


def plot_target(folder, target, results):
    # One common axis range over all three schemes and both methods.
    truth = results["A", 42, "source_only"]["truth"]
    curves = {}
    bounds = [truth.min(), truth.max()]
    for scheme in SCHEMES:
        for method in METHODS:
            values = np.stack([results[scheme, seed, method]["prediction"] for seed in SEEDS])
            curves[scheme, method] = (values.mean(0), values.std(0, ddof=1))
            bounds.extend([float((values.mean(0)-values.std(0, ddof=1)).min()),
                           float((values.mean(0)+values.std(0, ddof=1)).max())])
    pad = (max(bounds)-min(bounds))*.05
    for scheme in SCHEMES:
        fig, ax = plt.subplots(figsize=(11.5, 5.1), constrained_layout=True)
        ax.plot(base.ALL_CUTS, truth, color="black", lw=2, label="True VB")
        for method, color, label in (("source_only", "#2563a6", "Source-only"),
                                      ("daregram", "#db7026", "DARE-GRAM")):
            mean, sd = curves[scheme, method]
            ax.fill_between(base.ALL_CUTS, mean-sd, mean+sd, color=color, alpha=.16,
                            label=f"{label} ±1 seed SD")
            ax.plot(base.ALL_CUTS, mean, color=color, lw=1.7, label=f"{label} mean")
        ax.set(xlim=(1, 315), ylim=(min(bounds)-pad, max(bounds)+pad),
               xlabel=f"{target.upper()} Cut", ylabel="VB", title=f"C1→{target.upper()} | scheme {scheme}")
        ax.grid(alpha=.22)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        fig.savefig(folder / f"{target}_{scheme}_curves.png", dpi=170)
        plt.close(fig)


def distance_summary(source, target, scheme, seed, method, layer):
    stages = np.asarray([next(f"{lo}-{hi}" for lo, hi in STAGES if lo <= cut <= hi)
                         for cut in base.ALL_CUTS])
    dist, _ = diag.distance_and_oor(source, target, stages, seed, method, layer)
    frame = pd.DataFrame(dist)
    rows = []
    lo, hi = np.quantile(source, [.01, .99], axis=0)
    for stage in sorted(frame.stage.unique()):
        s = frame[(frame.stage == stage) & (frame.domain == "c1_holdout")].distance.to_numpy()
        t = frame[(frame.stage == stage) & (frame.domain == "c6")].distance.to_numpy()
        a, b = (int(v) for v in stage.split("-"))
        outside = np.mean((target[a-1:b] < lo) | (target[a-1:b] > hi))
        rows.append({"scheme": scheme, "seed": seed, "method": method, "layer": layer,
                     "stage": stage, "c1_holdout_nn_median": float(np.median(s)),
                     "target_nn_median": float(np.median(t)),
                     "target_over_c1_holdout_median": float(np.median(t)/np.median(s)),
                     "target_above_c1_holdout_p95_fraction": float(np.mean(t > np.quantile(s, .95))),
                     "target_outside_c1_full_01_99_fraction": float(outside)})
    return rows


def phase(root, target, device):
    if target == "c6" and not (root / "c4_selection_lock.json").is_file():
        raise RuntimeError("Run and lock C1->C4 before reading C6")
    if target == "c6":
        lock = jread(root / "c4_selection_lock.json")
        if lock["status"] != "fixed_before_c6" or lock["c4_metrics_sha256"] != base.file_hash(root / "c4_metrics.csv"):
            raise ValueError("C4 selection lock changed")
    folder = root / f"c1_to_{target}"
    folder.mkdir(parents=True, exist_ok=True)
    raw_root = Path(r"E:\QLP\source\source_mill")
    original_c1, manifest = check_baseline(BASELINE, "c1")
    original_target, other_manifest = check_baseline(BASELINE, target)
    if manifest != other_manifest:
        raise ValueError("Original cache manifests disagree")
    cache_folder = root / "cache_C"
    cache_folder.mkdir(exist_ok=True)
    energy_log = []
    for tool, original in (("c1", original_c1), (target, original_target)):
        cache_path = cache_folder / f"{tool}_stft.npy"
        if cache_path.exists():
            data = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if data.shape != (315, 6, 128, 128) or not np.isfinite(data).all():
                raise ValueError(f"Bad C cache: {cache_path}")
            del data
        else:
            energy_log.extend(build_energy_cache(raw_root, tool, cache_path, original))
    if energy_log:
        record_csv(cache_folder / f"{target}_energy_divisors.csv", energy_log)
    source_labels = base.wear_labels(raw_root, "c1")
    results, metrics, distances, audit_rows = {}, [], [], []
    stft_desc = {}
    for scheme in ("A", "C"):
        x_source, mean, std = normalized_inputs(root, "c1", scheme)
        x_target, _, _ = normalized_inputs(root, target, scheme, mean, std)
        if scheme == "A" and (not np.array_equal(x_source,
                                                  normalized_inputs(root, "c1", "B")[0])):
            raise ValueError("B identity input differs from A")
        stft_desc[scheme] = (diag.stft_descriptors(x_source), diag.stft_descriptors(x_target))
        distances.extend(distance_summary(*stft_desc[scheme], scheme, "shared", "shared", "stft_48d"))
        jwrite(folder / f"{scheme}_input_config.json", {
            "scheme": scheme, "source": "c1", "target": target,
            "source_channel_mean": mean.tolist(), "source_channel_std": std.tolist(),
            "normalization": "float32((image - C1 mean)/(C1 std + 1e-8))",
            "stft_image_cache": {"source": str((BASELINE / "feature_cache/c1_stft.npy" if scheme == "A" else
                                                  cache_folder / "c1_stft.npy").resolve()),
                                 "target": str((BASELINE / f"feature_cache/{target}_stft.npy" if scheme == "A" else
                                                  cache_folder / f"{target}_stft.npy").resolve())},
            "energy_formula": "none" if scheme == "A" else
            "M=abs(STFT); r=sqrt(mean(M**2 over all 6*129*121 entries of one Cut)); M'=M/max(r,1e-8); log1p(M'); zoom(order=1) to 128x128",
            "eps": EPS if scheme == "C" else None,
            "source_cache_sha256": manifest["tools"]["c1"]["feature_sha256"] if scheme == "A" else
            base.file_hash(cache_folder / "c1_stft.npy"),
            "target_cache_sha256": manifest["tools"][target]["feature_sha256"] if scheme == "A" else
            base.file_hash(cache_folder / f"{target}_stft.npy")})
        # A first: replay every frozen model against its full-cut historical predictions.
        if scheme == "A":
            for seed in SEEDS:
                for method in METHODS:
                    _, config_path, prediction_path = A_source(target, seed, method)
                    if not config_path.is_file() or not prediction_path.is_file():
                        raise FileNotFoundError(config_path)
                    features, pred, audit = check_A(target, seed, method, x_target, device, mean, std)
                    source_feature, _ = model_features(A_source(target, seed, method)[0], seed, x_source, device)
                    results["A", seed, method] = {"prediction": pred,
                                                   "source_feature": source_feature, "target_feature": features}
                    audit_rows.append({"scheme": "A", "seed": seed, "method": method, **audit})
                    print(f"A replay {target} seed {seed} {method}: max Δ={audit['max_replay_delta_vb']:.7g}", flush=True)
        else:
            # C is trained after A replay; B gets exactly A's normalized inputs below.
            pass
        del x_source, x_target
    # B and C are independent training runs. B is deliberately identical to A's input.
    for scheme in ("B", "C"):
        source_x, mean, std = normalized_inputs(root, "c1", scheme)
        target_x, _, _ = normalized_inputs(root, target, scheme, mean, std)
        if scheme == "B" and (not np.array_equal(stft_desc["A"][0], diag.stft_descriptors(source_x)) or
                              not np.array_equal(stft_desc["A"][1], diag.stft_descriptors(target_x))):
            raise ValueError("B descriptors differ from A")
        for seed in SEEDS:
            pair_dir = folder / scheme / f"seed_{seed}"
            pair_dir.mkdir(parents=True, exist_ok=False)
            for method in METHODS:
                (pair_dir / method).mkdir()
                trace = five.train(method, source_x, source_labels,
                                   target_x if method == "daregram" else None,
                                   "c1", target, seed, pair_dir, device, trend_lambda=0.0)
                a_config = jread(A_source(target, seed, method)[1])
                if (trace["initial_model_sha256"] != a_config["initial_model_sha256"] or
                    trace["source_order_sha256_by_epoch"] != a_config["source_order_sha256_by_epoch"] or
                    trace["actual_unlabeled_target_cuts"] !=
                    (base.ALL_CUTS if method == "daregram" else [])):
                    raise ValueError(f"Training protocol/order diverged: {target} {scheme} {seed} {method}")
                ck = pair_dir / method / "final.pth"
                feat_t, pred = model_features(ck, seed, target_x, device)
                feat_s, _ = model_features(ck, seed, source_x, device)
                if scheme == "B":
                    delta = float(np.max(np.abs(pred-results["A", seed, method]["prediction"])))
                    if delta > 1e-4:
                        raise ValueError(f"B identity retrain differs from A: {target} {seed} {method} Δ={delta}")
                results[scheme, seed, method] = {"prediction": pred,
                                                 "source_feature": feat_s, "target_feature": feat_t}
                audit_rows.append({"scheme": scheme, "seed": seed, "method": method,
                                   "checkpoint": str(ck.resolve()),
                                   "checkpoint_sha256": base.file_hash(ck),
                                   "source_order_sha256_all_epochs": trace["source_order_sha256_all_epochs"],
                                   "A_checkpoint_sha256": base.file_hash(A_source(target, seed, method)[0])})
                jwrite(pair_dir / method / "config.json", {
                    "source": "c1", "target": target, "scheme": scheme, "method": method,
                    "seed": seed, "epochs": 50, "batch_size": 63, "lr": .001,
                    "backbone": "ResNet18", "regressor": "Linear(512,1)",
                    "checkpoint_selection": "final epoch", "target_labels_training_reads": 0,
                    "actual_unlabeled_target_cuts": base.ALL_CUTS if method == "daregram" else [],
                    "source_order_sha256_by_epoch": trace["source_order_sha256_by_epoch"],
                    "initial_model_sha256": trace["initial_model_sha256"],
                    "checkpoint_sha256": base.file_hash(ck),
                    "input_config": str((folder / f"{'A' if scheme == 'B' else 'C'}_input_config.json").resolve()),
                    "B_is_A_identity_control": scheme == "B"})
                print(f"Trained {target} {scheme} seed {seed} {method}", flush=True)
        del source_x, target_x
    # Target wear is used only for finished-run evaluation. All train decisions are fixed.
    truth = base.wear_labels(raw_root, target)
    if target == "c6" and not (root / "c4_selection_lock.json").is_file():
        raise RuntimeError("C4 lock missing before C6 evaluation")
    for scheme in SCHEMES:
        source_x, mean, std = normalized_inputs(root, "c1", scheme)
        target_x, _, _ = normalized_inputs(root, target, scheme, mean, std)
        for seed in SEEDS:
            for method in METHODS:
                item = results[scheme, seed, method]
                pred = item["prediction"]
                item["truth"] = truth
                if scheme == "A":
                    ref = pd.read_csv(A_source(target, seed, method)[2])
                    if not np.array_equal(ref.true_vb.to_numpy(np.float32), truth):
                        raise ValueError(f"Archived A target label mismatch: {target} {seed} {method}")
                predictions = folder / scheme / f"seed_{seed}" / method
                predictions.mkdir(parents=True, exist_ok=True)
                record_csv(predictions / "predictions.csv", [{"cut_index": cut, "true_vb": truth[cut-1],
                                                               "pred_vb": pred[cut-1]}
                                                              for cut in base.ALL_CUTS])
                metrics.extend(metric_rows(target, scheme, seed, method, truth, pred))
                distances.extend(distance_summary(item["source_feature"], item["target_feature"],
                                                  scheme, seed, method, "resnet_512d"))
                del item["source_feature"], item["target_feature"]
        del source_x, target_x
    record_csv(root / f"{target}_metrics.csv", metrics)
    record_csv(root / f"{target}_distances.csv", distances)
    record_csv(root / f"{target}_checkpoint_audit.csv", audit_rows)
    summary = pd.DataFrame(metrics).groupby(["target", "scheme", "method", "stage"], as_index=False).agg(
        n_seeds=("seed", "count"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        signed_error_mean=("signed_error", "mean"),
        pred_increment_mean=("pred_increment", "mean"),
        true_increment_mean=("true_increment", "mean"),
        pred_max_mean=("pred_max", "mean"))
    summary.to_csv(root / f"{target}_summary.csv", index=False, float_format="%.17g")
    plot_target(folder, target, results)
    print(summary[["scheme", "method", "stage", "MAE_mean", "signed_error_mean", "pred_increment_mean"]].to_string(index=False), flush=True)
    if target == "c4":
        # Rule and numerical tolerance were fixed in the script before C4 evaluation.
        c4 = summary
        def v(scheme, method, stage, metric):
            return float(c4[(c4.scheme == scheme) & (c4.method == method) & (c4.stage == stage)][metric].iloc[0])
        eligible = all(v("C", m, "all", "MAE_mean") <= v("A", m, "all", "MAE_mean")+1.0 and
                       v("C", m, "211-315", "MAE_mean") <= v("A", m, "211-315", "MAE_mean")+1.0
                       for m in METHODS)
        preferred = "C" if eligible and sum(v("C", m, "all", "MAE_mean") for m in METHODS) < sum(
            v("A", m, "all", "MAE_mean") for m in METHODS) else "A"
        jwrite(root / "c4_selection_lock.json", {
            "status": "fixed_before_c6", "selected_for_recommendation": preferred,
            "B_status": "mathematically identical to A; separately retrained as identity control",
            "C_eligible": eligible,
            "rule": "C requires each method's C4 full and late five-seed mean MAE <= A+1.0 VB; choose C only if eligible and sum of two full MAEs is lower; B is identical to A",
            "c4_metrics_sha256": base.file_hash(root / "c4_metrics.csv"),
            "c6_label_reads_before_lock": 0,
            "c6_policy": "run all A/B/C as predeclared diagnostic; recommendation fixed from C4 only"})
        print(f"C4 scheme locked before C6: {preferred}; C eligible={eligible}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("c4", "c6"), required=True)
    parser.add_argument("--out-root", type=Path,
                        default=Path("artifacts/stft_amplitude_ablation_20260926"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.out_root.resolve()
    if args.phase == "c4" and root.exists():
        parser.error("C4 phase requires a new output directory")
    if args.phase == "c6" and not root.is_dir():
        parser.error("C6 phase requires completed C4 root")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=root / f"{args.phase}.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    phase(root, args.phase, torch.device(args.device))


if __name__ == "__main__":
    main()
