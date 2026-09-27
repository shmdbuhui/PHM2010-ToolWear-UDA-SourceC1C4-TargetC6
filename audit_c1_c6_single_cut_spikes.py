"""Read-only, label-free audit of single-cut dips in the frozen C1->C6 run.

Only ``evaluate`` reads C6 wear, after every detection and diagnostic file exists.
The frozen single-window checkpoints and predictions are never modified.
"""

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

import data_sampling as sampling
import run_single_source_pairs as base


ROOT = Path("artifacts/c1_c6_single_cut_spike_audit_20260926")
PRED = Path("artifacts/oor_pga_full_1_315_zscore_20260925/unlabeled_predictions")
CACHE = Path("artifacts/five_seed_paired/feature_cache")
RUN = Path("artifacts/five_seed_paired/c1_to_c6")
FEATURES = Path("artifacts/five_seed_feature_audit_20260925/experiments/c1_to_c6")
RAW = Path(r"E:\QLP\source\source_mill\c6")
SEEDS = range(42, 47)
CHANNELS = sampling.EXPECTED_INPUT_COLS


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def read_predictions() -> dict[int, pd.DataFrame]:
    tables = {}
    for seed in SEEDS:
        path = PRED / f"c1_to_c6_seed_{seed}_full_1_315.csv"
        frame = pd.read_csv(path)
        if frame.cut_index.tolist() != list(range(1, 316)) or not np.isfinite(frame.select_dtypes("number")).all().all():
            raise ValueError(f"Invalid ordered prediction file: {path}")
        original = pd.read_csv(RUN / f"seed_{seed}" / "daregram" / "predictions.csv",
                               usecols=["cut_index", "pred_vb"])
        # Historical C6 evaluation contains only cuts 95..315.
        matched = frame.set_index("cut_index").loc[original.cut_index, "daregram_pred_vb"].to_numpy()
        # Historical suffix inference used different batch boundaries on CUDA.
        # Full-cut CSV is the frozen OOR input; allow its small kernel variation.
        if not np.allclose(matched, original.pred_vb, atol=0.01, rtol=0):
            raise ValueError(f"Frozen original and full-cut predictions differ: {seed}")
        tables[seed] = frame
    return tables


def detect(tables: dict[int, pd.DataFrame], minimum: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed, table in tables.items():
        for stage, column in (("base", "daregram_pred_vb"), ("corrected", "daregram_oor_pga_pred_vb")):
            p = table[column].to_numpy(float)
            for cut in range(2, 315):
                i = cut - 1
                drop, rise = p[i - 1] - p[i], p[i + 1] - p[i]
                rows.append(dict(seed=seed, stage=stage, cut_index=cut,
                                 previous=p[i - 1], current=p[i], next=p[i + 1],
                                 drop_from_previous=drop, rise_to_next=rise,
                                 minimum_two_sided=min(drop, rise),
                                 spike=bool(drop >= minimum and rise >= minimum)))
    detail = pd.DataFrame(rows)
    base_rows = detail[(detail.stage == "base") & detail.spike]
    count = base_rows.groupby("cut_index").seed.nunique()
    summary = pd.DataFrame({"cut_index": count.index, "seed_count": count.values})
    summary["seeds"] = summary.cut_index.map(
        lambda cut: ",".join(str(s) for s in base_rows[base_rows.cut_index == cut].seed))
    summary["shared_all_five"] = summary.seed_count.eq(5)
    return detail, summary


def raw_and_stft(cuts: list[int], cache: np.ndarray, mean: np.ndarray,
                 std: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    raw_rows, image_rows, alternate_rows, waveforms = [], [], [], {}
    for cut in cuts:
        path = RAW / f"c_6_{cut:03d}.csv"
        df = sampling._read_pass_df(str(path))
        full = df[CHANNELS].to_numpy()
        start = len(full) // 2 - 2048
        segment = full[start:start + 4096]
        if segment.shape != (4096, 6):
            raise ValueError(f"Short center window: {path}")
        waveforms[cut] = segment
        for fraction in (0.25, 0.5, 0.75):
            alternate_start = int(len(full) * fraction) - 2048
            alternate = full[alternate_start:alternate_start + 4096]
            alternate_rows.append(dict(cut_index=cut, center_fraction=fraction,
                                       start_0based=alternate_start,
                                       end_exclusive_0based=alternate_start + 4096,
                                       rms_all_six=float(np.sqrt(np.mean(alternate * alternate))),
                                       finite=bool(np.isfinite(alternate).all())))
        regenerated = sampling._stft_crop_and_resize(segment.T)
        stored = np.asarray(cache[cut - 1])
        max_difference = float(np.max(np.abs(regenerated - stored)))
        if max_difference != 0:
            raise ValueError(f"Cached STFT differs from raw center window: cut {cut}, {max_difference}")
        normalized = ((stored - mean[:, None, None]) /
                      (std[:, None, None] + 1e-8)).astype(np.float32)
        for channel, name in enumerate(CHANNELS):
            x = segment[:, channel]
            raw_rows.append(dict(cut_index=cut, raw_file=str(path), raw_sha256=sha(path),
                                 raw_rows=len(full), raw_columns=df.shape[1], channel_index=channel,
                                 channel=name, window_index=0, window_count=1,
                                 requested_start=start, actual_start=start, end_exclusive=start + 4096,
                                 crop_adjustment=0, finite_count=int(np.isfinite(x).sum()),
                                 nan_count=int(np.isnan(x).sum()), inf_count=int(np.isinf(x).sum()),
                                 all_six_zero_rows=int(np.all(segment == 0, axis=1).sum()),
                                 mean=float(np.mean(x)), std=float(np.std(x)),
                                 rms=float(np.sqrt(np.mean(x*x))), minimum=float(np.min(x)),
                                 maximum=float(np.max(x)), p01=float(np.percentile(x, 1)),
                                 p99=float(np.percentile(x, 99)),
                                 repeated_min_count=int((x == x.min()).sum()),
                                 repeated_max_count=int((x == x.max()).sum())))
            for stage, image in (("stft", stored[channel]), ("normalized", normalized[channel])):
                image_rows.append(dict(cut_index=cut, window_index=0, channel_index=channel,
                                       channel=name, stage=stage, shape="128x128", dtype=str(image.dtype),
                                       finite_count=int(np.isfinite(image).sum()),
                                       mean=float(image.mean()), std=float(image.std()),
                                       minimum=float(image.min()), maximum=float(image.max()),
                                       p99=float(np.percentile(image, 99)),
                                       zero_fraction=float(np.mean(image == 0)),
                                       cache_regeneration_max_abs_difference=max_difference))
    return pd.DataFrame(raw_rows), pd.DataFrame(image_rows), pd.DataFrame(alternate_rows), waveforms


def features_and_predictions(cuts: list[int], tables: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, vectors, comparisons = [], [], []
    for seed in SEEDS:
        path = FEATURES / f"seed_{seed}" / "daregram" / "target_features.csv"
        feat = pd.read_csv(path)
        if feat.cut_index.tolist() != list(range(1, 316)):
            raise ValueError(f"Feature order mismatch: {path}")
        columns = [f"feature_{i:03d}" for i in range(512)]
        x = feat[columns].to_numpy(float)
        state = torch.load(RUN / f"seed_{seed}" / "daregram" / "final.pth",
                           map_location="cpu", weights_only=True)["model"]
        weight = state["regressor.0.weight"].numpy().reshape(512).astype(float)
        bias = float(state["regressor.0.bias"].item())
        p = tables[seed].set_index("cut_index")
        for cut in cuts:
            vector = x[cut - 1]
            projected = float(vector @ weight + bias)
            base_pred = float(p.at[cut, "daregram_pred_vb"])
            corrected = float(p.at[cut, "daregram_oor_pga_pred_vb"])
            factor = float(p.at[cut, "multiplicative_factor"])
            correction = corrected - base_pred
            if not np.isclose(corrected, base_pred * factor, atol=1e-9, rtol=0):
                raise ValueError(f"OOR correction mismatch: {seed}, {cut}")
            if not np.isclose(projected, base_pred, atol=5e-4, rtol=0):
                raise ValueError(f"Saved 512D vector / head mismatch: {seed}, {cut}: {projected-base_pred}")
            rows.append(dict(seed=seed, cut_index=cut, window_index=0, window_count=1,
                             aggregation="identity (one centered 4096-sample window)",
                             window_prediction_vb=base_pred, aggregated_before_vb=base_pred,
                             base_prediction_vb=base_pred, oor_factor=factor,
                             expected_oor_delta_from_recorded_factor_vb=base_pred*(factor-1),
                             corrected_minus_base_vb=correction,
                             corrected_prediction_vb=corrected,
                             correction_identity_error=correction-base_pred*(factor-1),
                             head_projection_vb=projected,
                             head_projection_error=projected-base_pred,
                             feature_l2=float(np.linalg.norm(vector)),
                             feature_mean=float(vector.mean()), feature_std=float(vector.std()),
                             regressor_input_equals_resnet_output=True))
            vectors.append(dict(seed=seed, cut_index=cut, window_index=0, stage="resnet18_512_and_regressor_input",
                                **dict(zip(columns, vector))))
        for cut in cuts:
            if cut in (1, 315):
                continue
            v = x[cut - 1]
            mid = (x[cut - 2] + x[cut]) / 2
            comparisons.append(dict(seed=seed, cut_index=cut,
                                    vector_vs_neighbor_mid_l2=float(np.linalg.norm(v-mid)),
                                    head_vs_neighbor_mid_vb=float((v-mid) @ weight),
                                    base_vs_neighbor_mid_vb=float(p.at[cut,"daregram_pred_vb"]-
                                    (p.at[cut-1,"daregram_pred_vb"]+p.at[cut+1,"daregram_pred_vb"])/2)))
    return pd.DataFrame(rows), pd.DataFrame(vectors), pd.DataFrame(comparisons)


def figures(shared: list[int], waveforms: dict[int, np.ndarray], cache: np.ndarray,
            tables: dict[int, pd.DataFrame], out: Path) -> None:
    out.mkdir(exist_ok=True)
    for center in shared:
        cuts = list(range(center-2, center+3))
        fig, axes = plt.subplots(2, 5, figsize=(18, 6), constrained_layout=True)
        for j, cut in enumerate(cuts):
            x = waveforms[cut]
            for ch in range(3):
                axes[0,j].plot(x[:,ch], lw=.35, label=CHANNELS[ch])
            axes[0,j].set_title(f"C6 cut {cut}, center 4096")
            axes[1,j].imshow(cache[cut-1,0], origin="lower", aspect="auto", vmin=0,
                             vmax=float(np.percentile(cache[c-1,0],99.5)) if (c:=center) else None)
            axes[1,j].set_title(f"STFT Fx {cut}")
        axes[0,0].legend(fontsize=7)
        fig.savefig(out / f"raw_stft_cut_{center}.png", dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    for seed, table in tables.items():
        ax.plot(table.cut_index, table.daregram_pred_vb, lw=.8, label=f"base {seed}")
        ax.plot(table.cut_index, table.daregram_oor_pga_pred_vb, lw=.55, alpha=.45)
    for cut in shared:
        ax.axvline(cut, color="gray", lw=.6, alpha=.5)
    ax.set(xlabel="C6 cut", ylabel="Predicted VB", title="Frozen C1 to C6 DARE-GRAM (solid base, faint OOR-PGA)")
    ax.legend(ncol=5, fontsize=7)
    fig.savefig(out / "five_seed_curves_without_labels.png", dpi=160)
    plt.close(fig)


def prepare(minimum: float, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Audit output exists: {out}")
    tables = read_predictions()
    detail, summary = detect(tables, minimum)
    flagged = summary.cut_index.tolist()
    cuts = sorted({c for spike in flagged for c in range(max(1,spike-2),min(315,spike+2)+1)})
    shared = summary.loc[summary.shared_all_five, "cut_index"].tolist()
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    cache_path = CACHE / "c6_stft.npy"
    if sha(cache_path) != manifest["tools"]["c6"]["feature_sha256"]:
        raise ValueError("C6 cache hash mismatch")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    config = json.loads((RUN / "seed_42" / "daregram" / "config.json").read_text(encoding="utf-8"))
    mean = np.asarray(config["source_normalization_mean"], np.float32)
    std = np.asarray(config["source_normalization_std"], np.float32)
    raw_stats, image_stats, alternate_stats, waveforms = raw_and_stft(cuts, cache, mean, std)
    predictions, vectors, comparisons = features_and_predictions(cuts, tables)
    out.mkdir(parents=True)
    save(detail, out / "spike_detection_all_cuts.csv")
    save(summary, out / "spike_summary.csv")
    save(raw_stats, out / "raw_window_channel_stats.csv")
    save(image_stats, out / "stft_normalized_channel_stats.csv")
    save(alternate_stats, out / "alternate_window_rms.csv")
    save(predictions, out / "window_prediction_trace.csv")
    save(vectors, out / "resnet_and_regressor_512.csv")
    save(comparisons, out / "feature_neighbor_comparison.csv")
    (out / "audit_manifest.json").write_text(json.dumps({
        "threshold_vb_each_side": minimum, "threshold_selected_without_c6_labels": True,
        "detected_cuts": flagged, "all_five_seed_cuts": shared,
        "audited_cuts_with_two_neighbors": cuts,
        "prediction_files_sha256": {str(s):sha(PRED/f"c1_to_c6_seed_{s}_full_1_315.csv") for s in SEEDS},
        "c6_stft_cache_sha256": sha(cache_path), "raw_signal_directory": str(RAW),
        "source_only_normalization": {"mean":mean.tolist(), "std":std.tolist()},
        "c6_target_wear_read": False, "training_performed": False,
        "checkpoint_modified": False, "aggregation": "one center window per cut; identity"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(shared, waveforms, cache, tables, out / "figures")
    print(f"Audited {len(flagged)} unique spike cuts, {len(shared)} common to all seeds, {len(cuts)} raw windows: {out}")


def evaluate(out: Path) -> None:
    if not (out / "audit_manifest.json").is_file():
        raise ValueError("Run prepare first; target labels are evaluation-only")
    from run_oor_pga_full_1_315 import metrics
    truth = base.wear_labels(RAW.parent, "c6").astype(float)
    rows = []
    fig, ax = plt.subplots(figsize=(15,5), constrained_layout=True)
    ax.plot(range(1,316), truth, color="black", lw=1.2, label="C6 true VB (evaluation only)")
    for seed, table in read_predictions().items():
        for stage, col in (("before", "daregram_pred_vb"), ("after", "daregram_oor_pga_pred_vb")):
            pred = table[col].to_numpy(float)
            rows.append(dict(seed=seed, stage=stage, **metrics(truth,pred)))
            ax.plot(range(1,316), pred, lw=.7, alpha=.6 if stage=="before" else .35,
                    label=f"{seed} {stage}")
    save(pd.DataFrame(rows), out / "evaluation_metrics_before_after_oor.csv")
    replay_folder = out / "replayed_unlabeled_predictions"
    if replay_folder.is_dir():
        replay_rows = []
        for seed in SEEDS:
            frame = pd.read_csv(replay_folder / f"seed_{seed}.csv")
            if frame.cut_index.tolist() != list(range(1,316)):
                raise ValueError(f"Replay cut order mismatch: {seed}")
            for stage, column in (("before", "base_prediction_vb"),
                                  ("after", "corrected_prediction_vb")):
                replay_rows.append(dict(seed=seed, stage=stage,
                                        **metrics(truth,frame[column].to_numpy(float))))
        save(pd.DataFrame(replay_rows), out / "replay_evaluation_metrics.csv")
    ax.set(xlabel="C6 cut", ylabel="VB", title="Frozen C1 to C6 DARE-GRAM and OOR-PGA")
    ax.legend(ncol=6, fontsize=6)
    fig.savefig(out / "figures" / "evaluation_curves_before_after_oor.png", dpi=160)
    plt.close(fig)


def replay(out: Path) -> None:
    """Re-run frozen inference on all 315 cuts, preserving original predictions."""
    import run_five_seed_pairs as five
    if not (out / "audit_manifest.json").is_file():
        raise ValueError("Run prepare first")
    folder = out / "replayed_unlabeled_predictions"
    if folder.exists():
        raise FileExistsError(folder)
    raw_target = np.load(CACHE / "c6_stft.npy", mmap_mode="r", allow_pickle=False)
    config = json.loads((RUN / "seed_42" / "daregram" / "config.json").read_text(encoding="utf-8"))
    mean = np.asarray(config["source_normalization_mean"], np.float32)
    std = np.asarray(config["source_normalization_std"], np.float32)
    target = ((raw_target - mean[None,:,None,None]) /
              (std[None,:,None,None] + 1e-8)).astype(np.float32)
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite normalized target input")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    folder.mkdir()
    for seed in SEEDS:
        method_config = json.loads((RUN / f"seed_{seed}" / "daregram" / "config.json").read_text(encoding="utf-8"))
        if (not np.array_equal(np.asarray(method_config["source_normalization_mean"],np.float32),mean)
            or not np.array_equal(np.asarray(method_config["source_normalization_std"],np.float32),std)
            or method_config["actual_unlabeled_target_cuts"] != list(range(1,316))
            or method_config["target_labels_training_reads"] != 0):
            raise ValueError(f"Protocol mismatch: seed {seed}")
        original = pd.read_csv(PRED / f"c1_to_c6_seed_{seed}_full_1_315.csv")
        base_pred = five.predict("daregram", target, list(range(1,316)), seed,
                                 RUN / f"seed_{seed}", device)
        factor = original.multiplicative_factor.to_numpy(float)
        corrected = base_pred * factor
        base_difference = float(np.max(np.abs(base_pred-original.daregram_pred_vb.to_numpy(float))))
        corrected_difference = float(np.max(np.abs(corrected-original.daregram_oor_pga_pred_vb.to_numpy(float))))
        if base_difference > 0.01 or corrected_difference > 0.02:
            raise ValueError(f"Replay differs from frozen output: {seed}: {base_difference}, {corrected_difference}")
        path = folder / f"seed_{seed}.csv"
        save(pd.DataFrame({"cut_index":range(1,316), "base_prediction_vb":base_pred,
                           "oor_factor":factor, "oor_delta_vb":corrected-base_pred,
                           "corrected_prediction_vb":corrected}), path)
        records.append(dict(seed=seed, checkpoint_sha256=sha(RUN/f"seed_{seed}"/"daregram"/"final.pth"),
                            replay_device=str(device), max_abs_base_difference_from_frozen=base_difference,
                            max_abs_corrected_difference_from_frozen=corrected_difference,
                            replay_prediction_sha256=sha(path), target_label_reads=0))
    save(pd.DataFrame(records), out / "replay_provenance.csv")
    print(f"Replayed frozen inference for seeds 42–46 on all 315 cuts: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "replay", "evaluate"))
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--minimum-vb", type=float, default=5.0)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.minimum_vb, args.out)
    elif args.mode == "replay":
        replay(args.out)
    else:
        evaluate(args.out)
