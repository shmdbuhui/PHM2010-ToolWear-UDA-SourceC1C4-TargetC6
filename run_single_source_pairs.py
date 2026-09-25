"""Paired single-source source-only and transductive DARE-GRAM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

import data_sampling as sampling
from models.DAREGRAM import Trainset as DAREGRAM
from networks.resnet import ResNet18


TOOLS = ("c1", "c4", "c6")
ALL_CUTS = list(range(1, 316))
METRICS = ("MAE", "RMSE", "R2", "MAPE_percent")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--source", choices=TOOLS)
    selection.add_argument("--all", action="store_true", help="Run all six directed pairs; reuse completed matching pairs")
    parser.add_argument("--target", choices=TOOLS)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--split-record", type=Path, default=Path(r"E:\QLP\test-9.3\outputs\predictions_comparison.csv"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/single_source_pairs_seed42"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=63, help="Must divide 315, so every cut is used each epoch")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.all and args.target is not None:
        parser.error("--target cannot be combined with --all")
    if not args.all and args.target is None:
        parser.error("--source requires --target")
    if args.source == args.target and not args.all:
        parser.error("--source and --target must be different tools")
    if args.epochs < 1 or args.batch_size < 2 or 315 % args.batch_size:
        parser.error("--epochs must be positive and --batch-size must divide 315")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def signal_files(root, tool):
    folder = root / tool
    files = sorted(folder.glob(f"c_{tool[1:]}_*.csv"), key=lambda p: sampling._extract_pass_idx(str(p)))
    cuts = [sampling._extract_pass_idx(str(path)) for path in files]
    if cuts != ALL_CUTS:
        raise ValueError(f"{tool}: expected one raw signal for each cut 1..315; found {cuts}")
    return files


def stft_features(root, tool):
    """Read signal CSVs only; this function has no path to a wear file or NPZ labels."""
    images = []
    for index, path in enumerate(signal_files(root, tool), 1):
        signal = sampling._read_pass_df(str(path))
        crop = sampling._crop_fixed_center(signal, sampling.EXPECTED_INPUT_COLS, 4096)
        if crop is None:
            raise ValueError(f"Cannot center-crop {path}")
        images.append(sampling._stft_crop_and_resize(crop))
        if index % 50 == 0 or index == 315:
            logging.info("STFT %s: %d/315", tool.upper(), index)
    features = np.stack(images)
    if features.shape != (315, 6, 128, 128) or not np.isfinite(features).all():
        raise ValueError(f"{tool}: invalid STFT tensor {features.shape}")
    return features


def wear_labels(root, tool, evaluation_dir=None):
    """Gate target labels on both final checkpoints being present."""
    if evaluation_dir is not None and not all(
        (evaluation_dir / method / "final.pth").is_file() for method in ("source_only", "daregram")
    ):
        raise RuntimeError("Target labels cannot be read before both final checkpoints")
    path = root / f"{tool}_wear.csv"
    frame = pd.read_csv(path)
    if list(frame.columns) != ["cut", "flute_1", "flute_2", "flute_3"]:
        raise ValueError(f"Unexpected wear schema: {path}")
    if frame["cut"].astype(int).tolist() != ALL_CUTS:
        raise ValueError(f"{tool}: wear rows must follow cuts 1..315")
    labels = frame[["flute_1", "flute_2", "flute_3"]].mean(axis=1).to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all():
        raise ValueError(f"{tool}: nonfinite wear labels")
    return labels


def evaluation_cuts(target, split_record):
    if target in ("c1", "c4"):
        # The existing train.py experiment evaluated target_{tool}.npz in full.
        return ALL_CUTS.copy(), "existing full-lifecycle C1/C4 target evaluation: 1..315"
    # Read split metadata only. Do not read y_true or prediction columns here.
    frame = pd.read_csv(split_record, usecols=["tool", "cut_index", "split"])
    cuts = frame.loc[(frame.tool.str.upper() == "C6") &
                     (frame.split == "target_test_suffix_70pct"), "cut_index"].astype(int).tolist()
    if cuts != list(range(95, 316)):
        raise ValueError("C6 fixed test cuts must be 95..315")
    return cuts, "existing fixed C6 suffix: 95..315"


def build_model(seed, device):
    seed_everything(seed)
    model = nn.Module()
    model.feature_extractor = ResNet18()
    model.regressor = nn.Sequential(nn.Linear(512, 1))
    digest = model_hash(model)
    return model.to(device), digest


def train_model(method, x_source, y_source, x_target, args, pair_dir, device):
    model, initial_hash = build_model(args.seed, device)
    generator = torch.Generator().manual_seed(args.seed)
    source_loader = DataLoader(TensorDataset(torch.from_numpy(x_source), torch.from_numpy(y_source)),
                               batch_size=args.batch_size, shuffle=True, drop_last=False, generator=generator)
    target_loader = None
    if method == "daregram":
        target_loader = DataLoader(TensorDataset(torch.from_numpy(x_target), torch.arange(1, 316)),
                                   batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   generator=torch.Generator().manual_seed(args.seed + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    usage = set()
    for epoch in range(1, args.epochs + 1):
        model.train()
        source_mse_sum = gram_sum = 0.0
        target_iter = iter(target_loader) if target_loader is not None else None
        epoch_usage = set()
        tradeoff = 0.0 if args.epochs == 1 else 2 / (1 + math.exp(-10 * (epoch - 1) / (args.epochs - 1))) - 1
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            source_features = model.feature_extractor(xb)
            source_mse = F.mse_loss(model.regressor(source_features), yb)
            loss = source_mse
            if target_iter is not None:
                xt, cut_ids = next(target_iter)
                epoch_usage.update(cut_ids.tolist())
                target_features = model.feature_extractor(xt.to(device))
                gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), source_features, target_features)
                loss = loss + tradeoff * gram
                gram_sum += gram.item()
            loss.backward()
            optimizer.step()
            source_mse_sum += source_mse.item()
        if target_iter is not None:
            if epoch_usage != set(ALL_CUTS):
                raise ValueError("DARE-GRAM did not use all 315 unlabeled target cuts this epoch")
            usage.update(epoch_usage)
        logging.info("%s epoch %d/%d source_MSE=%.6f gram=%.6f target_unique=%d",
                     method, epoch, args.epochs, source_mse_sum / len(source_loader),
                     gram_sum / len(source_loader), len(epoch_usage))
    path = pair_dir / method / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, path)
    logging.info("Saved %s final checkpoint: %s", method, path)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return initial_hash, sorted(usage)


def predict(method, x_target, cuts, args, pair_dir, device):
    model, _ = build_model(args.seed, device)
    checkpoint = torch.load(pair_dir / method / "final.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, len(cuts), args.batch_size):
            indices = np.asarray(cuts[offset:offset + args.batch_size]) - 1
            xb = torch.from_numpy(x_target[indices]).to(device)
            values.extend(model.regressor(model.feature_extractor(xb)).cpu().numpy().reshape(-1).tolist())
    del model
    return np.asarray(values, dtype=np.float64)


def metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE_percent": float(np.mean(np.abs((y_true - y_pred) /
                                                     np.maximum(np.abs(y_true), 1e-8))) * 100)}


def save_evaluation(method, target, cuts, y_true, y_pred, pair_dir):
    folder = pair_dir / method
    csv_path = folder / "predictions.csv"
    pd.DataFrame({"cut_index": cuts, "true_vb": y_true, "pred_vb": y_pred}).to_csv(
        csv_path, index=False, float_format="%.17g")
    reread = pd.read_csv(csv_path)
    result = metrics(reread.true_vb.to_numpy(), reread.pred_vb.to_numpy())
    (folder / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cuts, y_true, label="True VB")
    ax.plot(cuts, y_pred, label="Predicted VB")
    ax.set(xlabel=f"{target.upper()} cut index", ylabel="VB", title=f"{pair_dir.name}: {method}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "prediction.png", dpi=200)
    plt.close(fig)
    return result


def write_method_records(pair_dir):
    """Keep a method-local copy of the shared settings and relevant training log."""
    config = json.loads((pair_dir / "config.json").read_text(encoding="utf-8"))
    log_lines = (pair_dir / "run.log").read_text(encoding="utf-8").splitlines()
    for method in ("source_only", "daregram"):
        method_config = dict(config, method=method,
                             actual_training_target_cuts=[] if method == "source_only" else ALL_CUTS,
                             target_training_information=config["target_information_usage"][method])
        folder = pair_dir / method
        (folder / "config.json").write_text(json.dumps(method_config, indent=2, ensure_ascii=False), encoding="utf-8")
        relevant = [line for line in log_lines if f"{method} epoch " in line or
                    f"Saved {method} final checkpoint" in line or "Pair " in line or
                    f"STFT {config['source'].upper()}:" in line or
                    (method == "daregram" and f"STFT {config['target'].upper()}:" in line) or
                    "Completed " in line]
        (folder / "run.log").write_text("\n".join(relevant) + "\n", encoding="utf-8")


def run_pair(source, target, args):
    if source == target:
        raise ValueError("Source and target must differ")
    pair_dir = args.out_root / f"{source}_to_{target}"
    expected = {"source": source, "target": target, "seed": args.seed, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr,
                "raw_root": str(args.raw_root.resolve()), "split_record": str(args.split_record.resolve())}
    if pair_dir.exists():
        config_path = pair_dir / "config.json"
        if not args.all or not config_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing pair directory: {pair_dir}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in expected.items()) or not all(
            (pair_dir / method / "metrics.json").exists() for method in ("source_only", "daregram")
        ):
            raise ValueError(f"Existing pair is incomplete or has different configuration: {pair_dir}")
        logging.info("Reusing completed pair %s", pair_dir)
        return
    pair_dir.mkdir(parents=True)
    for method in ("source_only", "daregram"):
        (pair_dir / method).mkdir()
    handler = logging.FileHandler(pair_dir / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        device = torch.device(args.device)
        cuts, cut_definition = evaluation_cuts(target, args.split_record)
        logging.info("Pair %s -> %s; evaluation cuts %s", source, target, cuts)
        raw_source = stft_features(args.raw_root, source)
        source_labels = wear_labels(args.raw_root, source)
        mean = raw_source.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw_source.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        x_source = ((raw_source - mean[None, :, None, None]) /
                    (std[None, :, None, None] + 1e-8)).astype(np.float32)
        del raw_source
        config = dict(expected, device=str(device), protocol="transductive UDA for DARE-GRAM",
                      input="center 4096; 6 channels; STFT nperseg=256 noverlap=224; log1p magnitude; 128x128",
                      backbone="ResNet18", regressor="Linear(512, 1)", optimizer="Adam",
                      scheduler="fixed", source_loss="MSE", daregram_loss="original inverse Gram loss with exp epoch tradeoff",
                      checkpoint_selection="final epoch, no target labels or target metrics",
                      source_sample_count=len(source_labels), source_cuts=ALL_CUTS,
                      target_unlabeled_cuts=ALL_CUTS, evaluation_cuts=cuts,
                      evaluation_cut_definition=cut_definition, normalization_source=source,
                      normalization_mean=mean.tolist(), normalization_std=std.tolist(),
                      source_target_distinct=True,
                      target_labels_read_stage="after both final checkpoints and predictions",
                      target_information_usage={"source_only": "no target data during training; evaluation features only after training",
                                                "daregram": "all 315 unlabeled target STFT inputs in alignment, including evaluation cuts"})
        if target == "c6":
            config["split_record_sha256"] = file_hash(args.split_record)
        (pair_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        source_hash, source_usage = train_model("source_only", x_source, source_labels, None,
                                                args, pair_dir, device)
        if source_usage:
            raise ValueError("Source-only unexpectedly used target cuts")
        raw_target = stft_features(args.raw_root, target)
        x_target = ((raw_target - mean[None, :, None, None]) /
                    (std[None, :, None, None] + 1e-8)).astype(np.float32)
        del raw_target
        dare_hash, dare_usage = train_model("daregram", x_source, source_labels, x_target,
                                            args, pair_dir, device)
        if source_hash != dare_hash or dare_usage != ALL_CUTS:
            raise ValueError("Initializations differ or target usage is incomplete")
        config["initial_model_sha256"] = source_hash
        config["daregram_actual_unlabeled_cuts"] = dare_usage
        config["source_only_actual_training_target_cuts"] = source_usage
        config["target_label_reads_before_checkpoints"] = 0
        (pair_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        preds = {method: predict(method, x_target, cuts, args, pair_dir, device)
                 for method in ("source_only", "daregram")}
        # The first target wear read is here, after both models and predictions are frozen.
        target_labels = wear_labels(args.raw_root, target, evaluation_dir=pair_dir)
        y_true = target_labels[np.asarray(cuts) - 1].astype(np.float64)
        results = {method: save_evaluation(method, target, cuts, y_true, preds[method], pair_dir)
                   for method in ("source_only", "daregram")}
        delta = {key: results["daregram"][key] - results["source_only"][key] for key in METRICS}
        audit = {"source": source, "target": target, "source_target_distinct": True,
                 "source_sample_count": 315, "source_labeled_cuts": ALL_CUTS,
                 "target_unlabeled_cuts": dare_usage, "target_label_first_read": "after both final checkpoints and predictions",
                 "target_label_reads_before_checkpoints": 0,
                 "method_target_use": config["target_information_usage"], "evaluation_cuts": cuts,
                 "same_initial_model_sha256": source_hash}
        (pair_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        (pair_dir / "comparison.json").write_text(json.dumps({"source_only": results["source_only"],
            "daregram": results["daregram"], "daregram_minus_source_only": delta}, indent=2), encoding="utf-8")
        report = (f"# {source.upper()} -> {target.upper()}\n\n"
                  f"Protocol: transductive UDA. DARE-GRAM uses unlabeled target cuts 1-315, including test inputs. "
                  f"Target wear labels are read only after both final checkpoints and predictions.\n\n"
                  f"Evaluation: {cut_definition}; actual cuts: {', '.join(map(str, cuts))}. "
                  f"Other target tools have different evaluation intervals and their metrics are not the same-test-set comparison.\n\n"
                  "| Method | MAE | RMSE | R² | MAPE (%) |\n|---|---:|---:|---:|---:|\n" +
                  "\n".join(f"| {name} | {results[name]['MAE']:.6f} | {results[name]['RMSE']:.6f} | "
                            f"{results[name]['R2']:.6f} | {results[name]['MAPE_percent']:.6f} |"
                            for name in ("source_only", "daregram")) +
                  f"\n| DARE-GRAM minus source-only | {delta['MAE']:+.6f} | {delta['RMSE']:+.6f} | "
                  f"{delta['R2']:+.6f} | {delta['MAPE_percent']:+.6f} |\n")
        (pair_dir / "report.md").write_text(report, encoding="utf-8")
        logging.info("Completed %s -> %s: %s", source, target, results)
        handler.flush()
        write_method_records(pair_dir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def write_summary(root):
    rows = []
    for source in TOOLS:
        for target in TOOLS:
            if source == target:
                continue
            folder = root / f"{source}_to_{target}"
            if not (folder / "comparison.json").exists():
                continue
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
            row = {"source": source, "target": target, "evaluation_cuts": ",".join(map(str, config["evaluation_cuts"])),
                   "evaluation_count": len(config["evaluation_cuts"]), "protocol": "transductive UDA"}
            for method, prefix in (("source_only", "source_only"), ("daregram", "daregram"),
                                   ("daregram_minus_source_only", "delta")):
                row.update({f"{prefix}_{metric}": comparison[method][metric] for metric in METRICS})
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
        lines = ["# Single-source directed pairs", "",
                 "DARE-GRAM uses all 315 unlabeled target cut inputs, including evaluation inputs: transductive UDA.",
                 "C6 uses cuts 95-315; C1/C4 use 1-315. Compare methods within a pair; different target intervals are not the same test set.", "",
                 f"Completed {len(rows)}/6 directed pairs. Exact cuts and all four metrics/deltas are in summary.csv.", "",
                 "| Pair | Cuts | source-only MAE | DARE-GRAM MAE | ΔMAE | source-only RMSE | DARE-GRAM RMSE | ΔRMSE | source-only R² | DARE-GRAM R² | ΔR² | source-only MAPE % | DARE-GRAM MAPE % | ΔMAPE pp |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            vals = [f"{row['source'].upper()}->{row['target'].upper()}", str(row["evaluation_count"])]
            for metric in METRICS:
                vals.extend(f"{row[f'{prefix}_{metric}']:.4f}" for prefix in ("source_only", "daregram", "delta"))
            lines.append("| " + " | ".join(vals) + " |")
        (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = arguments()
    args.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out_root / "batch.log", encoding="utf-8")])
    pairs = [(source, target) for source in TOOLS for target in TOOLS if source != target] if args.all else [(args.source, args.target)]
    for source, target in pairs:
        run_pair(source, target, args)
        write_summary(args.out_root)


if __name__ == "__main__":
    main()
