"""Causal next-cut wear prediction using the existing PHM2010 STFT cache.

Stages are explicit: audit, source-only (K=1 and K=5), then K=5 DARE-GRAM.
Every output root must be new; completed stages are only read, never rewritten.
"""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import run_single_source_pairs as base
from models.DAREGRAM import Trainset as DAREGRAM
from networks.resnet import ResNet18

TOOLS = ("c1", "c4", "c6")
METHODS = ("k1_source_only", "k5_source_only", "k5_daregram")
EVAL_CUTS = list(range(6, 316))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=TOOLS, default="c1")
    parser.add_argument("--target", choices=TOOLS, default="c6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("audit", "source", "daregram", "report", "all"), default="all")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/historical_next_cut_seed42_20260926"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/five_seed_paired/feature_cache"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.source == args.target or args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error("Source and target must differ; epochs >= 1, batch size >= 2 and LR > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cache(args, tool):
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cuts"] == base.ALL_CUTS
    assert manifest["raw_root"].lower() == str(args.raw_root.resolve()).lower()
    assert manifest["stft_code_sha256"] == base.file_hash(Path(base.sampling.__file__))
    assert manifest["tools"][tool]["raw_fingerprint"] == raw_fingerprint(args.raw_root, tool)
    path = args.cache_root / f"{tool}_stft.npy"
    assert manifest["tools"][tool]["feature_sha256"] == base.file_hash(path)
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    assert x.shape == (315, 6, 128, 128) and x.dtype == np.float32
    return x, manifest["tools"][tool]["feature_sha256"]


def raw_fingerprint(root, tool):
    digest = hashlib.sha256()
    for path in base.signal_files(root, tool):
        stat = path.stat()
        digest.update(f"{path.name}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def mapping(tool, k, adaptation=False):
    # An extra unlabeled target window ending at cut 315 exposes all target
    # signals to transductive adaptation. It never generates an evaluated VB.
    last = 316 if adaptation else 315
    rows = []
    for prediction_cut in range(6, last + 1):
        inputs = list(range(prediction_cut - k, prediction_cut))
        rows.append({"tool": tool, "prediction_cut": prediction_cut,
                     "input_cuts": inputs, "last_input_cut": inputs[-1]})
    return rows


def audit_mapping(source, target, out):
    records = {}
    for tool in (source, target):
        for k in (1, 5):
            rows = mapping(tool, k)
            assert len(rows) == 310
            assert [r["prediction_cut"] for r in rows] == EVAL_CUTS
            assert all(r["prediction_cut"] == r["last_input_cut"] + 1 for r in rows)
            assert all(max(r["input_cuts"]) < r["prediction_cut"] for r in rows)
            assert all(len(r["input_cuts"]) == k and r["input_cuts"] == list(range(r["prediction_cut"] - k, r["prediction_cut"])) for r in rows)
            assert all(r["tool"] == tool for r in rows)
            key = f"{tool}_k{k}"
            records[key] = {"sample_count": len(rows), "first_3": rows[:3], "last_3": rows[-3:]}
            print(f"{key}: first_3={rows[:3]} last_3={rows[-3:]}", flush=True)
            pd.DataFrame({"tool": tool, "prediction_cut": EVAL_CUTS,
                          "input_cuts": [json.dumps(r["input_cuts"]) for r in rows]}).to_csv(
                              out / f"mapping_{key}.csv", index=False)
    adapt = [r for r in mapping(target, 5, adaptation=True) if r["prediction_cut"] != 160]
    assert len(adapt) == 310 and adapt[-1]["input_cuts"] == [311, 312, 313, 314, 315]
    assert set(c for row in adapt for c in row["input_cuts"]) == set(base.ALL_CUTS)
    records["adaptation"] = {"unlabeled_window_count": 310, "signal_cuts": base.ALL_CUTS,
                              "excluded_redundant_window_prediction_cut": 160,
                              "last_window": adapt[-1], "labels_used": False}
    write_json(out / "mapping_audit.json", records)
    return records


class WindowDataset(Dataset):
    def __init__(self, images, k, labels=None, adaptation=False):
        self.images = images
        self.k = k
        self.labels = labels
        self.cuts = list(range(6, 317 if adaptation else 316))
        if adaptation:
            self.cuts.remove(160)  # 310 windows, all 315 signals remain covered.
        if labels is not None:
            assert len(labels) == 315 and not adaptation

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, idx):
        cut = self.cuts[idx]
        # Array index cut-1 corresponds to actual cut index. The slice below
        # therefore ends at cut-2, strictly before prediction_cut.
        x = self.images[cut - self.k - 1:cut - 1]
        assert x.shape == (self.k, 6, 128, 128)
        if self.labels is None:
            return torch.from_numpy(x), cut
        return torch.from_numpy(x), np.float32(self.labels[cut - 1]), cut


class NextCutModel(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.feature_extractor = ResNet18()
        self.gru = nn.GRU(512, 512, batch_first=True) if k == 5 else None
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        assert x.ndim == 5 and x.shape[1:] == (self.k, 6, 128, 128)
        b = x.shape[0]
        features = self.feature_extractor(x.reshape(b * self.k, 6, 128, 128)).reshape(b, self.k, 512)
        aligned = self.gru(features)[1][-1] if self.gru is not None else features[:, 0]
        return self.regressor(aligned), aligned


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalized_images(raw, mean, std):
    x = ((raw - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)).astype(np.float32)
    assert x.shape == (315, 6, 128, 128) and np.isfinite(x).all()
    return x


def train_one(args, method, x_source, y_source, x_target, out):
    k = 1 if method == "k1_source_only" else 5
    is_dare = method == "k5_daregram"
    assert (x_target is not None) == is_dare
    folder = out / method
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite {folder}")
    folder.mkdir()
    seed_all(args.seed)
    device = torch.device(args.device)
    model = NextCutModel(k).to(device)
    init_hash = base.model_hash(model)
    source = DataLoader(WindowDataset(x_source, k, y_source), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    target = None
    if is_dare:
        target = DataLoader(WindowDataset(x_target, 5, adaptation=True), batch_size=args.batch_size,
                            shuffle=True, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + 1))
        assert len(target) == len(source)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gram_holder = type("Device", (), {"device": device})()
    history = []
    all_target_inputs = set()
    for epoch in range(1, args.epochs + 1):
        model.train()
        mse_total = gram_total = 0.0
        source_seen, target_seen = [], []
        target_iter = iter(target) if target is not None else None
        tradeoff = 0.0 if args.epochs == 1 else 2 / (1 + math.exp(-10 * (epoch - 1) / (args.epochs - 1))) - 1
        for xb, yb, cut in source:
            source_seen.extend(cut.tolist())
            optimizer.zero_grad(set_to_none=True)
            prediction, feat_s = model(xb.to(device))
            mse = F.mse_loss(prediction, yb.to(device).unsqueeze(1))
            loss = mse
            if target_iter is not None:
                xt, target_cut = next(target_iter)
                target_seen.extend(target_cut.tolist())
                _, feat_t = model(xt.to(device))
                gram = DAREGRAM.DARE_GRAM_LOSS(gram_holder, feat_s, feat_t)
                loss = loss + tradeoff * gram
                gram_total += float(gram.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {method} epoch {epoch}")
            loss.backward()
            optimizer.step()
            mse_total += float(mse.detach())
        assert sorted(source_seen) == EVAL_CUTS
        if is_dare:
            assert sorted(target_seen) == [c for c in range(6, 317) if c != 160]
            for pred_cut in target_seen:
                all_target_inputs.update(range(pred_cut - 5, pred_cut))
        row = {"epoch": epoch, "source_mse": mse_total / len(source),
               "gram": gram_total / len(source), "tradeoff": tradeoff,
               "source_samples": len(source_seen), "target_unlabeled_windows": len(target_seen),
               "optimizer_updates": len(source)}
        history.append(row)
        logging.info("%s epoch=%d/%d MSE=%.6f Gram=%.6f updates=%d", method, epoch,
                     args.epochs, row["source_mse"], row["gram"], len(source))
    if is_dare:
        assert all_target_inputs == set(base.ALL_CUTS)
    checkpoint = folder / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "k": k}, checkpoint)
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    record = {"method": method, "source": args.source, "target": args.target, "seed": args.seed,
              "k": k, "epochs": args.epochs, "actual_batch_size": args.batch_size,
              "gradient_accumulation": 1, "updates_per_epoch": len(source),
              "total_optimizer_updates": args.epochs * len(source), "source_samples_per_epoch": 310,
              "target_unlabeled_windows_per_epoch": 310 if is_dare else 0,
              "target_unlabeled_signal_cuts": sorted(all_target_inputs),
              "lr": args.lr, "optimizer": "Adam", "source_loss": "MSE",
              "daregram_loss": "original inverse Gram loss on GRU output, epoch logistic tradeoff" if is_dare else None,
              "initial_model_sha256": init_hash, "final_checkpoint_sha256": base.file_hash(checkpoint),
              "checkpoint_selection": "final epoch without target labels or target metrics"}
    write_json(folder / "config.json", record)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def predict_one(args, method, x_target, out):
    k = 1 if method == "k1_source_only" else 5
    model = NextCutModel(k).to(args.device)
    checkpoint = torch.load(out / method / "final.pth", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(WindowDataset(x_target, k), batch_size=args.batch_size,
                        shuffle=False, drop_last=False, num_workers=0)
    values, cuts = [], []
    with torch.no_grad():
        for xb, cut in loader:
            prediction, _ = model(xb.to(args.device))
            values.extend(prediction.cpu().numpy().reshape(-1).tolist())
            cuts.extend(cut.tolist())
    assert cuts == EVAL_CUTS and len(values) == 310
    return np.asarray(values, dtype=np.float64)


def report(args, out, x_target):
    if (out / "common_cut_metrics.json").exists() or (out / "common_cut_predictions.csv").exists():
        raise FileExistsError(f"Refusing to overwrite an existing report in {out}")
    assert all((out / method / "final.pth").is_file() for method in METHODS)
    predictions = {method: predict_one(args, method, x_target, out) for method in METHODS}
    # This is the first target wear read in this runner, after all three
    # checkpoints exist and all three prediction vectors are fixed.
    y = base.wear_labels(args.raw_root, args.target)[5:].astype(np.float64)
    assert len(y) == 310
    frame = pd.DataFrame({"cut_index": EVAL_CUTS, "true_vb": y})
    metrics = {}
    for method in METHODS:
        pred = predictions[method]
        method_frame = pd.DataFrame({"cut_index": range(1, 316),
                                     "input_cuts": ["" if cut < 6 else json.dumps(list(range(cut - (1 if method == "k1_source_only" else 5), cut))) for cut in range(1, 316)],
                                     "status": ["no_prediction" if cut < 6 else "predicted" for cut in range(1, 316)],
                                     "pred_vb": np.r_[np.full(5, np.nan), pred]})
        # Real VB is intentionally absent from the first five rows.
        method_frame["true_vb"] = np.r_[np.full(5, np.nan), y]
        method_frame.to_csv(out / method / "predictions_all_cuts.csv", index=False, float_format="%.17g")
        frame[method] = pred
        metrics[method] = {"cuts": "6..315", "n": 310,
                           "R2": float(r2_score(y, pred)),
                           "MAE": float(mean_absolute_error(y, pred)),
                           "RMSE": float(np.sqrt(mean_squared_error(y, pred)))}
    frame.to_csv(out / "common_cut_predictions.csv", index=False, float_format="%.17g")
    write_json(out / "common_cut_metrics.json", metrics)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(EVAL_CUTS, y, label="C6 true VB", color="black", linewidth=1.5)
    for method in METHODS:
        ax.plot(EVAL_CUTS, predictions[method], label=method, linewidth=1)
    ax.set(xlabel="C6 prediction cut", ylabel="VB", title=f"{args.source.upper()}→{args.target.upper()} next-cut prediction, seed {args.seed}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "common_cut_curves.png", dpi=200)
    plt.close(fig)
    write_json(out / "label_leakage_audit.json", {
        "source_labels": "C1 VB only used as next-cut supervised labels at training stage",
        "target_label_first_read": "report stage, after three final checkpoints and prediction vectors",
        "target_label_reads_before_model_and_predictions_fixed": 0,
        "source_only_target_training_signals": [],
        "daregram_target_training_signals": base.ALL_CUTS,
        "daregram_target_training_labels": [],
        "inference_window_rule": "each prediction at cut t uses only signal cuts t-K..t-1",
        "adaptation_protocol": "full target-domain unlabeled adaptation plus causal-window inference; not strict online adaptation",
        "predicted_cuts": EVAL_CUTS,
        "unpredicted_cuts": list(range(1, 6)),
    })
    return metrics


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out_root / f"{args.source}_to_{args.target}" / f"seed_{args.seed}"
    if args.stage in ("audit", "all"):
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment: {out}")
        out.mkdir(parents=True)
        audit_mapping(args.source, args.target, out)
        raw_source, source_hash = load_cache(args, args.source)
        _, target_hash = load_cache(args, args.target)
        mean = raw_source.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (raw_source.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        write_json(out / "config.json", {"source": args.source, "target": args.target, "seed": args.seed,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "device": args.device,
            "cache_root": str(args.cache_root.resolve()), "raw_root": str(args.raw_root.resolve()),
            "source_stft_sha256": source_hash, "target_stft_sha256": target_hash,
            "stft": "existing centered 4096, six channels, STFT 256/224, log1p magnitude, 128x128",
            "normalization": "source-only per-channel mean and std over all 315 C1 STFT images",
            "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
            "supervised_prediction_cuts": EVAL_CUTS, "checkpoint_selection": "last epoch"})
        if args.stage == "audit":
            return
    if not (out / "mapping_audit.json").is_file() or not (out / "config.json").is_file():
        raise FileNotFoundError("Run --stage audit first")
    log_handler = logging.FileHandler(out / "training.log", encoding="utf-8")
    logging.getLogger().addHandler(log_handler)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    for name in ("source", "target", "seed", "epochs", "batch_size", "lr", "device"):
        assert config[name] == getattr(args, name), f"Configuration mismatch: {name}"
    if args.stage in ("source", "daregram", "report", "all"):
        raw_source, source_hash = load_cache(args, args.source)
        raw_target, target_hash = load_cache(args, args.target)
        assert source_hash == config["source_stft_sha256"] and target_hash == config["target_stft_sha256"]
        mean = np.asarray(config["normalization_mean"], dtype=np.float32)
        std = np.asarray(config["normalization_std"], dtype=np.float32)
        x_target = normalized_images(raw_target, mean, std)
        del raw_target
    if args.stage in ("source", "all"):
        x_source = normalized_images(raw_source, mean, std)
        # Source wear is legal only for next-cut labels. No target wear read.
        y_source = base.wear_labels(args.raw_root, args.source)
        for method in METHODS[:2]:
            train_one(args, method, x_source, y_source, None, out)
        if args.stage == "source":
            return
    if args.stage in ("daregram", "all"):
        assert all((out / method / "final.pth").is_file() for method in METHODS[:2])
        if "x_source" not in locals():
            x_source = normalized_images(raw_source, mean, std)
            y_source = base.wear_labels(args.raw_root, args.source)
        train_one(args, METHODS[2], x_source, y_source, x_target, out)
        if args.stage == "daregram":
            return
    if args.stage in ("report", "all"):
        print(json.dumps(report(args, out, x_target), indent=2), flush=True)


if __name__ == "__main__":
    main()
