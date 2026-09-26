"""Three fixed signal windows per Cut, with one sampled window per training epoch.

Run from upstream-reproduction. The frozen A artifacts are only read by
``report_multi_window_ablation.py``; this script writes solely below --out-root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import data_sampling as sampling
import run_single_source_pairs as base
from models.DAREGRAM import Trainset as DAREGRAM


TOOLS = base.TOOLS
PAIRS = tuple((s, t) for s in TOOLS for t in TOOLS if s != t)
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("source_only", "daregram")
EPOCHS, BATCH_SIZE, LR = 50, 63, 1e-3
WINDOW_LENGTH = 4096
POSITIONS = (0.25, 0.50, 0.75)
DEFAULT_OUT = Path("artifacts/multi_window_25_50_75_20260926")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def window_start(signal: np.ndarray, fraction: float) -> tuple[int, str]:
    """Nearest valid window to the fixed center; no label or tool-specific rule.

    A sample row is invalid when any of six channels is nonfinite or all six are
    exactly zero. Ties are resolved toward the earlier start. In normal data the
    requested window is valid and this search is never used.
    """
    length = len(signal)
    if length < WINDOW_LENGTH:
        raise ValueError(f"Signal shorter than {WINDOW_LENGTH}: {length}")
    requested = math.floor(fraction * length) - WINDOW_LENGTH // 2
    clipped = min(max(requested, 0), length - WINDOW_LENGTH)
    bad = ~np.isfinite(signal).all(axis=1) | np.all(signal == 0, axis=1)
    counts = np.r_[0, np.cumsum(bad, dtype=np.int64)]
    valid = np.flatnonzero(counts[WINDOW_LENGTH:] - counts[:-WINDOW_LENGTH] == 0)
    if not len(valid):
        raise ValueError("No 4096-point window without nonfinite/all-zero rows")
    near = np.searchsorted(valid, clipped)
    candidates = valid[max(near - 1, 0):min(near + 1, len(valid))]
    start = int(min(candidates, key=lambda v: (abs(int(v) - clipped), int(v))))
    if start != clipped:
        reason = "invalid_rows_nearest_valid"
    elif clipped != requested:
        reason = "boundary_clip"
    else:
        reason = "none"
    return start, reason


def prepare_cache(raw_root: Path, out_root: Path) -> None:
    cache = out_root / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    code_hash = base.file_hash(Path(sampling.__file__))
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    manifest = {"raw_root": str(raw_root.resolve()), "stft_code_sha256": code_hash,
                "shape": [315, 3, 6, 128, 128], "window_length": WINDOW_LENGTH,
                "center_fractions": list(POSITIONS), "tools": {}}
    if existing and any(existing[k] != manifest[k] for k in
                        ("raw_root", "stft_code_sha256", "shape", "window_length", "center_fractions")):
        raise ValueError("Existing cache uses a different raw root, STFT code or window rule")
    for tool in TOOLS:
        files = base.signal_files(raw_root, tool)
        raw_fingerprint = hashlib.sha256("".join(
            f"{p.name}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}\n" for p in files).encode()).hexdigest()
        feature_path = cache / f"{tool}_stft.npy"
        positions_path = cache / f"{tool}_windows.csv"
        if existing and tool in existing["tools"]:
            item = existing["tools"][tool]
            if (item["raw_fingerprint"] != raw_fingerprint or
                item["feature_sha256"] != base.file_hash(feature_path) or
                item["windows_sha256"] != base.file_hash(positions_path)):
                raise ValueError(f"Cached {tool} files changed")
            manifest["tools"][tool] = item
            logging.info("Verified %s feature cache", tool)
            continue
        if feature_path.exists() or positions_path.exists():
            raise FileExistsError(f"Unrecorded cache for {tool}")
        features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                              shape=(315, 3, 6, 128, 128))
        records = []
        for cut, path in enumerate(files, 1):
            frame = sampling._read_pass_df(str(path))
            signal = frame[sampling.EXPECTED_INPUT_COLS].to_numpy()
            for wi, fraction in enumerate(POSITIONS):
                requested = math.floor(fraction * len(signal)) - WINDOW_LENGTH // 2
                start, reason = window_start(signal, fraction)
                end = start + WINDOW_LENGTH
                image = sampling._stft_crop_and_resize(signal[start:end].T)
                if image.shape != (6, 128, 128) or not np.isfinite(image).all():
                    raise ValueError(f"Invalid STFT {tool} cut {cut} window {wi}")
                features[cut - 1, wi] = image
                records.append({"tool": tool, "cut_index": cut, "window": wi,
                                "center_fraction": fraction, "signal_length": len(signal),
                                "requested_start_0based": requested, "start_0based": start,
                                "end_exclusive_0based": end, "correction_reason": reason,
                                "raw_signal": str(path.resolve())})
            if cut % 25 == 0 or cut == 315:
                logging.info("STFT %s %d/315", tool, cut)
        features.flush()
        del features
        pd.DataFrame(records).to_csv(positions_path, index=False)
        item = {"raw_fingerprint": raw_fingerprint,
                "feature_sha256": base.file_hash(feature_path),
                "windows_sha256": base.file_hash(positions_path),
                "corrections": pd.Series([r["correction_reason"] for r in records]).value_counts().to_dict()}
        manifest["tools"][tool] = item
        write_json(manifest_path, manifest)
        logging.info("Saved %s feature cache: %s", tool, item["corrections"])
    write_json(manifest_path, manifest)


def cache_view(out_root: Path, tool: str) -> np.ndarray:
    x = np.load(out_root / "feature_cache" / f"{tool}_stft.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (315, 3, 6, 128, 128) or x.dtype != np.float32:
        raise ValueError(f"Invalid cached tensor for {tool}")
    return x


def source_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float64 accumulators avoid allocating another 372 MB tensor.
    sums = np.zeros(6, dtype=np.float64)
    squares = np.zeros(6, dtype=np.float64)
    for cut in range(315):
        block = np.asarray(x[cut], dtype=np.float64)
        sums += block.sum(axis=(0, 2, 3))
        squares += np.square(block).sum(axis=(0, 2, 3))
    n = 315 * 3 * 128 * 128
    mean = sums / n
    std = np.sqrt(np.maximum(squares / n - mean**2, 0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


class CutWindows(Dataset):
    def __init__(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 labels: np.ndarray | None = None):
        self.x, self.mean, self.std, self.labels = x, mean[:, None, None], std[:, None, None], labels
        self.choices = np.zeros(315, dtype=np.int64)

    def set_epoch(self, seed: int, epoch: int, target: bool = False) -> None:
        self.choices = np.random.default_rng(seed + 100_003 * epoch + (10_000_019 if target else 0)).integers(0, 3, 315)

    def __len__(self) -> int:
        return 315

    def __getitem__(self, index: int):
        wi = self.choices[index]
        image = (self.x[index, wi] - self.mean) / self.std
        if self.labels is None:
            return torch.from_numpy(np.asarray(image, dtype=np.float32)), index + 1
        return torch.from_numpy(np.asarray(image, dtype=np.float32)), self.labels[index], index + 1


def train(method: str, source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray | None,
          mean: np.ndarray, std: np.ndarray, seed: int, folder: Path, device: torch.device) -> dict:
    ckpt = folder / method / "final.pth"
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=True)
        if saved["epoch"] != EPOCHS or saved["seed"] != seed:
            raise ValueError(f"Mismatched existing checkpoint: {ckpt}")
        logging.info("Reusing completed M checkpoint: %s", ckpt)
        return {"checkpoint_sha256": base.file_hash(ckpt), "resumed_checkpoint": True}
    model, initial_hash = base.build_model(seed, device)
    source = CutWindows(source_x, mean, std, source_y)
    source_loader = DataLoader(source, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                               generator=torch.Generator().manual_seed(seed))
    target = CutWindows(target_x, mean, std) if target_x is not None else None
    target_loader = (DataLoader(target, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                                generator=torch.Generator().manual_seed(seed + 1))
                     if target is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        source.set_epoch(seed, epoch)
        if target is not None:
            target.set_epoch(seed, epoch, target=True)
        model.train()
        target_iter = iter(target_loader) if target_loader is not None else None
        source_order, target_order = [], []
        mse_sum = gram_sum = 0.0
        tradeoff = 2 / (1 + math.exp(-10 * (epoch - 1) / (EPOCHS - 1))) - 1
        for xb, yb, cuts in source_loader:
            source_order.extend(cuts.tolist())
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            fs = model.feature_extractor(xb)
            mse = F.mse_loss(model.regressor(fs), yb)
            loss = mse
            if target_iter is not None:
                xt, target_cuts = next(target_iter)
                target_order.extend(target_cuts.tolist())
                ft = model.feature_extractor(xt.to(device))
                gram = DAREGRAM.DARE_GRAM_LOSS(type("Device", (), {"device": device})(), fs, ft)
                loss = loss + tradeoff * gram
                gram_sum += gram.item()
            loss.backward()
            optimizer.step()
            mse_sum += mse.item()
        if sorted(source_order) != base.ALL_CUTS or (target is not None and sorted(target_order) != base.ALL_CUTS):
            raise ValueError("Each epoch must see each Cut exactly once")
        epoch_rows.append({"epoch": epoch, "source_cuts": len(source_order),
                           "target_cuts": len(target_order), "steps": len(source_loader),
                           "source_MSE": mse_sum / len(source_loader),
                           "gram": gram_sum / len(source_loader), "domain_weight": tradeoff,
                           "source_order_sha256": hashlib.sha256(np.asarray(source_order, np.int32).tobytes()).hexdigest(),
                           "source_windows_sha256": hashlib.sha256(source.choices.tobytes()).hexdigest(),
                           "target_windows_sha256": hashlib.sha256(target.choices.tobytes()).hexdigest() if target else ""})
        logging.info("%s seed=%d epoch=%d/%d MSE=%.5f gram=%.5f", method, seed, epoch, EPOCHS,
                     epoch_rows[-1]["source_MSE"], epoch_rows[-1]["gram"])
    torch.save({"model": model.state_dict(), "epoch": EPOCHS, "seed": seed,
                "initial_model_sha256": initial_hash}, ckpt)
    pd.DataFrame(epoch_rows).to_csv(folder / method / "epoch_audit.csv", index=False)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"checkpoint_sha256": base.file_hash(ckpt), "initial_model_sha256": initial_hash,
            "resumed_checkpoint": False}


def predict_windows(method: str, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                    seed: int, folder: Path, device: torch.device) -> np.ndarray:
    model, _ = base.build_model(seed, device)
    checkpoint = torch.load(folder / method / "final.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    result = np.empty((315, 3), dtype=np.float64)
    with torch.no_grad():
        for wi in range(3):
            for offset in range(0, 315, BATCH_SIZE):
                block = np.asarray(x[offset:offset + BATCH_SIZE, wi])
                normalized = (block - mean[None, :, None, None]) / std[None, :, None, None]
                xb = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).to(device)
                pred = model.regressor(model.feature_extractor(xb)).cpu().numpy().reshape(-1)
                result[offset:offset + len(pred), wi] = pred
    del model
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite window prediction")
    return result


def run_pair(raw_root: Path, out_root: Path, source: str, target: str, seed: int,
             device: torch.device) -> None:
    folder = out_root / f"{source}_to_{target}" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        (folder / method).mkdir(exist_ok=True)
    source_x, target_x = cache_view(out_root, source), cache_view(out_root, target)
    mean, std = source_stats(source_x)
    source_y = base.wear_labels(raw_root, source)
    traces = {}
    for method in METHODS:
        traces[method] = train(method, source_x, source_y,
                               target_x if method == "daregram" else None,
                               mean, std, seed, folder, device)
    predictions = {method: predict_windows(method, target_x, mean, std, seed, folder, device)
                   for method in METHODS}
    # All model choices and predictions are fixed before opening target labels.
    target_y = base.wear_labels(raw_root, target, evaluation_dir=folder)
    for method, values in predictions.items():
        pd.DataFrame({"cut_index": base.ALL_CUTS, "true_vb": target_y,
                      "pred_w25": values[:, 0], "pred_center": values[:, 1],
                      "pred_w75": values[:, 2], "pred_mean": values.mean(axis=1)}).to_csv(
                          folder / method / "predictions.csv", index=False, float_format="%.17g")
    write_json(folder / "config.json", {"source": source, "target": target, "seed": seed,
               "epochs": EPOCHS, "batch_cuts": BATCH_SIZE, "steps_per_epoch": 5,
               "optimizer": "Adam", "lr": LR, "source_loss": "MSE",
               "daregram_loss": "original inverse Gram loss with exp epoch tradeoff",
               "checkpoint_selection": "final epoch", "source_cuts": base.ALL_CUTS,
               "daregram_unlabeled_target_cuts": base.ALL_CUTS,
               "source_only_training_target_cuts": [], "target_labels_training_reads": 0,
               "window_fractions": list(POSITIONS), "window_length": WINDOW_LENGTH,
               "window_sampling": "one seeded choice per Cut per epoch; same choices for both methods",
               "source_zscore_mean": mean.tolist(), "source_zscore_std": std.tolist(),
               "normalization_fit": "source 315 Cuts x 3 windows only",
               "raw_root": str(raw_root.resolve()), "traces": traces})
    write_json(folder / "complete.json", {"source": source, "target": target, "seed": seed,
               "methods": list(METHODS), "prediction_cuts": 315})
    logging.info("Completed M %s->%s seed=%d", source, target, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique members of 42..46")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    args.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.out_root / "run.log", encoding="utf-8")])
    prepare_cache(args.raw_root, args.out_root)
    if args.prepare_only:
        return
    for seed in args.seeds:
        for source, target in PAIRS:
            run_pair(args.raw_root, args.out_root, source, target, seed, torch.device(args.device))


if __name__ == "__main__":
    main()
