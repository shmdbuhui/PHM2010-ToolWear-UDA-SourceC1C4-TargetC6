"""Compare source-fitted channel Z-score and Min-Max on the cached PHM2010 STFTs.

Training, model, prediction, and evaluation are the existing five-seed implementations.
Only the input normalization is selected here. Each norm gets fresh models and Adam
optimizers through five_seed.train(), including independent BatchNorm buffers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_five_seed_pairs as five
import run_single_source_pairs as base


DEFAULT_CACHE = Path("artifacts/five_seed_paired/feature_cache")
DEFAULT_OUTPUT = Path("artifacts/norm_comparison_20260925")
NORMS = ("zscore", "minmax")


def evaluation_cuts(target, split_record):
    """Use the user's full-lifecycle C6 comparison while retaining metric code."""
    if target == "c6":
        return base.ALL_CUTS.copy(), "requested full-lifecycle C6 target evaluation: 1..315"
    return base.evaluation_cuts(target, split_record)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-method", choices=NORMS, default="zscore")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\QLP\source\source_mill"))
    parser.add_argument("--split-record", type=Path, default=Path(r"E:\QLP\test-9.3\outputs\predictions_comparison.csv"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preflight", action="store_true", help="Check C1->C4 seed 42 without training")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.out_root.resolve() == Path("artifacts/five_seed_paired").resolve():
        parser.error("Comparison output must not overwrite the previous run")
    return args


def read_cache(cache_root, manifest, tool):
    path = cache_root / f"{tool}_stft.npy"
    if base.file_hash(path) != manifest["tools"][tool]["feature_sha256"]:
        raise ValueError(f"Feature cache checksum mismatch: {path}")
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (315, 6, 128, 128) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid cached STFT: {path}")
    return raw


def source_parameters(source_cache, norm_method):
    if norm_method == "zscore":
        mean = source_cache.mean(axis=(0, 2, 3)).astype(np.float32)
        std = (source_cache.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        return {"mean": mean, "std": std}
    x_min = source_cache.min(axis=(0, 2, 3)).astype(np.float32)
    x_max = source_cache.max(axis=(0, 2, 3)).astype(np.float32)
    return {"min": x_min, "max": x_max}


def normalize(raw, parameters, norm_method):
    if norm_method == "zscore":
        mean, std = parameters["mean"], parameters["std"]
        values = (raw - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-8)
    else:
        x_min, x_max = parameters["min"], parameters["max"]
        values = (raw - x_min[None, :, None, None]) / np.maximum(
            x_max - x_min, 1e-8)[None, :, None, None]
    values = values.astype(np.float32)
    if values.shape != (315, 6, 128, 128) or not np.isfinite(values).all():
        raise ValueError("Normalization produced invalid model inputs")
    return values


def input_audit(values):
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "min": float(values.min()), "max": float(values.max()),
            "channel_min": values.min(axis=(0, 2, 3)).tolist(),
            "channel_max": values.max(axis=(0, 2, 3)).tolist(),
            "outside_0_1_fraction": float(np.count_nonzero((values < 0) | (values > 1)) / values.size),
            "finite": bool(np.isfinite(values).all())}


def preflight(args, manifest):
    source = read_cache(args.cache_root, manifest, "c1")
    target = read_cache(args.cache_root, manifest, "c4")
    report = {"direction": "c1_to_c4", "seed": 42, "checks": {}, "methods": {}}
    inputs = {}
    for norm in NORMS:
        params = source_parameters(source, norm)
        xs, xt = normalize(source, params, norm), normalize(target, params, norm)
        model, initial_hash = base.build_model(42, torch.device(args.device))
        captured = []
        hook = model.feature_extractor.backbone.conv1.register_forward_pre_hook(
            lambda _module, tensors: captured.append(tensors[0].detach().cpu().numpy().copy()))
        model.eval()
        with torch.no_grad():
            output = model.feature_extractor(torch.from_numpy(xt[:1].copy()).to(args.device))
        hook.remove()
        conv_input = captured[0]
        if conv_input.shape != (1, 6, 128, 128) or not np.isfinite(conv_input).all() or not torch.isfinite(output).all():
            raise ValueError(f"Invalid {norm} preflight input or forward output")
        inputs[norm] = conv_input
        report["methods"][norm] = {"source_parameters": {k: v.tolist() for k, v in params.items()},
                                   "source_input": input_audit(xs), "target_input": input_audit(xt),
                                   "conv1_pre_input_shape": list(conv_input.shape),
                                   "conv1_pre_input_finite": True, "forward_output_finite": True,
                                   "initial_model_sha256": initial_hash}
        del xs, xt, model
    report["checks"] = {"same_conv1_input_shape": inputs["zscore"].shape == inputs["minmax"].shape,
                        "different_conv1_input": not np.array_equal(inputs["zscore"], inputs["minmax"]),
                        "same_initial_model_weights": report["methods"]["zscore"]["initial_model_sha256"] ==
                        report["methods"]["minmax"]["initial_model_sha256"]}
    if not all(report["checks"].values()):
        raise ValueError(f"Preflight failed: {report['checks']}")
    five.json_write(args.out_root / "preflight_c1_to_c4_seed42.json", report)
    logging.info("Preflight passed: %s", report["checks"])


def run_pair(args, manifest, source, target, seed):
    norm = args.norm_method
    folder = args.out_root / norm / f"{source}_to_{target}" / f"seed_{seed}"
    if folder.exists():
        completed = folder / "complete.json"
        if not completed.is_file():
            raise FileExistsError(f"Incomplete result requires inspection: {folder}")
        saved = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if (saved["norm_method"], saved["source"], saved["target"], saved["seed"]) != (norm, source, target, seed):
            raise ValueError(f"Existing run has a different configuration: {folder}")
        for method in five.METHODS:
            five.metrics_from_csv(folder / method / "predictions.csv", saved["evaluation_cuts"])
        logging.info("Verified completed result: %s", folder)
        return
    folder.mkdir(parents=True)
    for method in five.METHODS:
        (folder / method).mkdir()
    handler = logging.FileHandler(folder / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        device = torch.device(args.device)
        cuts, cut_definition = evaluation_cuts(target, args.split_record)
        raw_source = read_cache(args.cache_root, manifest, source)
        parameters = source_parameters(raw_source, norm)
        x_source = normalize(raw_source, parameters, norm)
        source_audit = input_audit(x_source)
        del raw_source
        y_source = base.wear_labels(args.raw_root, source)
        if len(y_source) != 315:
            raise ValueError("Source label count must be 315")
        common = {"norm_method": norm, "normalization_source": source,
                  "normalization_axis": [0, 2, 3],
                  "source_normalization_parameters": {k: v.tolist() for k, v in parameters.items()},
                  "source_model_input": source_audit,
                  "source": source, "target": target, "seed": seed, "epochs": five.EPOCHS,
                  "batch_size": five.BATCH_SIZE, "lr": five.LR, "device": str(device),
                  "raw_root": str(args.raw_root.resolve()), "split_record": str(args.split_record.resolve()),
                  "cache_root": str(args.cache_root.resolve()),
                  "stft_code_sha256": manifest["stft_code_sha256"],
                  "source_feature_sha256": manifest["tools"][source]["feature_sha256"],
                  "target_feature_sha256": manifest["tools"][target]["feature_sha256"],
                  "source_wear_sha256": base.file_hash(args.raw_root / f"{source}_wear.csv"),
                  "source_count": 315, "source_cuts": base.ALL_CUTS,
                  "target_unlabeled_cuts": base.ALL_CUTS, "evaluation_cuts": cuts,
                  "evaluation_cut_definition": cut_definition,
                  "input": "center 4096; 6 channels; STFT 256/224; log1p magnitude; 128x128",
                  "backbone": "ResNet18", "regressor": "Linear(512, 1)", "optimizer": "Adam",
                  "scheduler": "fixed", "source_loss": "MSE",
                  "daregram_loss": "original inverse Gram loss with exp epoch tradeoff",
                  "checkpoint_selection": "final epoch", "protocol": "transductive UDA for DARE-GRAM",
                  "target_labels_training_reads": 0}
        if target == "c6":
            common["split_record_sha256"] = base.file_hash(args.split_record)
        logging.info("Start norm=%s %s->%s seed=%d", norm, source, target, seed)
        source_trace = five.train("source_only", x_source, y_source, None, source, target, seed, folder, device)
        raw_target = read_cache(args.cache_root, manifest, target)
        x_target = normalize(raw_target, parameters, norm)
        common["target_model_input"] = input_audit(x_target)
        common["target_outside_0_1_fraction"] = common["target_model_input"]["outside_0_1_fraction"]
        del raw_target
        dare_trace = five.train("daregram", x_source, y_source, x_target, source, target, seed, folder, device)
        if source_trace["initial_model_sha256"] != dare_trace["initial_model_sha256"] or \
                source_trace["source_order_sha256_by_epoch"] != dare_trace["source_order_sha256_by_epoch"] or \
                source_trace["actual_unlabeled_target_cuts"] != [] or \
                dare_trace["actual_unlabeled_target_cuts"] != base.ALL_CUTS:
            raise ValueError("Training protocol audit failed")
        predictions = {method: five.predict(method, x_target, cuts, seed, folder, device)
                       for method in five.METHODS}
        target_labels = base.wear_labels(args.raw_root, target, evaluation_dir=folder)
        y_true = target_labels[np.asarray(cuts) - 1].astype(np.float64)
        results = {}
        for method in five.METHODS:
            method_dir = folder / method
            pd.DataFrame({"cut_index": cuts, "true_vb": y_true, "pred_vb": predictions[method]}).to_csv(
                method_dir / "predictions.csv", index=False, float_format="%.17g")
            results[method] = five.metrics_from_csv(method_dir / "predictions.csv", cuts)
            five.json_write(method_dir / "metrics.json", results[method])
            fig, ax = base.plt.subplots(figsize=(10, 4.5))
            ax.plot(cuts, y_true, label="True VB")
            ax.plot(cuts, predictions[method], label="Predicted VB")
            ax.set(xlabel=f"{target.upper()} cut index", ylabel="VB",
                   title=f"{source.upper()}->{target.upper()} seed {seed}: {method}")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(method_dir / "prediction.png", dpi=200)
            base.plt.close(fig)
            trace = source_trace if method == "source_only" else dare_trace
            five.json_write(method_dir / "config.json", dict(common, method=method,
                            initial_model_sha256=trace["initial_model_sha256"],
                            source_order_sha256_by_epoch=trace["source_order_sha256_by_epoch"],
                            source_order_sha256_all_epochs=trace["source_order_sha256_all_epochs"],
                            actual_unlabeled_target_cuts=trace["actual_unlabeled_target_cuts"],
                            checkpoint_sha256=trace["checkpoint_sha256"],
                            target_training_information="none" if method == "source_only" else "all 315 unlabeled STFT inputs"))
        common["initial_model_sha256"] = source_trace["initial_model_sha256"]
        common["source_order_sha256_all_epochs"] = source_trace["source_order_sha256_all_epochs"]
        five.json_write(folder / "config.json", common)
        five.json_write(folder / "audit.json", {"target_label_reads_during_training": 0,
                       "source_only_training_target_cuts": [], "daregram_training_target_cuts": base.ALL_CUTS,
                       "source_only_checkpoint_sha256": source_trace["checkpoint_sha256"],
                       "daregram_checkpoint_sha256": dare_trace["checkpoint_sha256"]})
        five.json_write(folder / "comparison.json", results)
        five.json_write(folder / "complete.json", {"norm_method": norm, "source": source,
                        "target": target, "seed": seed, "methods": list(five.METHODS)})
        logging.info("Completed norm=%s %s->%s seed=%d: %s", norm, source, target, seed, results)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def summarize(args):
    rows = []
    for norm in NORMS:
        for source, target in five.PAIRS:
            for seed in five.SEEDS:
                folder = args.out_root / norm / f"{source}_to_{target}" / f"seed_{seed}"
                if not (folder / "complete.json").is_file():
                    continue
                config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                for method in five.METHODS:
                    metrics = five.metrics_from_csv(folder / method / "predictions.csv", config["evaluation_cuts"])
                    rows.append({"norm_method": norm, "source": source, "target": target, "seed": seed,
                                 "method": method, "evaluation_count": len(config["evaluation_cuts"]), **metrics})
    pd.DataFrame(rows).to_csv(args.out_root / "seed_metrics.csv", index=False)
    summary = []
    for norm in NORMS:
        for source, target in five.PAIRS:
            for method in five.METHODS:
                subset = [r for r in rows if (r["norm_method"], r["source"], r["target"], r["method"]) ==
                          (norm, source, target, method)]
                if len(subset) != len(five.SEEDS):
                    continue
                summary.append({"norm_method": norm, "source": source, "target": target, "method": method,
                                "n": len(subset), "evaluation_count": subset[0]["evaluation_count"],
                                **{f"{metric}_{stat}": float(getattr(np.asarray([r[metric] for r in subset]), stat)(**(
                                    {"ddof": 1} if stat == "std" else {})))
                                   for metric in ("R2", "MAE", "RMSE") for stat in ("mean", "std")}})
    pd.DataFrame(summary).to_csv(args.out_root / "summary.csv", index=False)
    deltas = []
    for source, target in five.PAIRS:
        for method in five.METHODS:
            matched = {(r["norm_method"], r["seed"]): r for r in rows
                       if (r["source"], r["target"], r["method"]) == (source, target, method)}
            if not all((norm, seed) in matched for norm in NORMS for seed in five.SEEDS):
                continue
            item = {"source": source, "target": target, "method": method, "n": 5}
            for metric in ("R2", "MAE", "RMSE"):
                values = np.asarray([matched[("minmax", seed)][metric] - matched[("zscore", seed)][metric]
                                     for seed in five.SEEDS])
                item[f"minmax_minus_zscore_{metric}_mean"] = float(values.mean())
                item[f"minmax_minus_zscore_{metric}_std"] = float(values.std(ddof=1))
            deltas.append(item)
    pd.DataFrame(deltas).to_csv(args.out_root / "deltas.csv", index=False)
    five.json_write(args.out_root / "completion.json", {"completed_trainings": len(rows),
                    "required_trainings": 120, "complete": len(rows) == 120})


def main():
    args = arguments()
    args.out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_root / "batch.log", encoding="utf-8")])
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["raw_root"] != str(args.raw_root.resolve()) or \
            manifest["stft_code_sha256"] != base.file_hash(Path(base.sampling.__file__)):
        raise ValueError("STFT cache does not match raw root or preprocessing code")
    if args.preflight:
        preflight(args, manifest)
        return
    if not (args.out_root / "preflight_c1_to_c4_seed42.json").is_file():
        raise RuntimeError("Run --preflight before training")
    summarize(args)
    try:
        for source, target in five.PAIRS:
            for seed in five.SEEDS:
                run_pair(args, manifest, source, target, seed)
                summarize(args)
    finally:
        summarize(args)


if __name__ == "__main__":
    main()
