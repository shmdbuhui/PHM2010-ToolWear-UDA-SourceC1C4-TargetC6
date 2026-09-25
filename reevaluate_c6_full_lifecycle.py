"""Re-evaluate existing target-C6 checkpoints on cuts 1..315, without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import run_five_seed_pairs as five
import run_single_source_pairs as base
import run_norm_comparison as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-method", choices=comparison.NORMS, default="zscore")
    parser.add_argument("--out-root", type=Path, default=comparison.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cuts, definition = comparison.evaluation_cuts("c6", Path("unused"))
    assert cuts == base.ALL_CUTS
    manifest = json.loads((comparison.DEFAULT_CACHE / "manifest.json").read_text(encoding="utf-8"))
    raw_target = comparison.read_cache(comparison.DEFAULT_CACHE, manifest, "c6")
    for source in ("c1", "c4"):
        for seed in five.SEEDS:
            pair_dir = args.out_root / args.norm_method / f"{source}_to_c6" / f"seed_{seed}"
            if not (pair_dir / "complete.json").is_file():
                raise FileNotFoundError(f"No completed checkpoint pair: {pair_dir}")
            common = json.loads((pair_dir / "config.json").read_text(encoding="utf-8"))
            if common["norm_method"] != args.norm_method or common["source"] != source or common["target"] != "c6":
                raise ValueError(f"Unexpected training configuration: {pair_dir}")
            params = {key: np.asarray(value, dtype=np.float32)
                      for key, value in common["source_normalization_parameters"].items()}
            x_target = comparison.normalize(raw_target, params, args.norm_method)
            predictions = {method: five.predict(method, x_target, cuts, seed, pair_dir, torch.device(args.device))
                           for method in five.METHODS}
            # The existing wear-label gate confirms both final checkpoints before any target label read.
            target_labels = base.wear_labels(Path(common["raw_root"]), "c6", evaluation_dir=pair_dir)
            y_true = target_labels.astype(np.float64)
            results = {}
            for method in five.METHODS:
                folder = pair_dir / method
                pd.DataFrame({"cut_index": cuts, "true_vb": y_true, "pred_vb": predictions[method]}).to_csv(
                    folder / "predictions.csv", index=False, float_format="%.17g")
                results[method] = five.metrics_from_csv(folder / "predictions.csv", cuts)
                five.json_write(folder / "metrics.json", results[method])
                fig, ax = base.plt.subplots(figsize=(10, 4.5))
                ax.plot(cuts, y_true, label="True VB")
                ax.plot(cuts, predictions[method], label="Predicted VB")
                ax.set(xlabel="C6 cut index", ylabel="VB",
                       title=f"{source.upper()}->C6 seed {seed}: {method}")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(folder / "prediction.png", dpi=200)
                base.plt.close(fig)
                config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                config["evaluation_cuts"] = cuts
                config["evaluation_cut_definition"] = definition
                five.json_write(folder / "config.json", config)
            common["evaluation_cuts"] = cuts
            common["evaluation_cut_definition"] = definition
            five.json_write(pair_dir / "config.json", common)
            five.json_write(pair_dir / "comparison.json", results)
            print(f"Re-evaluated {args.norm_method} {source}->c6 seed={seed} on 315 cuts", flush=True)
    comparison.summarize(SimpleNamespace(out_root=args.out_root))


if __name__ == "__main__":
    main()
