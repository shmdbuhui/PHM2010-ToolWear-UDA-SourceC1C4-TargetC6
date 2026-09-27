"""Create a direction-explicit report from all audited paired experiment rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_dann_mmd_joint import BASELINES, METHODS, PAIRS, SEEDS


def fmt(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}"


def main():
    root = Path("artifacts/dann_mmd_joint_20260926")
    audit = json.loads((root / "output_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["checked_new_runs"] != 120:
        raise ValueError("Full audit must pass before reporting")
    per = pd.read_csv(root / "per_seed_metrics.csv")
    summary = pd.read_csv(root / "five_seed_summary.csv")
    deltas = pd.read_csv(root / "paired_deltas.csv")
    method_order = (*BASELINES, *METHODS[:4])
    if (len(per), len(summary), len(deltas)) != (180, 36, 360):
        raise ValueError("Expected complete 6 direction x 5 seed x 6 method report")
    lines = ["# DANN/MMD and DARE-GRAM joint comparison", "",
             "All metrics use one target prediction per cut, cut 1–315. Values are five-seed mean ± sample standard deviation.",
             "Source-only and DARE-GRAM are the frozen audited baseline checkpoints; the other four methods were trained under the same 50-epoch protocol.",
             "The final checkpoint was fixed by epoch count before target wear labels were opened. No target label selected a weight or checkpoint.",
             "", "## Six-direction results", "",
             "| Direction | Method | R² | MAE | RMSE |",
             "|---|---|---:|---:|---:|"]
    for source, target in PAIRS:
        for method in method_order:
            row = summary.loc[(summary.source == source) & (summary.target == target) &
                              (summary.method == method)].iloc[0]
            if row.n_seeds != 5:
                raise ValueError("Missing seed")
            lines.append(f"| {source.upper()}→{target.upper()} | {method} | "
                         f"{fmt(row.R2_mean,row.R2_sd)} | {fmt(row.MAE_mean,row.MAE_sd)} | "
                         f"{fmt(row.RMSE_mean,row.RMSE_sd)} |")
    lines.extend(["", "## Paired MAE differences", "",
                  "Each difference is computed against the **same seed and direction** before averaging. Negative ΔMAE is better. The count shows seeds with lower MAE than the reference.",
                  "", "| Direction | Method | vs source-only, ΔMAE | Better seeds | vs DARE-GRAM, ΔMAE | Better seeds |",
                  "|---|---|---:|---:|---:|---:|"])
    for source, target in PAIRS:
        for method in METHODS[:4]:
            found = {}
            for ref in BASELINES:
                block = deltas.loc[(deltas.source == source) & (deltas.target == target) &
                                   (deltas.method == method) & (deltas.reference == ref)].sort_values("seed")
                if block.seed.tolist() != list(SEEDS):
                    raise ValueError("Missing paired delta")
                found[ref] = (fmt(block.delta_MAE.mean(), block.delta_MAE.std(ddof=1)),
                              int((block.delta_MAE < 0).sum()))
            lines.append(f"| {source.upper()}→{target.upper()} | {method} | "
                         f"{found['source_only'][0]} | {found['source_only'][1]}/5 | "
                         f"{found['daregram'][0]} | {found['daregram'][1]}/5 |")
    lines.extend(["", "## Files", "",
                  "- `per_seed_metrics.csv`: every direction, seed, method and checkpoint SHA-256.",
                  "- `five_seed_summary.csv`: R², MAE, RMSE mean and sample standard deviation.",
                  "- `paired_deltas.csv`: every metric difference to same-seed source-only and DARE-GRAM.",
                  "- `new_runs/<direction>/seed_<seed>/<method>/`: final checkpoint, config, 50-epoch loss log, one-row-per-cut predictions and curve.",
                  "- `frozen_baseline_snapshot/`: locked baseline metric/config/log copies, per-cut predictions and full-cut curves.",
                  "- `PROTOCOL.md`: implementation audit, exact signal processing, losses, weights and commands.",
                  "- `smoke.json`, `gradient_loss_checks.json`, and `output_audit.json`: forward/backward, MMD/GRL, and complete output audits.",
                  "", "## Interpretation", "",
                  "The paired rows show direction-dependent gains and regressions. The three-loss combination was configured and smoke-checked only; it was not part of these 180 full runs.",
                  "The chosen adaptation weights were fixed before target results and were not tuned on target wear.", ""])
    path = root / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
