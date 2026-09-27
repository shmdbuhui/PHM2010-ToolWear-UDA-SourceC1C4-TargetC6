"""Render the completed custom comparison tables into its independent README."""

from __future__ import annotations

import pandas as pd

import run_inverse_gram_subspace as run


def metric(value, sd):
    return f"{value:.2f} ± {sd:.2f}"


def main():
    root = run.DEFAULT_ROOT
    audit = run.stage.json_read(root / "numerical_and_protocol_audit.json")
    if audit["runs"] != 30 or audit["batches"] != 7500 or audit["failure_files"]:
        raise ValueError("Numerical/protocol audit incomplete")
    summary = pd.read_csv(root / "six_direction_five_seed_summary.csv")
    delta = pd.read_csv(root / "paired_five_seed_deltas.csv")
    if len(summary) != 36 or len(delta) != 24:
        raise ValueError("Incomplete six-direction comparison")
    labels = {"source_only": "source-only raw", "daregram": "DARE-GRAM raw",
              run.METHOD: "inverse_gram_subspace raw",
              "source_only_oor_pga": "source-only + OOR-PGA",
              "daregram_oor_pga": "DARE-GRAM + OOR-PGA",
              run.METHOD + "_oor_pga": "inverse_gram_subspace + OOR-PGA"}
    order = list(labels)
    lines = ["## Completed six-direction results", "",
             "Five seeds 42–46; target cuts 1–315. Entries are seed mean ± sample standard deviation. "
             "The source-only and DARE-GRAM entries are read from the frozen full-cut results, not retrained.", "",
             "| Direction | Method | R² | MAE | RMSE |",
             "|---|---|---:|---:|---:|"]
    for source, target in run.baseline.PAIRS:
        for method in order:
            row = summary[(summary.source == source) & (summary.target == target) &
                          (summary.method == method)]
            if len(row) != 1:
                raise ValueError(f"Missing result {source}->{target} {method}")
            row = row.iloc[0]
            lines.append(f"| {source.upper()}→{target.upper()} | {labels[method]} | "
                         f"{metric(row.R2_mean, row.R2_sd)} | "
                         f"{metric(row.MAE_mean, row.MAE_sd)} | "
                         f"{metric(row.RMSE_mean, row.RMSE_sd)} |")
    lines += ["", "## New method minus DARE-GRAM", "",
              "Positive ΔR² and negative ΔMAE/ΔRMSE favor the new method. "
              "These are paired seed deltas, summarized as mean ± sample standard deviation.", "",
              "| Direction | Stage | ΔR² | ΔMAE | ΔRMSE |",
              "|---|---|---:|---:|---:|"]
    raw_worse, corrected_worse = [], []
    for source, target in run.baseline.PAIRS:
        for stage_name in ("raw", "oor_pga"):
            row = delta[(delta.source == source) & (delta.target == target) &
                        (delta.stage == stage_name) & (delta.reference == "daregram")]
            if len(row) != 1:
                raise ValueError(f"Missing paired delta {source}->{target} {stage_name}")
            row = row.iloc[0]
            lines.append(f"| {source.upper()}→{target.upper()} | {stage_name} | "
                         f"{metric(row.delta_R2_mean, row.delta_R2_sd)} | "
                         f"{metric(row.delta_MAE_mean, row.delta_MAE_sd)} | "
                         f"{metric(row.delta_RMSE_mean, row.delta_RMSE_sd)} |")
            if row.delta_RMSE_mean > 0:
                (raw_worse if stage_name == "raw" else corrected_worse).append(
                    f"{source.upper()}→{target.upper()}")
    lines += ["", "## Interpretation and numerical audit", "",
              f"- Raw RMSE worse than DARE-GRAM: {', '.join(raw_worse) or 'none'}.",
              f"- After the same OOR-PGA, RMSE worse than DARE-GRAM + OOR-PGA: "
              f"{', '.join(corrected_worse) or 'none'}.",
              "- The fixed OOR-PGA has early triggers and a large multiplicative effect in "
              "C1→C4, C4→C1 and C6→C4. Their corrected errors are large for every method; "
              "relative improvement over DARE-GRAM does not imply a useful corrected prediction.",
              f"- {audit['runs']} runs, {audit['batches']} batches, "
              f"{audit['real_batch_preflights']} real-batch preflights, "
              f"{audit['failure_files']} numerical failures. All losses, gradients, selected "
              "singular values and epsilon values were finite. The observed k values were "
              f"{audit['k_values']}.",
              "- Full diagnostic ranges and provenance checks: `numerical_and_protocol_audit.json`. "
              "Per-seed comparisons, including seed-level regressions: `paired_per_seed_deltas.csv`.",
              ""]
    path = root / "README.md"
    start = path.read_text(encoding="utf-8").split("## Completed six-direction results")[0]
    path.write_text(start + "\n".join(lines), encoding="utf-8")
    print("Wrote", path)


if __name__ == "__main__":
    main()
