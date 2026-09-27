"""Read-only audit of frozen five-seed baseline and multiplicative STFT-48 OOR-PGA.

Run from upstream-reproduction: python audit_five_seed_oor_pga.py
No training, inference, or correction is performed. Existing paired predictions are checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
BASE = ART / "full_1_315_baseline_zscore_20260925"
OOR = ART / "oor_pga_full_1_315_zscore_20260925"
OLD = ART / "five_seed_paired"
COD = ART / "cod_comparison_20260926"
ALTERNATIVE = ART / "early_reference_persistent_oor_pga_full_1_315_20260925"
SEEDS = range(42, 47)
METHODS = ("source_only", "daregram")
CUTS = np.arange(1, 316)


def measure(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    sst = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "R2": 1 - np.sum(e * e) / sst if sst > 1e-12 else np.nan,
            "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e * e)),
            "MAPE_percent": np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-8)) * 100,
            "signed_error": e.mean()}


def check_cuts(frame: pd.DataFrame, path: Path, expected: np.ndarray = CUTS) -> None:
    if not np.array_equal(frame.cut_index.to_numpy(), expected):
        raise ValueError(f"Unordered/missing/duplicate cuts in {path}")


def plot_group(out: Path, direction: str, method: str, rows: pd.DataFrame) -> None:
    arrays = [rows[rows.seed == seed].sort_values("cut_index") for seed in SEEDS]
    y = arrays[0].true_vb.to_numpy(float)
    base = np.stack([a.base_pred_vb.to_numpy(float) for a in arrays])
    corrected = np.stack([a.corrected_pred_vb.to_numpy(float) for a in arrays])
    delta = corrected - base
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for i, seed in enumerate(SEEDS):
        color = plt.cm.tab10(i)
        axes[0].plot(CUTS, base[i], color=color, alpha=.55, lw=.85, label=f"seed {seed} base")
        axes[0].plot(CUTS, corrected[i], color=color, alpha=.75, lw=.85, ls="--", label=f"seed {seed} corrected")
        axes[1].plot(CUTS, base[i] - y, color=color, alpha=.5, lw=.8)
        axes[1].plot(CUTS, corrected[i] - y, color=color, alpha=.7, lw=.8, ls="--")
        axes[2].plot(CUTS, delta[i], color=color, alpha=.6, lw=.8)
    for ax, matrix, color, name in ((axes[0], base, "#174a7e", "base"),
                                    (axes[0], corrected, "#bd4f2b", "corrected"),
                                    (axes[1], base-y, "#174a7e", "base error"),
                                    (axes[1], corrected-y, "#bd4f2b", "corrected error"),
                                    (axes[2], delta, "#753f93", "correction")):
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        ax.plot(CUTS, mean, color=color, lw=2, label=f"{name} mean")
        ax.fill_between(CUTS, mean-sd, mean+sd, color=color, alpha=.13, label=f"{name} ±1 SD")
    axes[0].plot(CUTS, y, color="black", lw=1.4, label="true VB")
    axes[1].axhline(0, color="black", lw=.7)
    trigger = int(arrays[0].trigger_cut.iloc[0])
    for ax in axes:
        ax.axvline(trigger, color="gray", ls=":", lw=1, label=f"first trigger {trigger}")
        ax.grid(alpha=.2)
        ax.set_xlim(1, 315)
    axes[0].set_ylabel("prediction / true VB (µm)")
    axes[1].set_ylabel("pred − true (µm)")
    axes[2].set_ylabel("corrected − base (µm)")
    axes[2].set_xlabel("target cut index, 1–315")
    axes[0].legend(ncol=4, fontsize=7, loc="upper left")
    axes[2].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{direction.upper()} | {method} | seeds 42–46 | full 1–315 | STFT-48 OOR-PGA")
    fig.savefig(out / f"{direction}_{method}.png", dpi=155)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ART / "five_seed_oor_pga_audit_20260926")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite: {out}")
    audit_path = BASE / "checkpoint_audit_full_1_315.csv"
    audit = pd.read_csv(audit_path)
    published_base = pd.read_csv(BASE / "per_seed_metrics_full_1_315.csv")
    published_oor = pd.read_csv(OOR / "per_seed_metrics_full_1_315.csv")
    triggers = pd.read_csv(OOR / "triggers_and_factors_full_1_315.csv")
    lock = json.loads((OOR / "prediction_lock_before_target_labels.json").read_text(encoding="utf-8"))
    if lock.get("target_labels_opened_in_prepare") is not False:
        raise ValueError("OOR label separation lock missing")
    manifest, metric_rows, cut_rows, discrepancies = [], [], [], []
    for rec in audit.itertuples(index=False):
        source, target, seed, method = rec.source, rec.target, int(rec.seed), rec.method
        direction = f"{source}_to_{target}"
        original = Path(rec.original_prediction_file)
        corrected_file = OOR / "unlabeled_predictions" / f"{direction}_seed_{seed}_full_1_315.csv"
        old_file = OLD / direction / f"seed_{seed}" / method / "predictions.csv"
        config_file = Path(rec.training_config)
        checkpoint = Path(rec.checkpoint)
        files = (original, corrected_file, old_file, config_file, checkpoint)
        if not all(p.exists() for p in files):
            raise FileNotFoundError([str(p) for p in files if not p.exists()])
        b, c, old = (pd.read_csv(p) for p in (original, corrected_file, old_file))
        check_cuts(b, original)
        check_cuts(c, corrected_file)
        old_cuts = old.cut_index.to_numpy()
        if not np.array_equal(old_cuts, np.arange(95,316) if target == "c6" else CUTS):
            raise ValueError(f"Unexpected historical evaluation scope: {old_file}")
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
        if (cfg["source"], cfg["target"], cfg["seed"], cfg["method"]) != (source,target,seed,method):
            raise ValueError(f"Config identity mismatch: {config_file}")
        col = f"{method}_pred_vb"
        cor = f"{method}_oor_pga_pred_vb"
        y, p0, p1 = b.true_vb.to_numpy(float), b.pred_vb.to_numpy(float), c[cor].to_numpy(float)
        raw = Path(rec.target_wear_file)
        wear = pd.read_csv(raw)
        if wear.cut.astype(int).tolist() != CUTS.tolist():
            raise ValueError(f"Raw wear cut mapping mismatch: {raw}")
        raw_y = wear[["flute_1","flute_2","flute_3"]].mean(axis=1).to_numpy(np.float32).astype(float)
        if not np.allclose(y,raw_y,rtol=0,atol=1e-6):
            raise ValueError(f"Prediction labels differ from raw flute mean: {original}")
        if not np.isfinite(np.stack([y,p0,p1])).all() or not np.allclose(p0,c[col],rtol=0,atol=1e-10):
            raise ValueError(f"Base/corrected pairing mismatch: {direction} {seed} {method}")
        t = triggers[(triggers.source == source) & (triggers.target == target)]
        if len(t) != 1 or t.trigger_status.iloc[0] != "triggered":
            raise ValueError(f"Trigger record missing: {direction}")
        first = int(t.trigger_cut.iloc[0])
        derived_hits = np.flatnonzero((c.tau.to_numpy(float)>0)&(c.gate.to_numpy(float)>=1-1e-12))
        if not len(derived_hits) or first != int(derived_hits[0]+1):
            raise ValueError(f"First trigger does not follow gate rule: {direction}")
        factor = c.multiplicative_factor.to_numpy(float)
        expected_factor = np.ones(315)
        expected_factor[first:] = (c.tau.to_numpy(float)[first:] / float(t.t_oor.iloc[0])) ** float(t.m.iloc[0])
        if not np.allclose(factor,expected_factor,rtol=1e-12,atol=1e-12):
            raise ValueError(f"Factor does not follow saved source exponent: {direction}")
        if not np.allclose(p1,p0*factor,rtol=1e-12,atol=1e-9):
            raise ValueError(f"Correction rule mismatch: {direction} {seed} {method}")
        if not (np.all(factor[:first] == 1) and np.all(factor[first:] >= 1)):
            raise ValueError(f"Factor timing mismatch: {direction}")
        for stage, pred, processed in (("base",p0,False),("oor_pga",p1,True)):
            manifest.append({"direction":direction,"method":method,"seed":seed,"stage":stage,
                             "path":str(original if stage=="base" else corrected_file),
                             "checkpoint":str(checkpoint),"config":str(config_file),
                             "oor_score_path":str(OOR/"oor_scores"/f"{direction}_oor_score_full_1_315.csv"),
                             "oor_lock_path":str(OOR/"prediction_lock_before_target_labels.json"),
                             "cut_first":1,"cut_last":315,"n_cuts":315,"unit":"VB µm",
                             "postprocessed":processed,"pairable":True,"reason":"same checkpoint/seed/cuts; stored base matches",
                             "protocol":"unlabeled full_1_315"})
            published = published_base[(published_base.source==source)&(published_base.target==target)&
                                       (published_base.seed==seed)&(published_base.method==method)] if stage=="base" else \
                        published_oor[(published_oor.source==source)&(published_oor.target==target)&
                                      (published_oor.seed==seed)&(published_oor.method==method+"_oor_pga")&
                                      (published_oor.segment=="full")]
            calc = measure(y,pred)
            if len(published) != 1:
                raise ValueError(f"Published metric missing: {direction} {seed} {method} {stage}")
            for key in ("R2","MAE","RMSE","MAPE_percent"):
                if abs(calc[key] - float(published.iloc[0][key])) > 1e-8:
                    discrepancies.append((direction,seed,method,stage,key,calc[key],published.iloc[0][key]))
        manifest.append({"direction":direction,"method":method,"seed":seed,"stage":"historical_base",
                         "path":str(old_file),"checkpoint":str(checkpoint),"config":str(config_file),
                         "cut_first":int(old_cuts[0]),"cut_last":int(old_cuts[-1]),"n_cuts":len(old_cuts),
                         "unit":"VB µm","postprocessed":False,"pairable":False,
                         "reason":"old C6 suffix 95–315" if target=="c6" else "historical duplicate; full pair above",
                         "protocol":"old_95_315" if target=="c6" else "historical full_1_315"})
        gate = c.gate.to_numpy(float)
        phase = np.where(CUTS<first,"before_trigger",np.where(CUTS==first,"trigger_cut","after_trigger"))
        delta = p1-p0
        for i, cut in enumerate(CUTS):
            cut_rows.append({"direction":direction,"method":method,"seed":seed,"cut_index":cut,
                             "true_vb":y[i],"base_pred_vb":p0[i],"corrected_pred_vb":p1[i],
                             "base_signed_error":p0[i]-y[i],"corrected_signed_error":p1[i]-y[i],
                             "correction_vb":delta[i],"gate":gate[i],"factor":factor[i],
                             "trigger_cut":first,"phase":phase[i],"gate_fully_open":gate[i]>=1-1e-12})
        masks = {"full":np.ones(315,bool),"before_trigger":CUTS<first,"trigger_cut":CUTS==first,
                 "after_trigger":CUTS>first,"gate_not_fully_open":gate<1-1e-12,
                 "late_250_285":(CUTS>=250)&(CUTS<=285),"late_286_315":CUTS>=286}
        for segment, mask in masks.items():
            if not mask.any(): continue
            v0,v1=measure(y[mask],p0[mask]),measure(y[mask],p1[mask])
            row={"direction":direction,"method":method,"seed":seed,"segment":segment,
                 "cut_first":int(CUTS[mask].min()),"cut_last":int(CUTS[mask].max()),"n":int(mask.sum()),
                 "trigger_cut":first,"trigger_total":int(len(derived_hits)),
                 "corrected_cut_count":int((CUTS>first).sum()),
                 "base_max_vb":p0.max(),"base_min_vb":p0.min(),"base_range_vb":np.ptp(p0),
                 "correction_mean_vb":delta.mean(),"correction_max_vb":delta.max(),
                 "selected_checkpoint_epoch":int(cfg["epochs"]),"best_checkpoint_epoch":np.nan,
                 "source_validation_metric":np.nan}
            for k in ("R2","MAE","RMSE","MAPE_percent","signed_error"):
                row[f"base_{k}"]=v0[k]; row[f"corrected_{k}"]=v1[k]
                row[f"delta_{k}"]=v1[k]-v0[k]
            metric_rows.append(row)
    if discrepancies:
        raise ValueError(f"Independent metric mismatch: {discrepancies[:8]}")
    # COD has full-cut predictions, but no saved OOR-PGA correction under the fixed rule.
    for direction in sorted(p.name for p in COD.iterdir() if p.is_dir() and "_to_" in p.name):
        for seed in SEEDS:
            p=COD/direction/f"seed_{seed}"/"daregram_cod"/"predictions.csv"
            if p.exists():
                f=pd.read_csv(p); check_cuts(f,p)
                manifest.append({"direction":direction,"method":"cod","seed":seed,"stage":"base_only",
                                 "path":str(p),"checkpoint":"","config":"","cut_first":1,"cut_last":315,
                                 "n_cuts":315,"unit":"VB µm","postprocessed":False,"pairable":False,
                                 "reason":"no fixed-rule COD OOR-PGA output", "protocol":"full_1_315 COD"})
    # Another published full-cut OOR variant uses different gate/factor parameters.
    # Inventory it, but never substitute it for the fixed original OOR-PGA pairing.
    for p in sorted((ALTERNATIVE/"unlabeled_predictions").glob("*.csv")):
        f=pd.read_csv(p,usecols=["cut_index","source","target","seed","method"])
        check_cuts(f,p)
        manifest.append({"direction":f"{f.source.iloc[0]}_to_{f.target.iloc[0]}",
                         "method":f.method.iloc[0],"seed":int(f.seed.iloc[0]),
                         "stage":"alternative_oor_variant","path":str(p),
                         "checkpoint":"","config":str(ALTERNATIVE/"prediction_lock_before_target_labels.json"),
                         "cut_first":1,"cut_last":315,"n_cuts":315,"unit":"VB µm",
                         "postprocessed":True,"pairable":False,
                         "reason":"early-reference persistent variant has different gate/factor rule",
                         "protocol":"alternative full_1_315"})
    out.mkdir(parents=True)
    plots=out/"figures"; plots.mkdir()
    man=pd.DataFrame(manifest); met=pd.DataFrame(metric_rows); cuts=pd.DataFrame(cut_rows)
    man.to_csv(out/"artifact_manifest.csv",index=False)
    met.to_csv(out/"seed_paired_metrics.csv",index=False)
    cuts.to_csv(out/"per_cut_audit.csv",index=False,float_format="%.17g")
    for (direction,method), group in cuts.groupby(["direction","method"]):
        plot_group(plots,direction,method,group)
    summary=[]
    for (direction,method),g in met[met.segment=="full"].groupby(["direction","method"]):
        row={"direction":direction,"method":method}
        for k in ("RMSE","MAE","R2","MAPE_percent"):
            d=g[f"delta_{k}"].to_numpy(float)
            row[f"delta_{k}_mean"]=d.mean(); row[f"delta_{k}_sd"]=d.std(ddof=1)
            row[f"improved_{k}_seeds"]=int((d>0).sum() if k=="R2" else (d<0).sum())
        for stage,col in (("base","base_pred_vb"),("correction","correction_vb")):
            matrix=np.stack([cuts[(cuts.direction==direction)&(cuts.method==method)&(cuts.seed==s)].sort_values("cut_index")[col].to_numpy(float) for s in SEEDS])
            row[f"{stage}_mean_cutwise_seed_sd_vb"]=matrix.std(axis=0,ddof=1).mean()
            row[f"{stage}_late_286_315_seed_sd_vb"]=matrix[:,285:].std(axis=0,ddof=1).mean()
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/"paired_five_seed_summary.csv",index=False)
    c16=pd.DataFrame(summary); c16=c16[c16.direction=="c1_to_c6"]
    lines=["# Five-seed baseline vs OOR-PGA audit", "",
           "Run: `cd upstream-reproduction; python audit_five_seed_oor_pga.py --out artifacts/<new-directory>`.",
           "The script refuses to overwrite an output directory. It reads frozen baseline checkpoint/provenance, 1–315 predictions, stored OOR factors and paired corrected predictions; no training or new correction.",
           "", "## Protocol", "",
           "Five seeds 42–46, six transfer directions, source-only and DARE-GRAM, paired by direction/method/seed/cut/checkpoint. C6 historical base predictions cover 95–315 and appear separately in the manifest; no matching 95–315 original OOR output was located. The full 1–315 C6 predictions were regenerated previously from identical frozen checkpoints. The early-reference persistent OOR variant covers 1–315 but has a different rule and is inventoried only. COD has no corresponding stored original-rule OOR-PGA correction and is excluded. The C6 five-true-label calibration experiments are excluded.",
           "", "Each cut is one model inference on its STFT tensor; no window mean is applied at prediction time. Target VB is the float32 mean of the three flute labels, indexed by cut. Predictions preserve ascending cut order. Stored correction equals base prediction times the source-fitted factor; first tau>0 full gate at cut 161 for C1→C6, with correction starting at cut 162. The gate can subsequently close, but the factor remains active. `gate_not_fully_open` overlaps the chronological phases.",
           "", "MAPE = mean(|pred−true| / max(|true|, 1e−8)) × 100; R² on a single trigger cut is undefined. All published full-cut R², MAE, RMSE and MAPE match independent recomputation within 1e−8.",
           "", "## C1→C6 paired full-cut results", "",
           "| Method | ΔRMSE mean ± SD (µm) | improved seeds | base cutwise seed SD | correction cutwise seed SD |",
           "|---|---:|---:|---:|---:|"]
    for r in c16.itertuples():
        lines.append(f"| {r.method} | {r.delta_RMSE_mean:.3f} ± {r.delta_RMSE_sd:.3f} | {r.improved_RMSE_seeds}/5 | {r.base_mean_cutwise_seed_sd_vb:.3f} | {r.correction_mean_cutwise_seed_sd_vb:.3f} |")
    lines += ["", "Five-seed mean segment metrics (µm):", "",
              "| Method | Segment | base→corrected MAE | base→corrected signed error |",
              "|---|---|---:|---:|"]
    for method in METHODS:
        for segment in ("before_trigger","after_trigger","late_250_285","late_286_315"):
            g=met[(met.direction=="c1_to_c6")&(met.method==method)&(met.segment==segment)]
            lines.append(f"| {method} | {segment} | {g.base_MAE.mean():.2f}→{g.corrected_MAE.mean():.2f} | {g.base_signed_error.mean():+.2f}→{g.corrected_signed_error.mean():+.2f} |")
    lines += ["", "## Interpretation and missing data", "",
              "For C1→C6 the trigger and factor are shared across all seeds, so there is no observed seed-to-seed trigger timing variation. Seed differences exist in the frozen base predictions; correction differences inherit those base differences through multiplication. Full and late-segment paired errors are in `seed_paired_metrics.csv`; use these with the plots to judge direction and size. This supports prioritizing base training/prediction stability for the C1→C6 five-seed spread. It does not prove which training mechanism caused that spread. Across directions the fixed OOR rule may be harmful and deserves separate protocol assessment.",
              "", "The training logs report source training MSE, not a source validation metric. Checkpoints are selected at final epoch 50; no best-validation epoch exists, so these fields are missing. The STFT-48 thresholds 3/48 and 6/48 were transferred from a different feature set, not selected by target labels in the `prepare` code; this transfer is unvalidated. Target wear labels enter only the separate evaluation stage here. These claims cover this frozen OOR-PGA implementation and its provenance lock, not every historical exploratory analysis.",
              "", "Outputs: `artifact_manifest.csv`, `seed_paired_metrics.csv`, `paired_five_seed_summary.csv`, `per_cut_audit.csv`, `figures/*.png`."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {out}; {len(man)} manifest rows, {len(met)} segment rows, {len(cuts)} per-cut rows")


if __name__ == "__main__":
    main()
