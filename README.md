# Unsupervised Domain Adaptation for Fault CNC Tool-Wear Prediction 

## Overview

A project for unsupervised domain adaptation (UDA) for CNC tool-wear prediction. This project performs domain adaptation between different tool conditions using the PHM2010 dataset.


## 📊 Dataset

### PHM2010 Dataset

The project uses the [PHM2010 dataset](https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010).

#### Data Structure

```
dataset/
 ├── c1/
 │    ├── c1_wear.csv          # Tool wear labels
 │    └── c1/                  # Sensor data
 │         ├── c_1_001.csv
 │         ├── c_1_002.csv
 │         ├── ...
 │         └── c_1_315.csv
 ├── c4/
 │    ├── c4_wear.csv
 │    └── c4/
 │         ├── c_4_001.csv
 │         ├── ...
 └── c6/
      ├── c6_wear.csv
      └── c6/
           ├── c6_001.csv
           ├── ...
```

#### Dataset Characteristics

- Number of Tools: 3 (c1, c4, c6)
- Files per Tool: 315
- Sample Length: 200,000 samples per file
- Sensor Channels: 6 channels (Fx, Fy, Fz, Vx, Vy, Vz)
- Window Length: 4096

## Setup

### Requirements

Our code runs fine with the following prerequisites:

- Python >= 3.12.2
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0
- pandas >= 1.2.0
- scipy >= 1.6.0

## Usage

### Single-source directed experiments (C1, C4, C6)

#### Five-seed paired repetition

The 5-seed entry fixes seeds `[42,43,44,45,46]` and the previous experiment's
STFT, ResNet18 + linear head, Adam at 1e-3, batch size 63, 50 epochs, and
final-epoch checkpoint. It runs both methods for all six directed pairs and
records initial-weight and per-epoch source-order SHA256 digests. Both methods
start with exactly the same architecture and weights and see the same source
cut order within a pair and seed.

```powershell
.\.venv\Scripts\python.exe -u run_five_seed_pairs.py --device cuda --out-root .\artifacts\five_seed_paired
```

The runner generates a feature-only STFT cache from raw signals and never puts
wear labels in that cache. Source-only loads target features only after its
training is complete; DARE-GRAM uses all 315 target cuts without labels during
training. Both final checkpoints and predictions are fixed before target wear
labels are read. Each result is under
`artifacts/five_seed_paired/<source>_to_<target>/seed_<seed>/<method>/`.
The root `seed_metrics.csv` has all 30 paired observations, `summary.csv` has
sample standard deviations and 95% paired t intervals, and `report.md` gives
the stability interpretation. Re-running the command verifies completed pairs
and continues unfinished work without replacing completed results. The prior
seed=42 runs in `artifacts/single_source_pairs_seed42` remain untouched; their
audit is in `prior_seed42_audit.json`.

To draw the six target-wear comparisons from the completed five-seed results:

```powershell
.\.venv\Scripts\python.exe plot_five_seed_curves.py
```

`artifacts/five_seed_paired/plots/` contains one PNG and PDF per direction,
a six-panel overview, and the exact per-cut mean and sample SD values in CSV.
The plotted R² annotation is the mean of the five per-seed R² values, which
differs from computing R² on the mean prediction curve.
`plots/best_r2/` contains a second six-figure set using each method's highest-R²
seed. The two methods' selected seeds may differ, so those figures are
descriptive rather than paired comparisons.

#### Full-lifecycle evaluation of saved five-seed checkpoints

The earlier `artifacts/five_seed_paired/` results and plots use the **legacy
evaluation scope**: target C6 cuts 95–315, target C1/C4 cuts 1–315. To evaluate
all six directions on target cuts 1–315 without retraining, run:

```powershell
.\.venv\Scripts\python.exe evaluate_full_lifecycle.py --device cuda
.\.venv\Scripts\python.exe plot_five_seed_curves.py --result-root .\artifacts\five_seed_full_lifecycle --out-dir .\artifacts\five_seed_full_lifecycle\plots --evaluation-scope full
```

The evaluation script verifies the 60 saved final checkpoints, training audit,
and feature-only STFT cache. For C6 it infers cuts 1–94 from the checkpoints and
joins them to the verified earlier predictions for cuts 95–315. For C1/C4 it
verifies and reuses the already complete per-cut predictions. It reads target
wear only after the checkpoints and required new predictions are fixed, then
checks that each CSV has exactly one correctly aligned row for every cut 1–315.
The output is separate: `artifacts/five_seed_full_lifecycle/` contains all 60
prediction CSVs, per-run metrics and curves, `seed_metrics.csv`, `summary.csv`,
`report.md`, and six-direction plots under `plots/`. Paired deltas and 95% t
intervals are recomputed from per-seed full-lifecycle predictions. The old C6
suffix metrics and images remain in `artifacts/five_seed_paired/`.

#### Single-seed entry

Run from this directory with the project's virtual environment. Each pair trains a fresh
source-only model and a fresh DARE-GRAM model with identical STFT inputs, source-fitted
normalization, ResNet18 + linear regression head, seed, batch size, optimizer, and
epoch count. The source is exactly one tool; `--source` and `--target` must differ.

```powershell
.\.venv\Scripts\python.exe -u run_single_source_pairs.py --source c1 --target c6 --device cuda
.\.venv\Scripts\python.exe -u run_single_source_pairs.py --all --device cuda
```

Use `--device cpu` if CUDA is unavailable. The default is 50 epochs, seed 42,
batch size 63, and learning rate 1e-3. Override paths with `--raw-root`,
`--split-record`, and `--out-root`. The batch size must divide 315, ensuring
all target cuts are used in every DARE-GRAM epoch. A completed pair is reused
by `--all` only when its recorded configuration matches; individual runs never
overwrite an existing pair directory.

Outputs are under `artifacts/single_source_pairs_seed42/<source>_to_<target>/`:
`config.json`, `audit.json`, `run.log`, `comparison.json`, `report.md`, and
per-method final checkpoints, metrics, prediction CSVs with actual cut indices,
prediction curves, configs, and logs. The root `summary.csv` and `report.md` contain available
pairs and DARE-GRAM minus source-only metric differences. The six-row summary
appears after all six directed pairs finish.

The target evaluation intervals follow the existing project: C6 cut 95-315;
C1/C4 cut 1-315. Metrics for different target intervals are not comparisons
on one common test set. DARE-GRAM uses the unlabeled STFT inputs for all 315
target cuts, including evaluation inputs, so this is **transductive UDA**.
Target wear labels are read only after both models have completed training,
saved final checkpoints, and produced predictions. Source-only uses no target
inputs during training. These runs use raw signal CSVs, not target NPZ archives
that also contain labels. The earlier C1+C4->C6 scripts and artifacts are kept.

### Data Preprocessing

```bash
python data_sampling.py 
```

### Basic Training

The commands below are historical examples for `train.py`. That loader reads
target NPZ labels before training, so use `run_single_source_pairs.py` above
for the audited single-source experiment.

- Source-only
```bash
python train.py \
    --model_name resnet \
    --source c1 \
    --target c4 \
    --data_dir ./dataset \
    --batch_size 64 \
    --max_epoch 50 \
    --lr 1e-3 
```
- Adaptation
```bash
python train.py \
    --model_name DAREGRAM \
    --source c1 \
    --target c4 \
    --data_dir ./dataset \
    --batch_size 128 \
    --max_epoch 50 \
    --lr 1e-3 
```


##  Implemented Models

### Domain Adaptation Models

#### Classification-based Models

1. **DANN** (Domain Adversarial Neural Network)

2. **MMD** (Maximum Mean Discrepancy)

#### Regression-based Models

1. **DAREGRAM** (Unsupervised Domain Adaptation Regression by Aligning Inversed Gram Matrices, CVPR 2023)


## Citation

```bash
@misc{TL-Bearing-Fault-Diagnosis,
    author = {Jinyuan Zhang},
    title = {TL-Bearing-Fault-Diagnosis},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/Feaxure-fresh/TL-Bearing-Fault-Diagnosis}}
}
```
```bash
@inproceedings{nejjar2023domain,
  title={DARE-GRAM : Unsupervised Domain Adaptation Regression by Aligning Inversed Gram Matrices},
  author={Nejjar, Ismail and Wang, Qin and Fink, Olga},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.},
  year={2023}
}
```

## Contact

- dsym2894@yonsei.ac.kr


