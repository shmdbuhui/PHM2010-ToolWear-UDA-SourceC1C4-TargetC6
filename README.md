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
 │         ├── c1_001.csv
 │         ├── c1_002.csv
 │         ├── ...
 │         └── c1_315.csv
 ├── c4/
 │    ├── c4_wear.csv
 │    └── c4/
 │         ├── c4_001.csv
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

### Data Preprocessing

```bash
python data_sampling.py 
```

### Basic Training

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


