# Binary Liver Segmentation with U-Net

**Course:** CM2026 Advanced Machine Learning for Data-Driven Health, KTH Royal Institute of Technology  
**Author:** Mansour Arefi  
**Dataset:** [AbdomenAtlas 1.0 Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) — 100 abdominal CT patients

> ⚠️ Raw CT data not included — data was accessed from the KTH cloud (S3) during training on Kaggle.

---

## Problem

Liver segmentation in CT imaging is a foundational task in medical image analysis — enabling volume measurement, surgical planning, and disease monitoring. This project implements a **2D U-Net from scratch** for binary segmentation (liver / not-liver) on axial CT slices, following the original U-Net paper architecture.

---

## Project structure — three notebooks

| Notebook | Content |
|---|---|
| `i-data-preparation.ipynb` | Load 3D CT volumes from cloud, build slice index, train/val/test split, compute normalisation stats |
| `ii-trainunet-baseline.ipynb` | Baseline U-Net: 10 patients, weighted CE loss, minimal augmentation |
| `iii-trainunet-extended.ipynb` | Extended U-Net: 40 patients, BatchNorm, combined CE+Dice loss, richer augmentation, adaptive LR, early stopping |

---

## Data pipeline

- **Source:** AbdomenAtlas 1.0 Mini — 3D NIfTI CT volumes + binary liver masks
- **Preprocessing:** HU clipping to [-100, 400] (soft tissue window), z-score normalisation per dataset
- **Slicing:** 3D volumes converted to 2D axial slices; each slice labelled as liver-positive or liver-negative
- **Split:** 60% train / 20% val / 20% test at patient level (no slice leakage)
- **Class imbalance:** liver pixels ~10% of total — handled via class-weighted loss and Dice loss

---

## Model — U-Net

```
Input: (B, 1, 256, 256)   ← single-channel CT slice

Encoder (contracting path):
  DoubleConv(1→64)   → MaxPool
  DoubleConv(64→128) → MaxPool
  DoubleConv(128→256)→ MaxPool
  DoubleConv(256→512)→ MaxPool

Bottleneck:
  DoubleConv(512→1024)

Decoder (expanding path):
  ConvTranspose2d + skip connection + DoubleConv  ×4

Output: (B, 2, 256, 256)   ← per-pixel class logits
```

`DoubleConv` = Conv2d → [BatchNorm] → ReLU → Conv2d → [BatchNorm] → ReLU  
*(BatchNorm added in extended version)*

---

## Baseline vs. Extended

| | Baseline | Extended |
|---|---|---|
| Patients | 10 | 40 |
| BatchNorm | ❌ | ✅ |
| Loss | Weighted CE | 0.5×CE + 0.5×Dice |
| Augmentation | HorizontalFlip | Flip · Rotate90 · Brightness · ElasticTransform |
| LR schedule | Static (1e-4) | ReduceLROnPlateau (halve after 8 epochs without improvement) |
| Early stopping | ❌ | ✅ patience=15 |
| Gradient clipping | ❌ | ✅ max_norm=1.0 |
| Mixed precision | ❌ | ✅ AMP (torch.amp) |
| Weight init | Kaiming normal | Kaiming normal |

---

## Why these design choices

**BatchNorm** — eliminates the wild validation Dice oscillations seen in the baseline by normalising activations between layers; acts as implicit regularisation and allows higher effective learning rates.

**Dice loss component** — directly optimises the evaluation metric (Dice score) rather than a proxy; insensitive to class imbalance because it operates on the overlap ratio rather than per-pixel accuracy.

**ReduceLROnPlateau** — more stable than a fixed schedule; LR halves when the model stops improving, preventing oscillation around a local minimum late in training.

**Elastic deformation augmentation** — simulates realistic organ shape variability in CT; liver shape varies significantly between patients and breathing phases.

---

## Evaluation metrics

- **Dice score** — primary metric; measures overlap between prediction and ground truth
- **IoU (Jaccard index)** — secondary metric; stricter than Dice on boundary accuracy
- **Weighted cross-entropy loss** — per-pixel loss with class weights proportional to inverse frequency

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run in order

```
i-data-preparation.ipynb     ← prepare and cache the dataset
ii-trainunet-baseline.ipynb  ← train baseline model
iii-trainunet-extended.ipynb ← train extended model
```

> The notebooks were developed and trained on **Kaggle** with GPU acceleration. Data was loaded from the KTH cloud S3 bucket. Local reproduction requires re-pointing the data paths.

---

## Repository structure

```
liver-segmentation-unet/
├── i-data-preparation.ipynb       # Data loading, preprocessing, split, normalisation stats
├── ii-trainunet-baseline.ipynb    # Baseline U-Net training (10 patients)
├── iii-trainunet-extended.ipynb   # Extended U-Net training (40 patients, improved design)
└── requirements.txt               # Python dependencies
```

---

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

`python` `pytorch` `unet` `medical-imaging` `segmentation` `ct-scan` `liver` `dice-loss` `augmentation` `kaggle` `kth`
