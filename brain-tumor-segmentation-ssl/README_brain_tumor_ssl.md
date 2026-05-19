# Brain Tumor Segmentation & Classification with Self-Supervised Multi-Task Learning

**Course:** CM2026 Advanced Machine Learning for Data-Driven Health, KTH Royal Institute of Technology  
**Group 8:** Qusai Al Haj Ali · Mansour Arefi · De Chi Hao · Shreyas Balakarthikeyan  
**Dataset:** [BRISC2025 — Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)  
**Report:** [📄 GP_8_Report.pdf](./GP_8_Report.pdf)

---

## Overview

We develop two multi-task deep learning models for **simultaneous brain tumor segmentation and classification** from T1-weighted MRI. Both models follow a two-stage training protocol:

1. **Stage 1 — Self-supervised pre-training (MAE):** The encoder learns anatomical representations by reconstructing masked image patches — no labels required.
2. **Stage 2 — Multi-task fine-tuning:** The pre-trained encoder is jointly optimised for pixel-wise segmentation (U-Net decoder) and 4-class tumor classification (classification head).

An ablation study (`GP_8_NoSSL.ipynb`) runs the same fine-tuning pipeline **without** SSL pre-training to isolate the contribution of self-supervised learning.

**Classes:** Glioma · Meningioma · Pituitary Tumor · No Tumor

---

## Architecture

Two encoder architectures are compared within the same multi-task pipeline:

### U-Net encoder (baseline)

```
Input: (B, 1, 224, 224)

Encoder:  DoubleConv+MaxPool ×4  →  64→128→256→512→1024 channels
          Spatial: 224→112→56→28→14

Bottleneck: (B, 1024, 14, 14)

Decoder:  ConvTranspose2d + skip connections ×4  →  (B, 1, 224, 224) mask logit
Cls head: AdaptiveAvgPool → Flatten → Linear  →  4-class logits
```

### ResNet18-UNet encoder

```
Input: (B, 1, 224, 224)    ← first conv reshaped for single-channel MRI

Encoder:  ResNet-18 backbone  →  64→128→256→512 channels
          Spatial: 224→112→56→28→14→7

Bottleneck: (B, 512, 7, 7)

Decoder:  ConvTranspose2d + skip connections ×4  →  (B, 1, 224, 224) mask logit
Cls head: AdaptiveAvgPool → Flatten → Linear  →  4-class logits
```

---

## Self-Supervised Pre-Training (MAE)

A **Masked Autoencoder** is trained on unlabelled MRI images before supervised fine-tuning:

- Image patches (16×16) are randomly masked at **60% ratio**
- Encoder learns to produce representations from visible patches only
- Decoder reconstructs the full image, forcing the encoder to learn meaningful anatomical structure
- Pre-trained encoder weights are then transferred to the multi-task model

---

## Training strategy

| | Stage 1 (SSL) | Stage 2 (MTL) |
|---|---|---|
| Task | Patch reconstruction | Segmentation + Classification |
| Loss | MSE (reconstruction) | Dice + CrossEntropy (weighted sum) |
| Optimizer | Adam, lr=1e-4 | Adam, lr=1e-4 |
| Epochs | 100 | 80 (early stopping) |
| Batch size | 32 | 16 |

---

## Evaluation metrics

- **Dice score** — overlap between predicted and ground-truth mask
- **IoU (Jaccard)** — stricter overlap; penalises false positives more heavily
- **Hausdorff Distance** — measures worst-case boundary error (pixels)
- **Macro F1** — classification performance across all 4 tumor classes

---

## Notebooks

| Notebook | Description |
|---|---|
| `GP_8_SSL.ipynb` | **Primary.** MAE pre-training → multi-task fine-tuning |
| `GP_8_NoSSL.ipynb` | **Ablation.** Multi-task fine-tuning only (no SSL) |

Both notebooks are fully self-contained and executable top-to-bottom. Two flags in Section 2 control execution:

```python
LOCAL_QUICK_TEST           = True   # 5-epoch sanity check (fast)
USE_PRETRAINED_CHECKPOINTS = True   # Skip training, load shipped weights
```

Setting both to `True` (default) reproduces all test results in minutes.

---

## Pretrained models

| File | Size | Included |
|---|---|---|
| `best_model_resnet_SSL.pth` | 55 MB | ✅ |
| `best_model_resnet_NoSSL.pth` | 55 MB | ✅ |
| `best_model_unet_SSL.pth` | 119 MB | ⬇️ See below |
| `best_model_unet_NoSSL.pth` | 119 MB | ⬇️ See below |

> The two U-Net checkpoints exceed GitHub's 100 MB file limit and are not included. Contact the authors or request access to the full checkpoint archive.

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

The notebook downloads BRISC2025 automatically via `kagglehub`:

```python
# Cell A (local)
import kagglehub
raw_path = kagglehub.dataset_download("briscdataset/brisc2025")
```

Requires a Kaggle account. If automatic download fails, download manually from [kaggle.com/datasets/briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025).

### 3. Run a notebook end to end

Open `GP_8_SSL.ipynb` → run Cell A (local dataset config) → continue running all cells.  
Default flags (`LOCAL_QUICK_TEST=True`, `USE_PRETRAINED_CHECKPOINTS=True`) reproduce test results instantly.

---

## Repository structure

```
brain-tumor-segmentation-ssl/
├── GP_8_SSL.ipynb                     # Primary: SSL pre-training + multi-task fine-tuning
├── GP_8_NoSSL.ipynb                   # Ablation: multi-task only (no SSL)
├── GP_8_Report.pdf                    # Written project report
├── requirements.txt                   # Python dependencies
└── pretrained_models/
    ├── best_model_resnet_SSL.pth      # ResNet18-UNet with SSL pre-training
    └── best_model_resnet_NoSSL.pth    # ResNet18-UNet without SSL pre-training
```

---

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

`python` `pytorch` `unet` `resnet` `self-supervised-learning` `masked-autoencoder` `multi-task-learning` `brain-tumor` `mri` `segmentation` `classification` `medical-imaging` `kth`
