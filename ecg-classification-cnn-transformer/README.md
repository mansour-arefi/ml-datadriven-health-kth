# Generalizable ECG Classification with CNN-Transformer

**Course:** CM2011 Machine Learning for Health, KTH Royal Institute of Technology  
**Authors:** Mansour Arefi, Qusai Al Haj Ali  
**Notebook:** `assignment3-final.ipynb`

**Datasets:**
- [PTB-XL — PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) — 21,799 clinical 12-lead ECGs
- [LTDB — PhysioNet](https://physionet.org/content/ltdb/1.0.0/) — 7 long-term ambulatory 2-lead recordings

> ⚠️ Raw data not included — download from the PhysioNet links above.

---

## Problem

ECG interpretation is highly environment-dependent — models trained on clean clinical recordings often fail on ambulatory or out-of-distribution data. This project trains a CNN-Transformer on PTB-XL to classify five diagnostic categories, then uses **semi-supervised pseudo-labelling on LTDB** to improve robustness across recording environments.

**Target classes:** NORM · MI · STTC · CD · HYP

---

## Architecture — CNN-Transformer

A hybrid model where the CNN captures local morphology (QRS shape, P-waves) and the Transformer captures long-range patterns (beat-to-beat variation, RR intervals).

```
Input: (B, 12, 1000)              ← 12 leads, 10 seconds at 100 Hz
  CNNBlock(12→64,  k=15, pool=2)  → (B, 64, 500)
  CNNBlock(64→128, k=9,  pool=2)  → (B, 128, 250)
  CNNBlock(128→256, k=5, pool=2)  → (B, 256, 125)
  Positional Encoding
  Transformer Encoder (4 layers, 8 heads, d_model=256)
  CLS token → Linear → 5-class output
```

**Design rationale:**
- CNN reduces 1000 timesteps → 125 before attention, keeping computation tractable
- GELU activations throughout; BatchNorm after each conv layer
- Transformer attends across 125 positions — each representing ~80ms of ECG

---

## Training strategy

### Phase 1 — Supervised on PTB-XL

| Parameter | Value | Rationale |
|---|---|---|
| Loss | Focal Loss (γ=3, per-class weights) | 7:1 class imbalance — suppresses easy negatives |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 | Adaptive LR with decoupled L2 regularisation |
| Scheduler | Linear warmup + cosine decay | Stable early training, smooth convergence |
| Split | Folds 1-8 train / fold 9 val / fold 10 test | Stratified official PTB-XL split |

### Phase 2 — Semi-Supervised on LTDB

LTDB has no labels. We generate pseudo-labels using the Phase 1 model:
1. Run inference on all LTDB 10-second windows
2. Discard high-entropy predictions (bottom 40% confidence)
3. Filter by per-class confidence thresholds (HYP: 0.15 — lower threshold needed as model rarely predicts HYP with high confidence)
4. Fine-tune on combined PTB-XL + pseudo-labelled LTDB with label smoothing

**Semi-supervised improvement: +1.5 pp macro-F1** — exposing the model to ambulatory recording characteristics improved generalisation.

---

## Preprocessing

Both datasets processed through the same pipeline:
- 4th-order Butterworth bandpass filter (0.5–40 Hz) — removes baseline wander and high-frequency noise
- Z-score normalisation per record
- LTDB: resampled 128 Hz → 100 Hz; 2 leads tiled circularly to fill 12 channels

---

## Results

| Class | F1-score |
|---|---|
| NORM | 0.84 |
| MI | ~0.70 |
| STTC | ~0.68 |
| CD | ~0.65 |
| HYP | 0.28 |

HYP remains the hardest class — high-voltage QRS morphology overlaps with normal variants.

**Noise robustness:** Model tested with Gaussian noise at increasing σ levels. F1 at σ=0.4: reported in notebook.

---

## Interpretability

Three complementary methods applied:

**Transformer attention maps** — visualise which timesteps the last attention layer focuses on per class. Consistently highlights QRS complexes and T-waves.

**SHAP (GradientExplainer)** — assigns contribution scores to each time-step. Red = pushes toward predicted class, blue = pushes away. Confirms model attends to clinically relevant ECG regions.

**Integrated Gradients** — integrates gradients along a path from zero baseline to input signal. Complementary to SHAP; works natively on GPU without model copy.

All three methods consistently identify the same clinically meaningful ECG segments — providing convergent evidence that the model learns real cardiac morphology rather than dataset artefacts.

---

## Generalisability testing

- **Gaussian noise injection** at σ = 0.1, 0.2, 0.4, 0.8 — measures degradation curve
- **Cross-dataset transfer** — LTDB used as held-out ambulatory test environment (no labels seen during Phase 1)
- **Semi-supervised adaptation** — pseudo-labelling explicitly bridges the domain gap between clinical and ambulatory recordings

---

## Repository structure

```
ecg-classification-cnn-transformer/
├── assignment3-final.ipynb    # Full pipeline: data, model, training, evaluation, interpretability
└── requirements.txt           # Python dependencies
```

---

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

`python` `pytorch` `cnn` `transformer` `ecg` `semi-supervised` `shap` `attention` `integrated-gradients` `medical-imaging` `kth`
