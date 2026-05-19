# VT Alarm False Alarm Classification

**Course:** Applied Machine Learning for Health (CM2011), KTH Royal Institute of Technology  
**Dataset:** [VTaC — PhysioNet](https://physionet.org/content/vtac/)  
**Notebook:** `False_Alarms.ipynb`

---

## Problem

Alarm fatigue is a critical issue in intensive care. Clinical staff receive hundreds of alarms daily and rely on intuition to distinguish **true** from **false** alarms — with no systematic tool to help. This assignment trains ML models to automatically classify VT (Ventricular Tachycardia) alarms as true or false using physiological waveform data.

---

## Dataset

The **VTaC dataset** (PhysioNet) contains 6-minute waveform recordings centred on VT alarm events from ICU patients. Each event includes multi-lead ECG and physiological signals sampled at 250 Hz.

> ⚠️ Raw data not included — download from [physionet.org/content/vtac](https://physionet.org/content/vtac/)

---

## Approach

### Signal selection & two experimental cases

| | Signals used | Strategy for missing values |
|---|---|---|
| **Case 1 (baseline)** | II, V, PLETH | Drop events missing any signal |
| **Case 2 (extended)** | II, V, PLETH, AVR, ABP | Compare drop vs. imputation |

### Feature extraction
- 15-second window around alarm onset (10s pre, 5s post)
- Per-signal statistical features extracted from raw waveform samples
- WFDB headers scanned to determine signal availability per event
- 7 events removed due to unreadable headers

### Models
- Logistic Regression (with StandardScaler)
- Random Forest (`n_estimators=300`, `class_weight=balanced_subsample`)

### Evaluation
- Primary metric: **AUPRC** (Area Under Precision-Recall Curve) — chosen due to class imbalance
- Secondary: AUROC, F1

---

## Key results

| Model | Case | Val AUROC | Val AUPRC |
|---|---|---|---|
| Logistic Regression | Case 1 | ~0.57 | — |
| Random Forest | Case 1 | ~0.62 | — |
| Logistic Regression | Case 2 (drop) | ~0.66 | — |
| Random Forest | Case 2 (drop) | ~0.73 | — |
| Logistic Regression | Case 2 (impute) | ~0.60 | — |
| Random Forest | Case 2 (impute) | ~0.64 | — |

**Random Forest with drop strategy (Case 2) was the best-performing configuration.**  
Dropping rows with missing signals outperformed median imputation in this dataset — likely because missingness was non-random (sensors not connected), making imputed values misleading.

---

## Feature importance

- Permutation importance applied to the best Random Forest model on the validation set
- Recursive feature elimination performed by iteratively removing the least important feature and tracking AUPRC
- Result: a small subset of waveform features drove most of the predictive performance

---

## Discussion

- Missing data was **not random** — certain sensors were simply not connected, so imputing their values introduced noise rather than information
- AUPRC was the primary metric because the alarm dataset is class-imbalanced
- The 15-second window around alarm onset was chosen to capture the most clinically relevant signal segment

---

## Repository structure

```
assignment-1-false-alarms/
├── False_Alarms.ipynb                             # Full analysis
├── utility_functions.py                           # WFDB scanning, feature extraction helpers
├── requirements.txt                               # Python dependencies
└── README.md
```

---

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

`python` `machine-learning` `healthcare` `ecg` `classification` `random-forest` `feature-engineering` `imbalanced-data` `kth`
