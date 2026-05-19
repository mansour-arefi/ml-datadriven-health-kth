# utility_functions.py
# Helper functions and model definition for Sepsis Early Prediction pipeline.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

STEP_SIZE   = 0.5
HORIZONS    = {"2h": 2, "4h": 4, "6h": 6}
WINDOW_SIZE = 24


def create_labels(df, horizons=HORIZONS, step_size=STEP_SIZE):
    """
    Creates early-warning labels for each timestep per patient.

    Label logic:
    ------------
    For sepsis patients (any row where severity >= 2 exists):
        t_onset = first timestep where severity >= 2
        For each t < t_onset:
            sepsis_within_Xh = 1  if  (t_onset - t) <= X hours
        Rows at t >= t_onset are removed (leakage prevention)

    For non-sepsis patients (severity stays < 2 throughout):
        All labels = 0, all rows kept

    Parameters
    ----------
    df        : pd.DataFrame — long format, sorted by (id, timestep)
    horizons  : dict         — {"2h": 2, "4h": 4, "6h": 6}
    step_size : float        — 0.5 (hours per timestep)

    Returns
    -------
    pd.DataFrame with columns: sepsis_within_2h, sepsis_within_4h,
                                sepsis_within_6h added; post-onset rows removed
    """
    records = []

    for pid, group in df.groupby("id"):
        group = group.sort_values("timestep").copy()

        # Use severity column to find onset
        onset_rows = group[group["severity"] >= 2]

        if onset_rows.empty:
            # Non-sepsis patient — severity never reaches 2
            for h_name in horizons:
                group[f"sepsis_within_{h_name}"] = 0
            records.append(group)

        else:
            # Sepsis patient — find first timestep of clinical sepsis
            t_onset = onset_rows["timestep"].min()

            # Remove onset and post-onset rows
            group = group[group["timestep"] < t_onset].copy()

            if group.empty:
                # Onset at first recorded timestep — no usable history
                continue

            # Label each pre-onset row per horizon
            # Both t_onset and timestep are in hours so compare directly in hours
            for h_name, h_val in horizons.items():
                group[f"sepsis_within_{h_name}"] = (
                    (t_onset - group["timestep"]) <= h_val
                ).astype(int)

            records.append(group)

    return pd.concat(records, ignore_index=True)



def create_sliding_windows(df, feature_cols, label_col,
                            window_size=WINDOW_SIZE, stride=2):
    """
    stride=1 → every 30min  → ~375,000 windows (too heavy)
    stride=2 → every 1h     → ~187,500 windows (chosen)
    stride=4 → every 2h     → ~93,750  windows (lightweight)
    """
    X_list, y_list, pid_list = [], [], []

    for pid, group in df.groupby("id"):
        group    = group.sort_values("timestep")
        features = group[feature_cols].values.astype(np.float32)
        labels   = group[label_col].values.astype(np.float32)
        T        = len(features)

        if T < window_size:
            pad      = np.zeros((window_size - T, features.shape[1]),
                                dtype=np.float32)
            features = np.vstack([pad, features])
            X_list.append(features)
            y_list.append(labels[-1])
            pid_list.append(pid)
        else:
            # ← stride controls how often we sample a window
            for t in range(window_size, T + 1, stride):
                X_list.append(features[t - window_size : t])
                y_list.append(labels[t - 1])
                pid_list.append(pid)

    return (
        np.array(X_list,   dtype=np.float32),
        np.array(y_list,   dtype=np.float32),
        np.array(pid_list),
    )


class SepsisGRU(nn.Module):
    """
    Two-layer stacked GRU for binary sepsis early prediction.
    Input  : (batch, seq_len, input_size)
    Output : (batch,) raw logits — sigmoid applied at inference
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _  = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        logits      = self.classifier(last_hidden)  # (batch, 1)
        return logits                               


def compute_shap_importance(model, X_train, X_val,
                             feature_names, n_background=100, n_explain=200):
    """
    SHAP GradientExplainer

    Parameters
    ----------
    model        : trained SepsisGRU
    X_train      : np.ndarray (N_tr, W, F)
    X_val        : np.ndarray (N_va, W, F)
    feature_names: list[str]
    n_background : int
    n_explain    : int

    Returns
    -------
    shap_importance_df : pd.DataFrame with columns ['feature', 'importance']
                         sorted by importance descending
    shap_values        : np.ndarray (n_explain, W, F) — raw SHAP values
    """
    import shap

    model.eval()

    background = X_train[np.random.choice(X_train.shape[0],
                                           n_background, replace=False)]
    test_sample = X_val[np.random.choice(X_val.shape[0],
                                          n_explain, replace=False)]

    background_t  = torch.tensor(background,  dtype=torch.float32)
    test_sample_t = torch.tensor(test_sample, dtype=torch.float32)

    explainer   = shap.GradientExplainer(model, background_t)
    shap_values = explainer.shap_values(test_sample_t)

    # Normalise to list
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    # Convert tensors to numpy
    shap_values = [
        v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
        for v in shap_values
    ]
    
    per_output = []
    for v in shap_values:
        per_output.append(np.abs(v).mean(axis=(0, 1)).flatten())

    importances = np.mean(per_output, axis=0).flatten()

    shap_importance_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importances          
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return shap_importance_df, shap_values[0]