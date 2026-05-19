# utility_functions.py
# Helper functions for Assignment 1 (VTaC)

from pathlib import Path

import pandas as pd
import numpy as np
import wfdb


def read_wfdb_header(record_id: str, event_id: str, waveforms_dir: Path):
    """
    Use:
        Read WFDB header for one event to get sampling rate and channel names.
    Inputs:
        record_id (str): patient waveform record folder name
        event_id (str): event record base name (WFDB record without extension)
        waveforms_dir (Path): path to the 'waveforms' folder
    Outputs:
        ok (bool): True if header is read successfully, else False
        fs (float or None): sampling rate in Hz (e.g., 250.0), or None if failed
        sig_names (list[str]): list of channel names (e.g., ['II','V','PLETH'])
        err (str or None): error message if failed, else None
    """
    base = waveforms_dir / record_id / event_id  # WFDB base path without extension
    try:
        h = wfdb.rdheader(str(base))
        fs = float(h.fs)
        sig_names = [str(s).strip() for s in h.sig_name]
        return True, fs, sig_names, None
    except Exception as e:
        return False, None, [], str(e)


def scan_wfdb_headers(events_df: pd.DataFrame, waveforms_dir: Path) -> pd.DataFrame:
    """
    Use:
        Loop through all events and add WFDB header info columns.
    Inputs:
        events_df (pd.DataFrame): must contain columns ['record','event']
        waveforms_dir (Path): path to the 'waveforms' folder
    Outputs:
        df_out (pd.DataFrame): copy of events_df with added columns:
            - header_ok (bool)
            - fs (float)
            - sig_names (list[str])
            - header_error (str)
    """
    header_ok = []
    fs_list = []
    sig_names_list = []
    err_list = []

    for r, e in zip(events_df["record"], events_df["event"]):
        ok, fs, sigs, err = read_wfdb_header(r, e, waveforms_dir)
        header_ok.append(ok)
        fs_list.append(fs)
        sig_names_list.append(sigs)
        err_list.append(err)

    df_out = events_df.copy()
    df_out["header_ok"] = header_ok
    df_out["fs"] = fs_list
    df_out["sig_names"] = sig_names_list
    df_out["header_error"] = err_list
    return df_out


def compute_sig_name_counts(df_with_headers: pd.DataFrame) -> pd.DataFrame:
    """
    Use:
        Compute how often each signal name appears across events (based on sig_names lists).
    Inputs:
        df_with_headers (pd.DataFrame): must contain 'header_ok' and 'sig_names'
    Outputs:
        sig_counts (pd.DataFrame): index = sig_name, columns = ['count','pct_events']
    """
    ok_df = df_with_headers[df_with_headers["header_ok"] == True].copy()
    n_ok = len(ok_df)

    exploded = ok_df.explode("sig_names").dropna(subset=["sig_names"])
    exploded["sig_names"] = exploded["sig_names"].astype(str).str.strip().str.upper()

    sig_counts = (
        exploded.groupby("sig_names")
        .size()
        .sort_values(ascending=False)
        .to_frame("count")
    )
    sig_counts["pct_events"] = 100 * sig_counts["count"] / n_ok
    return sig_counts


def summarize_sampling_rate(df_with_headers: pd.DataFrame, expected_fs: float = 250.0):
    """
    Use:
        Simple summary of sampling rates across events (for header_ok=True).
    Inputs:
        df_with_headers (pd.DataFrame): must contain 'header_ok' and 'fs'
        expected_fs (float): expected sampling rate, default 250.0
    Outputs:
        fs_counts (pd.Series): value_counts of fs among readable events
        mismatch_df (pd.DataFrame): events where fs != expected_fs
    """
    ok_df = df_with_headers[df_with_headers["header_ok"] == True].copy()
    fs_counts = ok_df["fs"].value_counts(dropna=False).sort_index()
    mismatch_df = ok_df[ok_df["fs"] != expected_fs][["record", "event", "split", "decision", "fs"]]
    return fs_counts, mismatch_df



def basic_signal_features(x: np.ndarray) -> dict:
    """
    Use:
        Compute simple statistics from a 1D signal segment.
    Inputs:
        x (np.ndarray): shape (n_samples,), signal values
    Outputs:
        feats (dict): dictionary with simple features:
            mean, std, min, max, ptp, rms, linelen
    """
    x = x.astype(float)

    feats = {}
    feats["mean"] = float(np.mean(x))
    feats["std"]  = float(np.std(x))
    feats["min"]  = float(np.min(x))
    feats["max"]  = float(np.max(x))
    feats["ptp"]  = float(np.max(x) - np.min(x))
    feats["rms"]  = float(np.sqrt(np.mean(x**2)))
    feats["linelen"] = float(np.sum(np.abs(np.diff(x))))
    return feats


def extract_case1_features_for_event(record_id: str,
                                     event_id: str,
                                     waveforms_dir: Path,
                                     pre_sec: int = 10,
                                     post_sec: int = 5,
                                     onset_sec: int = 300,
                                     fs_expected: int = 250,
                                     signals = ("II", "V", "PLETH")) -> dict:
    """
    Use:
        Extract features for Case 1 signals (II, V, PLETH) from a time window around alarm onset.
        Reads only the needed channels and only the needed samples (fast).
    Inputs:
        record_id (str): folder name for patient record
        event_id (str): WFDB record name inside that folder
        waveforms_dir (Path): path to waveforms folder
        pre_sec (int): seconds before onset to include
        post_sec (int): seconds after onset to include
        onset_sec (int): onset time in seconds (5 min = 300 sec)
        fs_expected (int): expected sampling rate (VTaC uses 250 Hz)
        signals (tuple/list): signal names to extract, default ('II','V','PLETH')
    Outputs:
        row (dict): contains record/event and features:
            e.g. II_mean, II_std, ... V_mean, ... PLETH_mean, ...
    """
    base = waveforms_dir / record_id / event_id

    # Read header first to find channel indices
    h = wfdb.rdheader(str(base))
    fs = int(h.fs)
    if fs != fs_expected:
        raise ValueError(f"Unexpected fs={fs} (expected {fs_expected}) for {record_id}/{event_id}")

    sig_names = [str(s).strip().upper() for s in h.sig_name]
    name_to_idx = {name: i for i, name in enumerate(sig_names)}

    # We assume Case 1 events contain all three signals
    channel_idxs = []
    for s in signals:
        if s not in name_to_idx:
            raise ValueError(f"Missing signal {s} for {record_id}/{event_id}")
        channel_idxs.append(name_to_idx[s])

    # Calculate sample window
    onset_samples = onset_sec * fs
    sampfrom = onset_samples - pre_sec * fs
    sampto   = onset_samples + post_sec * fs

    if sampfrom < 0:
        sampfrom = 0

    # Read only those channels and only that sample range
    rec = wfdb.rdrecord(str(base), channels=channel_idxs, sampfrom=sampfrom, sampto=sampto)

    # Get signal matrix
    if rec.p_signal is not None:
        X = rec.p_signal
    else:
        X = rec.d_signal.astype(float)

    row = {"record": record_id, "event": event_id, "fs": fs}

    # X has columns in the same order as channel_idxs -> same order as "signals"
    for col, s in enumerate(signals):
        x = X[:, col]
        feats = basic_signal_features(x)
        for k, v in feats.items():
            row[f"{s}_{k}"] = v

    return row


def extract_features_for_event_allow_missing(record_id: str,
                                             event_id: str,
                                             waveforms_dir: Path,
                                             signals,
                                             pre_sec: int,
                                             post_sec: int,
                                             onset_sec: int = 300,
                                             fs_expected: int = 250) -> dict:
    """
    Use:
        Extract features for a list of signals, allowing some signals to be missing.
        Missing signals -> feature values become NaN.
    Inputs/Outputs:
        Similar to extract_case1_features_for_event, but supports missing channels.
    """
    base = waveforms_dir / record_id / event_id
    h = wfdb.rdheader(str(base))
    fs = int(h.fs)
    if fs != fs_expected:
        raise ValueError(f"Unexpected fs={fs} (expected {fs_expected}) for {record_id}/{event_id}")

    sig_names = [str(s).strip().upper() for s in h.sig_name]
    name_to_idx = {name: i for i, name in enumerate(sig_names)}

    # which signals are present?
    present = [s for s in signals if s in name_to_idx]
    idxs = [name_to_idx[s] for s in present]

    onset_samples = onset_sec * fs
    sampfrom = max(0, onset_samples - pre_sec * fs)
    sampto   = onset_samples + post_sec * fs

    rec = wfdb.rdrecord(str(base), channels=idxs, sampfrom=sampfrom, sampto=sampto)
    X = rec.p_signal if rec.p_signal is not None else rec.d_signal.astype(float)

    row = {"record": record_id, "event": event_id, "fs": fs}

    # map present signal -> column index in X
    present_to_col = {s: i for i, s in enumerate(present)}

    for s in signals:
        if s in present_to_col:
            x = X[:, present_to_col[s]]
            feats = basic_signal_features(x)  # same feature function as before
            for k, v in feats.items():
                row[f"{s}_{k}"] = v
        else:
            # signal missing -> fill NaN features
            for k in ["mean","std","min","max","ptp","rms","linelen"]:
                row[f"{s}_{k}"] = np.nan

    return row
