"""Industrial feature engineering for time series forecasting.

Instead of flattening a (window_size, n_features) matrix into a high-dimensional
vector, this module extracts compact, physically meaningful features from each
sliding window, reducing dimensionality from ~360 to ~42 while preserving
predictive information.

Feature groups per sample:
  - Rolling Statistics (24d): mean/std/max/min of 6 sensors over the full window
  - Recent Values (12d): raw values of 6 sensors at t-1 and t (last 2 steps)
  - Target-point Time Features (6d): cyclical + calendar encodings at the
    prediction point (hour_sin, hour_cos, month_sin, month_cos, day_of_week,
    is_weekend)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

TIME_FEATURE_COLUMNS = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'day_of_week', 'is_weekend'
]


def create_window_features(
    window: pd.DataFrame,
    sensor_columns: List[str],
    time_columns: List[str],
    target_column: str
) -> np.ndarray:
    """
    Extract a compact feature vector from a single sliding window.

    Instead of flattening all (window_size x n_features) values, this function
    compresses each window into a fixed-size vector with three groups:

    1. **Rolling Statistics (24 features)**
       For each of the 6 sensor columns, compute mean, std, max, and min
       across the entire window. This captures the aggregate level and
       variability of each sensor over the lookback period.

    2. **Recent Values (12 features)**
       The raw sensor readings at the two most recent time steps (t-1 and t).
       These provide the model with precise "current state" information
       immediately preceding the forecast point.

    3. **Target-point Time Features (6 features)**
       Cyclical encodings (hour_sin, hour_cos, month_sin, month_cos) and
       calendar features (day_of_week, is_weekend) extracted from the last
       row of the window (the forecast origin). These encode periodic patterns
       without leaking absolute temporal position.

    Parameters:
        window (pd.DataFrame): A DataFrame slice of shape (window_size, n_columns)
            containing pre-standardized sensor values, time features, and the
            target column for one sliding window.
        sensor_columns (List[str]): Names of the 6 sensor columns
            (e.g. HUFL, HULL, MUFL, MULL, LUFL, LULL).
        time_columns (List[str]): Names of the time feature columns to extract
            from the last row. Must be sorted for deterministic feature order.
        target_column (str): Name of the target column (OT), included in the
            window DataFrame but excluded from features.

    Returns:
        np.ndarray: 1D feature vector of length
            len(sensor_columns)*4 + len(sensor_columns)*2 + len(time_columns)
            For 6 sensors and 6 time features: 24 + 12 + 6 = 42.
    """
    features = {}

    # --- Group 1: Rolling statistics over full window (mean, std, max, min) ---
    for col in sorted(sensor_columns):
        values = window[col].values
        features[f'{col}_mean'] = np.mean(values)
        features[f'{col}_std'] = np.std(values)
        features[f'{col}_max'] = np.max(values)
        features[f'{col}_min'] = np.min(values)

    # --- Group 2: Recent values (last 2 timesteps: t-1 and t) ---
    for col in sorted(sensor_columns):
        features[f'{col}_t-1'] = window[col].iloc[-2]
        features[f'{col}_t'] = window[col].iloc[-1]

    # --- Group 3: Target-point time features (last row only) ---
    last_row = window.iloc[-1]
    for col in sorted(time_columns):
        if col in window.columns:
            features[col] = last_row[col]

    # Build vector in sorted key order for determinism
    sorted_keys = sorted(features.keys())
    return np.array([features[k] for k in sorted_keys])


def create_industrial_windows(
    data: pd.DataFrame,
    sensor_columns: List[str],
    time_columns: List[str],
    target_column: str,
    window_size: int = 24,
    horizon: int = 1,
    step: int = 1,
    target_mode: str = "absolute"
) -> Tuple[np.ndarray, ...]:
    """
    Build sliding-window samples using industrial feature extraction.

    For each window position, calls ``create_window_features`` to extract a
    compact 42-dimensional feature vector instead of flattening the full
    (window_size, n_features) matrix.

    Parameters:
        data (pd.DataFrame): Pre-processed DataFrame (standardized features,
            time features, and target column).
        sensor_columns (List[str]): Names of the 6 sensor columns.
        time_columns (List[str]): Names of time feature columns to extract
            from the last row of each window.
        target_column (str): Name of the target column (OT).
        window_size (int): Number of past time steps in each window (default: 24).
        horizon (int): Number of future time steps to predict (default: 1).
        step (int): Step size between consecutive windows (default: 1).
        target_mode (str): "absolute" for direct OT values, "delta" for
            OT[t+h] - OT[t+h-1] (default: "absolute").

    Returns:
        In "absolute" mode: (X, y)
            X shape: (n_samples, n_features_per_window)  e.g. (N, 42)
            y shape: (n_samples, horizon)
        In "delta" mode: (X, y, y_baseline)
            y shape: (n_samples, horizon) — delta targets
            y_baseline shape: (n_samples, horizon)

    Raises:
        ValueError: If target_mode is not "absolute" or "delta".
    """
    if target_mode not in ("absolute", "delta"):
        raise ValueError(f"target_mode must be 'absolute' or 'delta', got '{target_mode}'")

    X, y, y_baseline = [], [], []

    for i in range(0, len(data) - window_size - horizon + 1, step):
        window = data.iloc[i:i + window_size]
        window_target = data[target_column].iloc[i + window_size:i + window_size + horizon].values

        feature_vec = create_window_features(window, sensor_columns, time_columns, target_column)
        X.append(feature_vec)

        if target_mode == "delta":
            baseline = data[target_column].iloc[i + window_size - 1]
            delta = window_target - baseline
            y.append(delta.flatten())
            y_baseline.append(np.full(horizon, baseline))
        else:
            y.append(window_target.flatten())

    X_arr = np.array(X)
    y_arr = np.array(y)

    if target_mode == "delta":
        return X_arr, y_arr, np.array(y_baseline)

    return X_arr, y_arr