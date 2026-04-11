"""Time series data splitting utilities."""

import pandas as pd
import numpy as np
from typing import Tuple, List


def time_series_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training, validation, and test sets.
    
    Parameters:
        data (pd.DataFrame): The time series data to split
        train_ratio (float): Ratio of training data (default: 0.7)
        val_ratio (float): Ratio of validation data (default: 0.15)
        test_ratio (float): Ratio of test data (default: 0.15)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
    
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data


def create_sliding_windows(
    data: pd.DataFrame,
    target_column: str,
    window_size: int = 24,
    horizon: int = 1,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series forecasting.
    
    Parameters:
        data (pd.DataFrame): The time series data
        target_column (str): Name of the target column
        window_size (int): Size of the input window (default: 24)
        horizon (int): Forecast horizon (default: 1)
        step (int): Step size between windows (default: 1)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and targets (y) arrays
    """
    feature_cols = [col for col in data.columns if col != target_column]
    
    X, y = [], []
    
    for i in range(0, len(data) - window_size - horizon + 1, step):
        window_features = data[feature_cols].iloc[i:i + window_size].values
        window_target = data[target_column].iloc[i + window_size:i + window_size + horizon].values
        
        X.append(window_features.flatten())
        y.append(window_target.flatten())
    
    return np.array(X), np.array(y)


def create_rolling_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    windows: List[int] = [3, 6, 12, 24]
) -> pd.DataFrame:
    """
    Create rolling window features for time series data.
    
    Parameters:
        data (pd.DataFrame): The time series data
        feature_columns (List[str]): List of feature columns to create rolling features for
        windows (List[int]): List of window sizes (default: [3, 6, 12, 24])
    
    Returns:
        pd.DataFrame: DataFrame with added rolling features
    """
    result = data.copy()
    
    for col in feature_columns:
        for window in windows:
            result[f'{col}_rolling_mean_{window}'] = result[col].rolling(window=window).mean()
            result[f'{col}_rolling_std_{window}'] = result[col].rolling(window=window).std()
            result[f'{col}_rolling_min_{window}'] = result[col].rolling(window=window).min()
            result[f'{col}_rolling_max_{window}'] = result[col].rolling(window=window).max()
    
    result = result.fillna(method='bfill')
    
    return result


def create_lag_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    lags: List[int] = [1, 2, 3, 6, 12, 24]
) -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Parameters:
        data (pd.DataFrame): The time series data
        feature_columns (List[str]): List of feature columns to create lag features for
        lags (List[int]): List of lag values (default: [1, 2, 3, 6, 12, 24])
    
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    result = data.copy()
    
    for col in feature_columns:
        for lag in lags:
            result[f'{col}_lag_{lag}'] = result[col].shift(lag)
    
    result = result.fillna(method='bfill')
    
    return result