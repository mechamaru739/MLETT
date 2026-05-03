"""Evaluation metrics for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    # MAPE with epsilon protection to avoid division by zero
    epsilon = 1e-8
    abs_true = np.abs(y_true)
    mask = abs_true > epsilon
    if np.any(mask):
        metrics['MAPE'] = np.mean(np.abs(y_pred[mask] - y_true[mask]) / abs_true[mask]) * 100
    else:
        metrics['MAPE'] = np.nan
    
    # SMAPE with epsilon protection
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > epsilon
    if np.any(mask):
        metrics['SMAPE'] = np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
    else:
        metrics['SMAPE'] = np.nan
    
    return {k: float(v) for k, v in metrics.items()}


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metrics to string with specified precision.
    
    Parameters:
        metrics (Dict[str, float]): Dictionary of metrics
        precision (int): Decimal precision (default: 4)
    
    Returns:
        Dict[str, str]: Formatted metrics
    """
    formatted = {}
    for key, value in metrics.items():
        if pd.isna(value):
            formatted[key] = "N/A"
        else:
            formatted[key] = f"{value:.{precision}f}"
    
    return formatted