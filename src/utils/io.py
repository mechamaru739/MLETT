"""IO utilities for data and model handling."""

import os
import json
import pandas as pd
import joblib
from typing import Any, Dict
import yaml


def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False):
    """
    Save DataFrame to CSV file.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save file
        index (bool): Whether to save index (default: False)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Parameters:
        filepath (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    return pd.read_csv(filepath)


def save_json(data: Any, filepath: str):
    """
    Save data to JSON file.
    
    Parameters:
        data: Data to save (must be JSON serializable)
        filepath (str): Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Parameters:
        filepath (str): Path to JSON file
    
    Returns:
        Any: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: str):
    """
    Save data to YAML file.
    
    Parameters:
        data (Dict[str, Any]): Data to save
        filepath (str): Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from YAML file.
    
    Parameters:
        filepath (str): Path to YAML file
    
    Returns:
        Dict[str, Any]: Loaded data
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def ensure_directory(directory: str):
    """
    Ensure directory exists, create if not.
    
    Parameters:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model: Any, filepath: str):
    """
    Save model using joblib.
    
    Parameters:
        model: Model object to save
        filepath (str): Path to save model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model using joblib.
    
    Parameters:
        filepath (str): Path to model file
    
    Returns:
        Any: Loaded model
    """
    return joblib.load(filepath)