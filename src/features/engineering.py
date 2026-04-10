"""Feature engineering for time series data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Optional


def extract_time_features(data: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Extract time-based features from datetime column.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        datetime_column (str): Name of the datetime column
    
    Returns:
        pd.DataFrame: DataFrame with extracted time features
    """
    result = data.copy()
    
    if datetime_column not in result.columns:
        return result
    
    result[datetime_column] = pd.to_datetime(result[datetime_column])
    
    result['hour'] = result[datetime_column].dt.hour
    result['day'] = result[datetime_column].dt.day
    result['day_of_week'] = result[datetime_column].dt.dayofweek
    result['day_of_year'] = result[datetime_column].dt.dayofyear
    result['month'] = result[datetime_column].dt.month
    result['quarter'] = result[datetime_column].dt.quarter
    result['year'] = result[datetime_column].dt.year
    
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    return result


def select_features(data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Select specified features from DataFrame.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        feature_columns (List[str]): List of feature column names
    
    Returns:
        pd.DataFrame: DataFrame with selected features
    """
    existing_columns = [col for col in feature_columns if col in data.columns]
    missing_columns = set(feature_columns) - set(existing_columns)
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    return data[existing_columns]


def scale_features(
    X: pd.DataFrame, 
    numerical_features: Optional[List[str]] = None,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale numerical features using StandardScaler.
    
    Parameters:
        X (pd.DataFrame): Input DataFrame
        numerical_features (List[str]): List of numerical feature columns
        fit (bool): Whether to fit the scaler (default: True)
        scaler (StandardScaler): Pre-fitted scaler (optional)
    
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled DataFrame and fitted scaler
    """
    result = X.copy()
    
    if numerical_features is None:
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if fit:
        scaler = StandardScaler()
        result[numerical_features] = scaler.fit_transform(result[numerical_features])
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        result[numerical_features] = scaler.transform(result[numerical_features])
    
    return result, scaler


def encode_features(
    X: pd.DataFrame, 
    categorical_features: List[str],
    fit: bool = True,
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder]]:
    """
    Encode categorical features using OneHotEncoder.
    
    Parameters:
        X (pd.DataFrame): Input DataFrame
        categorical_features (List[str]): List of categorical feature columns
        fit (bool): Whether to fit the encoder (default: True)
        encoder (OneHotEncoder): Pre-fitted encoder (optional)
    
    Returns:
        Tuple[pd.DataFrame, OneHotEncoder]: Encoded DataFrame and fitted encoder
    """
    result = X.copy()
    
    existing_cat_features = [col for col in categorical_features if col in result.columns]
    
    if not existing_cat_features:
        return result, encoder
    
    if fit:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_features = encoder.fit_transform(result[existing_cat_features])
    else:
        if encoder is None:
            raise ValueError("encoder must be provided when fit=False")
        encoded_features = encoder.transform(result[existing_cat_features])
    
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(existing_cat_features)
    )
    
    result = result.drop(columns=existing_cat_features).reset_index(drop=True)
    result = pd.concat([result.reset_index(drop=True), encoded_df], axis=1)
    
    return result, encoder


def feature_pipeline(
    data: pd.DataFrame,
    datetime_column: str,
    numerical_features: List[str],
    categorical_features: Optional[List[str]] = None,
    fit_scaler: bool = True,
    fit_encoder: bool = True,
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, Optional[StandardScaler], Optional[OneHotEncoder]]:
    """
    Complete feature engineering pipeline.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        datetime_column (str): Name of datetime column
        numerical_features (List[str]): List of numerical features
        categorical_features (List[str]): List of categorical features (optional)
        fit_scaler (bool): Whether to fit the scaler (default: True)
        fit_encoder (bool): Whether to fit the encoder (default: True)
        scaler (StandardScaler): Pre-fitted scaler (optional)
        encoder (OneHotEncoder): Pre-fitted encoder (optional)
    
    Returns:
        Tuple[pd.DataFrame, StandardScaler, OneHotEncoder]: Processed DataFrame and fitted transformers
    """
    result = extract_time_features(data, datetime_column)
    
    result, scaler = scale_features(result, numerical_features, fit_scaler, scaler)
    
    if categorical_features:
        result, encoder = encode_features(result, categorical_features, fit_encoder, encoder)
    
    return result, scaler, encoder