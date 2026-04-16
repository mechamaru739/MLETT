"""Feature engineering for time series data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Optional, Tuple, Dict, Any


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
    
    result = result.drop(columns=[datetime_column])
    
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
    
    When fit=True, a new scaler is fitted on the data (use on training set).
    When fit=False, a pre-fitted scaler is applied (use on val/test sets).
    
    Parameters:
        X (pd.DataFrame): Input DataFrame
        numerical_features (List[str]): List of numerical feature columns
        fit (bool): Whether to fit the scaler (default: True)
        scaler (StandardScaler): Pre-fitted scaler (required when fit=False)
    
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
    
    When fit=True, a new encoder is fitted on the data (use on training set).
    When fit=False, a pre-fitted encoder is applied (use on val/test sets).
    
    Parameters:
        X (pd.DataFrame): Input DataFrame
        categorical_features (List[str]): List of categorical feature columns
        fit (bool): Whether to fit the encoder (default: True)
        encoder (OneHotEncoder): Pre-fitted encoder (required when fit=False)
    
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


class FeatureTransformer:
    """
    Stateful feature engineering transformer that avoids data leakage.
    
    Standardizes both features and target column. Target is standardized
    using its own scaler, and can be inverse-transformed for metrics
    calculation in the original scale.
    
    Usage:
        transformer = FeatureTransformer(datetime_column, numerical_features, target_column, ...)
        train_data = transformer.fit_transform(train_raw)
        val_data = transformer.transform(val_raw)
        test_data = transformer.transform(test_raw)
        
        # Inverse transform predictions to original scale
        y_original = transformer.inverse_transform_target(y_standardized)
    """
    
    def __init__(
        self,
        datetime_column: str,
        numerical_features: List[str],
        target_column: Optional[str] = None,
        categorical_features: Optional[List[str]] = None
    ):
        self.datetime_column = datetime_column
        self.numerical_features = numerical_features
        self.target_column = target_column
        self.categorical_features = categorical_features or []
        self.scaler = None
        self.target_scaler = None
        self.encoder = None
        self._fitted = False
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit transformers on training data and transform it.
        
        Parameters:
            data (pd.DataFrame): Training data
        
        Returns:
            pd.DataFrame: Transformed training data (features and target standardized)
        """
        result = extract_time_features(data, self.datetime_column)
        result, self.scaler = scale_features(result, self.numerical_features, fit=True)
        
        if self.categorical_features:
            result, self.encoder = encode_features(result, self.categorical_features, fit=True)
        
        if self.target_column and self.target_column in result.columns:
            self.target_scaler = StandardScaler()
            result[[self.target_column]] = self.target_scaler.fit_transform(result[[self.target_column]])
        
        self._fitted = True
        return result
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using pre-fitted transformers (for val/test).
        
        Parameters:
            data (pd.DataFrame): Validation or test data
        
        Returns:
            pd.DataFrame: Transformed data (features and target standardized)
        
        Raises:
            RuntimeError: If transformer has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before transform")
        
        result = extract_time_features(data, self.datetime_column)
        result, _ = scale_features(result, self.numerical_features, fit=False, scaler=self.scaler)
        
        if self.categorical_features:
            result, _ = encode_features(result, self.categorical_features, fit=False, encoder=self.encoder)
        
        if self.target_column and self.target_column in result.columns and self.target_scaler is not None:
            result[[self.target_column]] = self.target_scaler.transform(result[[self.target_column]])
        
        return result
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform standardized target values back to original scale.
        
        Parameters:
            y (np.ndarray): Standardized target values
        
        Returns:
            np.ndarray: Target values in original scale
        
        Raises:
            RuntimeError: If target_scaler is not available
        """
        if self.target_scaler is None:
            raise RuntimeError("Target scaler not available (target_column not set or not fitted)")
        
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()


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
    
    Note: For proper time series workflow without data leakage, use FeatureTransformer
    instead, which fits on training data only and transforms val/test separately.
    
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