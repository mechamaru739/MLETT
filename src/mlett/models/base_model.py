"""Base model class for time series forecasting."""

import abc
import joblib
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseModel(abc.ABC):
    """Abstract base class for time series forecasting models."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Parameters:
            model_params (Dict[str, Any]): Model parameters (optional)
        """
        self.model_params = model_params or {}
        self._estimator = None
        self.feature_columns = None
        self.is_fitted = False
    
    @property
    @abc.abstractmethod
    def native_model(self):
        """Return the underlying native model instance (e.g. XGBRegressor, nn.Module)."""
        pass
    
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on the given data.
        
        Parameters:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
        """
        pass
    
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
            X (pd.DataFrame): Features to predict on
        
        Returns:
            np.ndarray: Predicted values
        """
        pass
    
    def save_model(self, filepath: str):
        """
        Save the trained model to a file.
        
        Parameters:
            filepath (str): Path to save the model
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        feature_importances = None
        if self._estimator is not None and hasattr(self._estimator, 'feature_importances_'):
            feature_importances = self._estimator.feature_importances_
        
        model_data = {
            'model': self._estimator,
            'model_params': self.model_params,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'feature_importances_': feature_importances
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model from a file.
        
        Parameters:
            filepath (str): Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self._estimator = model_data.get('model')
        self.model_params = model_data['model_params']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']
        self.feature_importances_ = model_data.get('feature_importances_')
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        return self.model_params
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Parameters:
            **params: Parameter names and values
        """
        self.model_params.update(params)