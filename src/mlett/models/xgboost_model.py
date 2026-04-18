"""XGBoost model for time series forecasting."""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model wrapper for time series forecasting."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Parameters:
            model_params (Dict[str, Any]): XGBoost model parameters
        """
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params)
    
    @property
    def native_model(self):
        """
        Expose the underlying XGBRegressor instance.
        
        Use this to access framework-specific features:
            xgb.plot_importance(model.native_model)
            booster = model.native_model.get_booster()
        
        Returns None if the model has not been initialized yet.
        """
        return self._estimator
    
    def fit(self, X, y, eval_set: Optional[tuple] = None):
        """
        Train the XGBoost model.
        
        Parameters:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training target (pd.Series or np.ndarray)
            eval_set (tuple): Optional validation set (X_val, y_val)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()
        else:
            self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        self._estimator = xgb.XGBRegressor(**self.model_params)
        
        if eval_set:
            X_val, y_val = eval_set
            self._estimator.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self._estimator.fit(X, y)
        
        self.is_fitted = True
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        
        Parameters:
            X: Features to predict on (pd.DataFrame or np.ndarray)
        
        Returns:
            np.ndarray: Predicted values
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        return self._estimator.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns:
            pd.DataFrame: DataFrame with feature names and importance values
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        importances = self.native_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df