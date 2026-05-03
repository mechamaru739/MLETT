"""LightGBM model for time series forecasting."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model wrapper for time series forecasting."""

    model_type = "lightgbm"
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model.

        Parameters:
            model_params (Dict[str, Any]): LightGBM model parameters
        """
        default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 5,
            'num_leaves': 31,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        if model_params:
            default_params.update(model_params)

        super().__init__(default_params)
        self.eval_history: Dict[str, List[float]] = {}

    @property
    def native_model(self):
        """
        Expose the underlying LGBMRegressor instance.

        Returns None if the model has not been initialized yet.
        """
        return self._estimator

    def fit(self, X, y, eval_set: Optional[tuple] = None):
        """
        Train the LightGBM model.

        Parameters:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training target (pd.Series or np.ndarray)
            eval_set (tuple): Optional validation set (X_val, y_val)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()
        else:
            self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]

        early_stopping_rounds = self.model_params.pop('early_stopping_rounds', None)
        
        self._estimator = lgb.LGBMRegressor(**self.model_params)
        self.eval_history = {}

        # Convert y to 1d array to avoid DataConversionWarning
        y_1d = np.asarray(y).ravel()

        fit_params = {}
        if eval_set:
            X_val, y_val = eval_set
            fit_params['eval_set'] = [(X_val, np.asarray(y_val).ravel())]
            callbacks = [lgb.log_evaluation(period=0)]
            if early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
            fit_params['callbacks'] = callbacks

        self._estimator.fit(X, y_1d, **fit_params)

        if eval_set and hasattr(self._estimator, 'evals_result_'):
            self.eval_history = self._estimator.evals_result_

        self.is_fitted = True

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained LightGBM model.

        Parameters:
            X: Features to predict on (pd.DataFrame or np.ndarray)

        Returns:
            np.ndarray: Predicted values

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Convert numpy array to DataFrame with feature names to avoid warning
        if isinstance(X, np.ndarray) and self.feature_columns:
            X = pd.DataFrame(X, columns=self.feature_columns)

        return self._estimator.predict(X)

    def get_eval_scores(self, metric_name: str = 'mae') -> List[float]:
        """
        Get per-iteration validation scores from the last fit.

        Parameters:
            metric_name (str): Metric name to extract (default: 'mae')

        Returns:
            List[float]: Per-iteration validation scores, or empty list if unavailable
        """
        if not self.eval_history:
            return []

        validation_key = 'valid_0'
        if validation_key not in self.eval_history:
            return []

        available_metrics = self.eval_history[validation_key]
        if metric_name in available_metrics:
            return available_metrics[metric_name]

        return list(available_metrics.values())[0] if available_metrics else []

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

        importances = self._estimator.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df