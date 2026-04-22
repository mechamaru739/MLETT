"""Trainer class for model training and evaluation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
from datetime import datetime

from mlett.models.xgboost_model import XGBoostModel
from mlett.utils.metrics import calculate_metrics, format_metrics
from mlett.utils.logger import setup_logger, get_timestamp
from mlett.utils.io import save_yaml


class Trainer:
    """Model trainer for time series forecasting."""
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs"
    ):
        """
        Initialize the trainer.
        
        Parameters:
            model_params (Dict[str, Any]): Model parameters
            log_dir (str): Directory for logging
        """
        self.model_params = model_params or {}
        self.log_dir = log_dir
        self.model = None
        self.scaler = None
        self.encoder = None
        self.logger = None
        self.training_history = {}
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the trainer."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "trainer.log")
        self.logger = setup_logger("Trainer", log_file)
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        model_type: str = "xgboost",
        target_inverse_fn=None,
        y_train_baseline=None,
        y_val_baseline=None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Parameters:
            X_train: Training features
            y_train: Training target (standardized; delta values if target_mode="delta")
            X_val: Validation features (optional)
            y_val: Validation target (optional, standardized; delta values if target_mode="delta")
            model_type (str): Type of model to train (default: "xgboost")
            target_inverse_fn: Function to inverse transform target to original scale (optional)
            y_train_baseline: Baseline values for delta mode reconstruction (optional)
                Shape must match y_train. In delta mode: absolute_std = y_delta_std + y_baseline_std
            y_val_baseline: Baseline values for delta mode on validation set (optional)
                Shape must match y_val.
        
        Returns:
            Dict[str, Any]: Training results and metrics (in original scale)
        """
        self.logger.info(f"Starting {model_type} model training...")
        self.logger.info(f"Training data shape: {X_train.shape}")
        is_delta = y_train_baseline is not None
        if is_delta:
            self.logger.info("Target mode: delta (predicting OT[t+h] - OT[t+h-1])")
        
        if model_type == "xgboost":
            self.model = XGBoostModel(self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            self.logger.info(f"Validation data shape: {X_val.shape}")
        
        self.model.fit(X_train, y_train, eval_set=eval_set)
        
        results = {
            'model_type': model_type,
            'training_samples': len(X_train),
            'feature_count': X_train.shape[1],
            'target_mode': 'delta' if is_delta else 'absolute',
            'timestamp': get_timestamp()
        }
        
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            if is_delta and y_val_baseline is not None:
                y_val_flat = np.asarray(y_val).flatten()
                baseline_flat = np.asarray(y_val_baseline).flatten()
                y_val_abs = y_val_flat + baseline_flat
                val_pred_abs = val_predictions + baseline_flat
                y_val_orig, val_pred_orig = self._inverse_transform(
                    y_val_abs, val_pred_abs, target_inverse_fn
                )
            else:
                y_val_orig, val_pred_orig = self._inverse_transform(
                    np.asarray(y_val), val_predictions, target_inverse_fn
                )
            val_metrics = calculate_metrics(y_val_orig, val_pred_orig)
            results['validation_metrics'] = val_metrics
            
            self.logger.info("Validation Metrics (original scale):")
            for metric, value in format_metrics(val_metrics).items():
                self.logger.info(f"  {metric}: {value}")
        
        train_predictions = self.model.predict(X_train)
        if is_delta and y_train_baseline is not None:
            y_train_flat = np.asarray(y_train).flatten()
            baseline_flat = np.asarray(y_train_baseline).flatten()
            y_train_abs = y_train_flat + baseline_flat
            train_pred_abs = train_predictions + baseline_flat
            y_train_orig, train_pred_orig = self._inverse_transform(
                y_train_abs, train_pred_abs, target_inverse_fn
            )
        else:
            y_train_orig, train_pred_orig = self._inverse_transform(
                np.asarray(y_train), train_predictions, target_inverse_fn
            )
        train_metrics = calculate_metrics(y_train_orig, train_pred_orig)
        results['training_metrics'] = train_metrics
        
        self.logger.info("Training Metrics (original scale):")
        for metric, value in format_metrics(train_metrics).items():
            self.logger.info(f"  {metric}: {value}")
        
        self.training_history = results
        self.logger.info("Model training completed successfully!")
        
        return results
    
    @staticmethod
    def _inverse_transform(y_true_std, y_pred_std, fn):
        """
        Inverse transform standardized values to original scale.
        
        Parameters:
            y_true_std: Standardized true values
            y_pred_std: Standardized predicted values
            fn: Inverse transform function, or None
        
        Returns:
            Tuple of (y_true_original, y_pred_original)
        """
        if fn is not None:
            return fn(y_true_std), fn(y_pred_std)
        return y_true_std, y_pred_std
    
    def evaluate(
        self,
        X_test,
        y_test,
        target_inverse_fn=None,
        y_baseline=None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
            X_test: Test features
            y_test: Test target (standardized; delta values if target_mode="delta")
            target_inverse_fn: Function to inverse transform target to original scale (optional)
            y_baseline: Baseline values for delta mode reconstruction (optional)
                Shape must match y_test. In delta mode: absolute_std = y_delta_std + y_baseline_std
        
        Returns:
            Dict[str, float]: Evaluation metrics (in original scale)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        is_delta = y_baseline is not None
        self.logger.info(f"Evaluating model on test data: {X_test.shape}")
        if is_delta:
            self.logger.info("Target mode: delta (reconstructing absolute values from delta + baseline)")
        
        predictions = self.model.predict(X_test)
        
        if is_delta and y_baseline is not None:
            y_flat = np.asarray(y_test).flatten()
            baseline_flat = np.asarray(y_baseline).flatten()
            y_abs = y_flat + baseline_flat
            pred_abs = predictions + baseline_flat
            y_test_orig, pred_orig = self._inverse_transform(
                y_abs, pred_abs, target_inverse_fn
            )
        else:
            y_test_orig, pred_orig = self._inverse_transform(
                np.asarray(y_test), predictions, target_inverse_fn
            )
        metrics = calculate_metrics(y_test_orig, pred_orig)
        
        self.logger.info("Test Metrics (original scale):")
        for metric, value in format_metrics(metrics).items():
            self.logger.info(f"  {metric}: {value}")
        
        return metrics
    
    def save_training_results(
        self,
        results_dir: str = "results",
        experiment_name: Optional[str] = None
    ) -> str:
        """
        Save model and training results to an experiment directory.
        
        All files for one experiment are saved in a single directory:
            results/<experiment_name>/
                model.pkl
                model_results.yaml
        
        Parameters:
            results_dir (str): Root directory for all experiment results
            experiment_name (str): Name of the experiment (used as subdirectory name)
        
        Returns:
            str: Path to the experiment directory
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")
        
        if experiment_name is None:
            experiment_name = get_timestamp()
        
        experiment_dir = os.path.join(results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        model_path = os.path.join(experiment_dir, "model.pkl")
        results_path = os.path.join(experiment_dir, "model_results.yaml")
        
        self.model.save_model(model_path)
        save_yaml(self.training_history, results_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info(f"Results saved to: {results_path}")
        
        return experiment_dir
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Parameters:
            model_path (str): Path to saved model
        """
        if self.model is None:
            self.model = XGBoostModel()
        
        self.model.load_model(model_path)
        self.logger.info(f"Model loaded from: {model_path}")