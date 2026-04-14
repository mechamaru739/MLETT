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
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Parameters:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation target (optional)
            model_type (str): Type of model to train (default: "xgboost")
        
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        self.logger.info(f"Starting {model_type} model training...")
        self.logger.info(f"Training data shape: {X_train.shape}")
        
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
            'timestamp': get_timestamp()
        }
        
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_metrics = calculate_metrics(np.asarray(y_val), val_predictions)
            results['validation_metrics'] = val_metrics
            
            self.logger.info("Validation Metrics:")
            for metric, value in format_metrics(val_metrics).items():
                self.logger.info(f"  {metric}: {value}")
        
        train_predictions = self.model.predict(X_train)
        train_metrics = calculate_metrics(np.asarray(y_train), train_predictions)
        results['training_metrics'] = train_metrics
        
        self.logger.info("Training Metrics:")
        for metric, value in format_metrics(train_metrics).items():
            self.logger.info(f"  {metric}: {value}")
        
        self.training_history = results
        self.logger.info("Model training completed successfully!")
        
        return results
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        self.logger.info(f"Evaluating model on test data: {X_test.shape}")
        
        predictions = self.model.predict(X_test)
        metrics = calculate_metrics(np.asarray(y_test), predictions)
        
        self.logger.info("Test Metrics:")
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