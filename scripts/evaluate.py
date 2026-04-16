"""Evaluation script for trained models."""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

from mlett.models.xgboost_model import XGBoostModel
from mlett.data.preprocessing import clean_data
from mlett.data.time_series_split import create_sliding_windows
from mlett.utils.metrics import calculate_metrics, format_metrics
from mlett.utils.logger import setup_logger, get_timestamp
from mlett.utils.io import load_yaml, save_yaml


def evaluate_model(
    experiment_dir: str,
    data_path: str
):
    """
    Evaluate a trained model on test data using artifacts from an experiment directory.
    
    Parameters:
        experiment_dir (str): Path to experiment directory
        data_path (str): Path to raw test CSV data
    """
    logger = setup_logger("Evaluate", os.path.join(experiment_dir, "evaluate.log"))
    
    try:
        # Load experiment artifacts
        model_path = os.path.join(experiment_dir, "model.pkl")
        transformer_path = os.path.join(experiment_dir, "transformer.pkl")
        config_path = os.path.join(experiment_dir, "config.yaml")
        
        logger.info(f"Loading model from: {model_path}")
        model = XGBoostModel()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        
        logger.info(f"Loading transformer from: {transformer_path}")
        transformer = joblib.load(transformer_path)
        logger.info("Transformer loaded successfully")
        
        config = load_yaml(config_path)
        target_column = config['data']['target_column']
        window_size = config['features']['window_size']
        horizon = config['features']['forecast_horizon']
        step = config['features']['window_step']
        
        # Load and preprocess raw data
        logger.info(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Raw data shape: {data.shape}")
        
        data = clean_data(data)
        logger.info(f"Cleaned data shape: {data.shape}")
        
        # Feature engineering using saved transformer
        logger.info("Applying feature transformation...")
        processed_data = transformer.transform(data)
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Build sliding window samples
        logger.info("Building sliding window samples...")
        X, y = create_sliding_windows(processed_data, target_column, window_size, horizon, step)
        logger.info(f"Window samples shape: X={X.shape}, y={y.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_std = model.predict(X)
        
        # Inverse transform to original scale for metrics
        y_original = transformer.inverse_transform_target(y)
        predictions_original = transformer.inverse_transform_target(predictions_std)
        logger.info("Predictions and targets inverse-transformed to original scale")
        
        # Calculate metrics in original scale
        logger.info("Calculating evaluation metrics (original scale)...")
        metrics = calculate_metrics(y_original, predictions_original)
        
        # Display results
        logger.info("Evaluation Results:")
        for metric, value in format_metrics(metrics).items():
            logger.info(f"  {metric}: {value}")
        
        # Save results
        results = {
            'data_path': data_path,
            'metrics': metrics,
            'sample_count': len(X),
            'sample_definition': {
                'window_size': window_size,
                'forecast_horizon': horizon,
                'window_step': step
            },
            'timestamp': get_timestamp()
        }
        
        results_path = os.path.join(experiment_dir, "evaluation_results.yaml")
        save_yaml(results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to experiment directory (e.g. results/20260414_210510)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to raw test CSV data file')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        args.experiment,
        args.data
    )
    
    print(f"Evaluation completed. Results saved to {args.experiment}")
    return results


if __name__ == "__main__":
    results = main()