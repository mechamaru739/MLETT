"""Evaluation script for trained models."""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

from mlett.models.base_model import BaseModel
from mlett.data.preprocessing import clean_data
from mlett.data.time_series_split import create_sliding_windows
from mlett.features.industrial_features import create_industrial_windows, TIME_FEATURE_COLUMNS
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
        model = BaseModel.load_model_from_file(model_path)
        logger.info(f"Model loaded successfully (type: {model.model_type})")
        
        logger.info(f"Loading transformer from: {transformer_path}")
        transformer = joblib.load(transformer_path)
        logger.info("Transformer loaded successfully")
        
        config = load_yaml(config_path)
        target_column = config['data']['target_column']
        window_size = config['features']['window_size']
        horizon = config['features']['forecast_horizon']
        step = config['features']['window_step']
        target_mode = config.get('features', {}).get('target_mode', 'absolute')
        feature_mode = config.get('features', {}).get('feature_mode', 'flat')
        sensor_columns = config['data']['numerical_features']
        time_columns = config.get('features', {}).get('time_columns', TIME_FEATURE_COLUMNS)
        
        logger.info(f"Feature mode: {feature_mode}")
        logger.info(f"Target mode: {target_mode}")
        
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
        
        # Build window samples
        logger.info("Building window samples...")
        inverse_fn = transformer.inverse_transform_target
        
        if feature_mode == "industrial":
            if target_mode == "delta":
                X, y, y_baseline = create_industrial_windows(
                    processed_data, sensor_columns, time_columns, target_column,
                    window_size, horizon, step, target_mode="delta"
                )
            else:
                X, y = create_industrial_windows(
                    processed_data, sensor_columns, time_columns, target_column,
                    window_size, horizon, step, target_mode="absolute"
                )
                y_baseline = None
        else:
            if target_mode == "delta":
                X, y, y_baseline = create_sliding_windows(
                    processed_data, target_column, window_size, horizon, step, target_mode="delta"
                )
            else:
                X, y = create_sliding_windows(
                    processed_data, target_column, window_size, horizon, step, target_mode="absolute"
                )
                y_baseline = None
        
        logger.info(f"Window samples shape: X={X.shape}, y={y.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_std = model.predict(X)
        
        # Reconstruct absolute values and inverse transform to original scale
        if target_mode == "delta" and y_baseline is not None:
            logger.info("Delta mode: reconstructing absolute values from delta + baseline")
            y_flat = y.flatten()
            baseline_flat = y_baseline.flatten()
            y_abs = y_flat + baseline_flat
            pred_abs = predictions_std + baseline_flat
            y_original = inverse_fn(y_abs)
            predictions_original = inverse_fn(pred_abs)
        else:
            y_original = inverse_fn(y)
            predictions_original = inverse_fn(predictions_std)
        
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
            'feature_mode': feature_mode,
            'target_mode': target_mode,
            'metrics': metrics,
            'sample_count': len(X),
            'sample_definition': {
                'window_size': window_size,
                'forecast_horizon': horizon,
                'window_step': step,
                'feature_mode': feature_mode,
                'target_mode': target_mode
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