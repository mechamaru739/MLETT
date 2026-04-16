"""Prediction script for trained models."""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

from mlett.models.xgboost_model import XGBoostModel
from mlett.data.preprocessing import clean_data
from mlett.data.time_series_split import create_sliding_windows
from mlett.utils.logger import setup_logger, get_timestamp
from mlett.utils.io import save_yaml, load_yaml


def make_predictions(
    experiment_dir: str,
    data_path: str,
    output_path: str
):
    """
    Make predictions using a trained model from an experiment directory.
    
    Parameters:
        experiment_dir (str): Path to experiment directory
        data_path (str): Path to raw input CSV data
        output_path (str): Path to save predictions
    """
    logger = setup_logger("Predict", os.path.join(experiment_dir, "predict.log"))
    
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
        logger.info(f"Window samples shape: {X.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions.flatten()
        })
        
        if y is not None and len(y) > 0:
            results['actual'] = y.flatten()
        
        # Save predictions
        logger.info(f"Saving predictions to: {output_path}")
        results.to_csv(output_path, index=False)
        
        # Summary statistics
        logger.info("Prediction Statistics:")
        logger.info(f"  Mean: {float(predictions.mean()):.4f}")
        logger.info(f"  Std:  {float(predictions.std()):.4f}")
        logger.info(f"  Min:  {float(predictions.min()):.4f}")
        logger.info(f"  Max:  {float(predictions.max()):.4f}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        raise


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to experiment directory (e.g. results/20260414_210510)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to raw input CSV data file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions file (default: <experiment>/predictions.csv)')
    
    args = parser.parse_args()
    
    output_path = args.output or os.path.join(args.experiment, "predictions.csv")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    predictions = make_predictions(
        args.experiment,
        args.input,
        output_path
    )
    
    print(f"Predictions completed. Results saved to {output_path}")
    return predictions


if __name__ == "__main__":
    predictions = main()