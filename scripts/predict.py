"""Prediction script for trained models."""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.xgboost_model import XGBoostModel
from utils.logger import setup_logger, get_timestamp
from utils.io import save_dataframe


def make_predictions(
    model_path: str,
    data_path: str,
    output_path: str
):
    """
    Make predictions using a trained model.
    
    Parameters:
        model_path (str): Path to trained model
        data_path (str): Path to input data
        output_path (str): Path to save predictions
    """
    # Setup logging
    logger = setup_logger("Predict", f"logs/predict_{get_timestamp()}.log")
    
    try:
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = XGBoostModel()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load data
        logger.info(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Data shape: {data.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(data)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions
        })
        
        # Save predictions
        logger.info(f"Saving predictions to: {output_path}")
        save_dataframe(results, output_path)
        
        # Summary statistics
        logger.info("Prediction Statistics:")
        logger.info(f"  Mean: {predictions.mean():.4f}")
        logger.info(f"  Std: {predictions.std():.4f}")
        logger.info(f"  Min: {predictions.min():.4f}")
        logger.info(f"  Max: {predictions.max():.4f}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        raise


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save predictions file')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    predictions = make_predictions(
        args.model,
        args.data,
        args.output
    )
    
    print(f"Predictions completed. Results saved to {args.output}")
    return predictions


if __name__ == "__main__":
    predictions = main()