"""Evaluation script for trained models."""

import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.xgboost_model import XGBoostModel
from utils.metrics import calculate_metrics, format_metrics
from utils.logger import setup_logger, get_timestamp
from utils.io import load_yaml, save_yaml


def evaluate_model(
    model_path: str,
    data_path: str,
    config_path: str,
    output_dir: str
):
    """
    Evaluate a trained model on test data.
    
    Parameters:
        model_path (str): Path to trained model
        data_path (str): Path to test data
        config_path (str): Path to configuration file
        output_dir (str): Directory to save evaluation results
    """
    # Setup logging
    logger = setup_logger("Evaluate", os.path.join(output_dir, f"evaluate_{get_timestamp()}.log"))
    
    # Load configuration
    config = load_yaml(config_path)
    logger.info(f"Configuration loaded from: {config_path}")
    
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
        
        # Prepare features and target
        target_column = config['data']['target_column']
        feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(X)
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        metrics = calculate_metrics(y.values, predictions)
        
        # Display results
        logger.info("Evaluation Results:")
        for metric, value in format_metrics(metrics).items():
            logger.info(f"  {metric}: {value}")
        
        # Save results
        results = {
            'model_path': model_path,
            'data_path': data_path,
            'metrics': metrics,
            'timestamp': get_timestamp(),
            'sample_count': len(data)
        }
        
        results_path = os.path.join(output_dir, f"evaluation_results_{get_timestamp()}.yaml")
        save_yaml(results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data file')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    results = evaluate_model(
        args.model,
        args.data,
        args.config,
        args.output
    )
    
    print(f"Evaluation completed. Results saved to {args.output}")
    return results


if __name__ == "__main__":
    results = main()