"""Main training script for MLETT time series forecasting."""

import argparse
import os
import yaml

from mlett.data.preprocessing import load_data, clean_data
from mlett.data.time_series_split import time_series_split
from mlett.features.engineering import feature_pipeline
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger, get_timestamp
from mlett.utils.io import save_yaml


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters:
        config_path (str): Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train time series forecasting model')
    parser.add_argument('--config', type=str, default='src/mlett/config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name for the saved model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger("MainTrain", os.path.join(config['paths']['logs_dir'], f"main_train_{get_timestamp()}.log"))
    logger.info("Starting MLETT training pipeline")
    logger.info(f"Configuration loaded from: {args.config}")
    
    try:
        # Step 1: Load data
        logger.info(f"Loading data from: {config['data']['raw_data_path']}")
        data = load_data(config['data']['raw_data_path'])
        logger.info(f"Raw data shape: {data.shape}")
        
        # Step 2: Clean data
        logger.info("Cleaning data...")
        data = clean_data(data)
        logger.info(f"Cleaned data shape: {data.shape}")
        
        # Step 3: Feature engineering
        logger.info("Performing feature engineering...")
        processed_data, scaler, encoder = feature_pipeline(
            data=data,
            datetime_column=config['data']['datetime_column'],
            numerical_features=config['data']['numerical_features'],
            categorical_features=config['data']['categorical_features'],
            fit_scaler=True,
            fit_encoder=True
        )
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Step 4: Split data
        logger.info("Splitting data into train/val/test sets...")
        train_data, val_data, test_data = time_series_split(
            processed_data,
            train_ratio=config['split']['train_ratio'],
            val_ratio=config['split']['val_ratio'],
            test_ratio=config['split']['test_ratio']
        )
        logger.info(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")
        
        # Step 5: Prepare features and targets
        target_column = config['data']['target_column']
        feature_columns = [col for col in processed_data.columns if col != target_column]
        
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_val = val_data[feature_columns]
        y_val = val_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        logger.info(f"Feature columns count: {len(feature_columns)}")
        
        # Step 6: Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model_params=config['model']['xgboost'],
            log_dir=config['paths']['logs_dir']
        )
        
        # Step 7: Train model
        logger.info("Training model...")
        if config['training']['use_validation']:
            training_results = trainer.train(
                X_train, y_train,
                X_val, y_val,
                model_type=config['model']['type']
            )
        else:
            training_results = trainer.train(
                X_train, y_train,
                model_type=config['model']['type']
            )
        
        # Step 8: Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = trainer.evaluate(X_test, y_test)
        
        # Step 9: Save model and results
        if config['training']['save_model']:
            logger.info("Saving model and results...")
            trainer.save_training_results(
                save_dir=config['paths']['models_dir'],
                model_name=args.model_name
            )
            
            # Save test results
            results_summary = {
                'training_results': training_results,
                'test_metrics': test_metrics,
                'config': config,
                'timestamp': get_timestamp()
            }
            
            results_path = os.path.join(
                config['paths']['results_dir'],
                f"results_{args.model_name or get_timestamp()}.yaml"
            )
            os.makedirs(config['paths']['results_dir'], exist_ok=True)
            save_yaml(results_summary, results_path)
            logger.info(f"Results saved to: {results_path}")
        
        logger.info("Training pipeline completed successfully!")
        
        return {
            'training_results': training_results,
            'test_metrics': test_metrics,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()