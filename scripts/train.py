"""Main training script for MLETT time series forecasting."""

import argparse
import os
import yaml

from mlett.data.preprocessing import load_data, clean_data
from mlett.data.time_series_split import time_series_split, create_sliding_windows
from mlett.features.engineering import FeatureTransformer
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger, get_timestamp
from mlett.utils.io import save_yaml, save_model
from mlett.utils.seed import set_random_seed


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
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    set_random_seed(seed)
    
    # Setup logging
    logger = setup_logger("MainTrain", os.path.join(config['paths']['logs_dir'], f"main_train_{get_timestamp()}.log"))
    logger.info("Starting MLETT training pipeline")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Random seed set to: {seed}")
    
    logger.info("=" * 60)
    logger.info("SAMPLE DEFINITION (Sliding Window):")
    logger.info("  Input X:  past window_size hours of all features")
    logger.info("  Output y: future forecast_horizon hours of OT")
    logger.info(f"  window_size = {config['features']['window_size']}")
    logger.info(f"  forecast_horizon = {config['features']['forecast_horizon']}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load data
        logger.info(f"Loading data from: {config['data']['raw_data_path']}")
        data = load_data(config['data']['raw_data_path'])
        logger.info(f"Raw data shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        
        # Step 2: Clean data
        logger.info("Cleaning data...")
        data = clean_data(data)
        logger.info(f"Cleaned data shape: {data.shape}")
        
        # Step 3: Chronological split FIRST (avoid data leakage)
        logger.info("Splitting raw data into train/val/test (before feature engineering)...")
        train_raw, val_raw, test_raw = time_series_split(
            data,
            train_ratio=config['split']['train_ratio'],
            val_ratio=config['split']['val_ratio'],
            test_ratio=config['split']['test_ratio']
        )
        logger.info(f"Raw split -> Train: {train_raw.shape}, Val: {val_raw.shape}, Test: {test_raw.shape}")
        
        # Step 4: Feature engineering with anti-leakage fitting
        logger.info("Performing feature engineering (fit on train only)...")
        transformer = FeatureTransformer(
            datetime_column=config['data']['datetime_column'],
            numerical_features=config['data']['numerical_features'],
            categorical_features=config['data']['categorical_features'] or None
        )
        
        train_data = transformer.fit_transform(train_raw)
        val_data = transformer.transform(val_raw)
        test_data = transformer.transform(test_raw)
        
        logger.info(f"Feature-engineered -> Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        logger.info(f"Feature columns: {[c for c in train_data.columns if c != config['data']['target_column']]}")
        
        # Step 5: Build sliding window samples
        logger.info("Building sliding window samples...")
        window_size = config['features']['window_size']
        horizon = config['features']['forecast_horizon']
        step = config['features']['window_step']
        target_column = config['data']['target_column']
        
        X_train, y_train = create_sliding_windows(train_data, target_column, window_size, horizon, step)
        X_val, y_val = create_sliding_windows(val_data, target_column, window_size, horizon, step)
        X_test, y_test = create_sliding_windows(test_data, target_column, window_size, horizon, step)
        
        logger.info(f"Window samples -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Target shapes -> Train y: {y_train.shape}, Val y: {y_val.shape}, Test y: {y_test.shape}")
        logger.info(f"Each sample: {window_size} time steps x {X_train.shape[1] // window_size} features = {X_train.shape[1]} dimensions")
        
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
        experiment_name = args.model_name or get_timestamp()
        
        if config['training']['save_model']:
            logger.info("Saving model and results...")
            experiment_dir = trainer.save_training_results(
                results_dir=config['paths']['results_dir'],
                experiment_name=experiment_name
            )
            
            # Save transformer for inference
            import joblib
            transformer_path = os.path.join(experiment_dir, "transformer.pkl")
            joblib.dump(transformer, transformer_path)
            logger.info(f"Feature transformer saved to: {transformer_path}")
            
            # Save config snapshot
            config_snapshot_path = os.path.join(experiment_dir, "config.yaml")
            save_yaml(config, config_snapshot_path)
            logger.info(f"Config snapshot saved to: {config_snapshot_path}")
            
            # Save test results with sample definition info
            results_summary = {
                'training_results': training_results,
                'test_metrics': test_metrics,
                'sample_definition': {
                    'window_size': window_size,
                    'forecast_horizon': horizon,
                    'window_step': step,
                    'input_dimensions': X_train.shape[1],
                    'description': f'Each sample uses past {window_size} hours of features to predict next {horizon} hour(s) of OT'
                },
                'timestamp': get_timestamp()
            }
            
            results_path = os.path.join(experiment_dir, "results.yaml")
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