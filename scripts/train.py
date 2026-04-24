"""Main training script for MLETT time series forecasting."""

import argparse
import os
import yaml
import copy

from mlett.data.preprocessing import load_data, clean_data
from mlett.data.time_series_split import time_series_split, create_sliding_windows
from mlett.features.engineering import FeatureTransformer
from mlett.features.industrial_features import create_industrial_windows, TIME_FEATURE_COLUMNS
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger, get_timestamp
from mlett.utils.io import save_yaml
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


def run_training(config: dict, experiment_name: str) -> dict:
    """
    Complete training pipeline, callable from train.py or tune.py.
    
    Parameters:
        config (dict): Full configuration dictionary
        experiment_name (str): Name of the experiment (used as directory name)
    
    Returns:
        dict: Training results including training_results, test_metrics, and status
    """
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    set_random_seed(seed)
    
    # Determine experiment directory
    experiment_dir = os.path.join(config['paths']['results_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging to experiment directory
    log_file = os.path.join(experiment_dir, "train.log")
    logger = setup_logger("MainTrain", log_file)
    logger.info("Starting MLETT training pipeline")
    logger.info(f"Random seed set to: {seed}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Log model parameters prominently for tuning comparison
    xgb_params = config.get('model', {}).get('xgboost', {})
    logger.info("=" * 60)
    logger.info("MODEL PARAMETERS:")
    for k, v in xgb_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
    # Determine target mode and feature mode
    target_mode = config.get('features', {}).get('target_mode', 'absolute')
    feature_mode = config.get('features', {}).get('feature_mode', 'flat')
    sensor_columns = config['data']['numerical_features']
    time_columns = config.get('features', {}).get('time_columns', TIME_FEATURE_COLUMNS)
    
    logger.info("=" * 60)
    logger.info("SAMPLE DEFINITION (Sliding Window):")
    logger.info(f"  Feature mode: {feature_mode}")
    logger.info(f"  Target mode: {target_mode}")
    if target_mode == "delta":
        logger.info("  (delta mode: y = OT[t+h] - OT[t+h-1], baseline = OT[t+h-1] for reconstruction)")
    logger.info("=" * 60)
    
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
    target_column = config['data']['target_column']
    transformer = FeatureTransformer(
        datetime_column=config['data']['datetime_column'],
        numerical_features=config['data']['numerical_features'],
        target_column=target_column,
        categorical_features=config['data']['categorical_features'] or None
    )
    
    train_data = transformer.fit_transform(train_raw)
    val_data = transformer.transform(val_raw)
    test_data = transformer.transform(test_raw)
    
    logger.info(f"Feature-engineered -> Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    logger.info(f"Feature columns: {[c for c in train_data.columns if c != target_column]}")
    
    # Step 5: Build sliding window samples
    logger.info("Building sliding window samples...")
    window_size = config['features']['window_size']
    horizon = config['features']['forecast_horizon']
    step = config['features']['window_step']
    
    if feature_mode == "industrial":
        logger.info(f"Using industrial features (compact: ~42d per sample)")
        logger.info(f"  Sensor columns: {sensor_columns}")
        logger.info(f"  Time columns: {time_columns}")
        
        if target_mode == "delta":
            X_train, y_train, y_train_baseline = create_industrial_windows(
                train_data, sensor_columns, time_columns, target_column,
                window_size, horizon, step, target_mode="delta"
            )
            X_val, y_val, y_val_baseline = create_industrial_windows(
                val_data, sensor_columns, time_columns, target_column,
                window_size, horizon, step, target_mode="delta"
            )
            X_test, y_test, y_test_baseline = create_industrial_windows(
                test_data, sensor_columns, time_columns, target_column,
                window_size, horizon, step, target_mode="delta"
            )
            logger.info(f"Delta mode: y shape = {y_train.shape}, y_baseline shape = {y_train_baseline.shape}")
        else:
            X_train, y_train = create_industrial_windows(
                train_data, sensor_columns, time_columns, target_column,
                window_size, horizon, step, target_mode="absolute"
            )
            X_val, y_val = create_industrial_windows(
                val_data, sensor_columns, time_columns, target_column,
                window_size, horizon, step, target_mode="absolute"
            )
            X_test, y_test = create_industrial_windows(
                test_data, sensor_columns, time_columns, target_column,
                window_size, horizon, step, target_mode="absolute"
            )
            y_train_baseline = None
            y_val_baseline = None
            y_test_baseline = None
    else:
        logger.info(f"Using flat features (full flatten: {train_data.shape[1]-1} features x {window_size} steps)")
        
        if target_mode == "delta":
            X_train, y_train, y_train_baseline = create_sliding_windows(
                train_data, target_column, window_size, horizon, step, target_mode="delta"
            )
            X_val, y_val, y_val_baseline = create_sliding_windows(
                val_data, target_column, window_size, horizon, step, target_mode="delta"
            )
            X_test, y_test, y_test_baseline = create_sliding_windows(
                test_data, target_column, window_size, horizon, step, target_mode="delta"
            )
            logger.info(f"Delta mode: y shape = {y_train.shape}, y_baseline shape = {y_train_baseline.shape}")
        else:
            X_train, y_train = create_sliding_windows(
                train_data, target_column, window_size, horizon, step, target_mode="absolute"
            )
            y_train_baseline = None
            X_val, y_val = create_sliding_windows(
                val_data, target_column, window_size, horizon, step, target_mode="absolute"
            )
            y_val_baseline = None
            X_test, y_test = create_sliding_windows(
                test_data, target_column, window_size, horizon, step, target_mode="absolute"
            )
            y_test_baseline = None
    
    logger.info(f"Window samples -> Train: X={X_train.shape}, Val: X={X_val.shape}, Test: X={X_test.shape}")
    logger.info(f"Target shapes -> Train y: {y_train.shape}, Val y: {y_val.shape}, Test y: {y_test.shape}")
    if feature_mode == "industrial":
        logger.info(f"Industrial features per sample: {X_train.shape[1]} dimensions")
    else:
        n_features = X_train.shape[1] // window_size
        logger.info(f"Each sample: {window_size} time steps x {n_features} features = {X_train.shape[1]} dimensions")
    
    # Step 6: Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model_params=config['model']['xgboost'],
        log_dir=experiment_dir
    )
    
    # Step 7: Train model
    logger.info("Training model...")
    inverse_fn = transformer.inverse_transform_target
    
    if config['training']['use_validation']:
        training_results = trainer.train(
            X_train, y_train,
            X_val, y_val,
            model_type=config['model']['type'],
            target_inverse_fn=inverse_fn,
            y_train_baseline=y_train_baseline,
            y_val_baseline=y_val_baseline
        )
    else:
        training_results = trainer.train(
            X_train, y_train,
            model_type=config['model']['type'],
            target_inverse_fn=inverse_fn,
            y_train_baseline=y_train_baseline
        )
    
    # Step 8: Evaluate on test set
    logger.info("Evaluating model on test set...")
    test_metrics = trainer.evaluate(
        X_test, y_test,
        target_inverse_fn=inverse_fn,
        y_baseline=y_test_baseline
    )
    
    # Step 9: Save model and results
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
                'feature_mode': feature_mode,
                'target_mode': target_mode,
                'description': f'Each sample uses past {window_size} hours of features to predict next {horizon} hour(s) of OT'
            },
            'model_params': config['model']['xgboost'],
            'timestamp': get_timestamp()
        }
        
        if feature_mode == "industrial":
            results_summary['sample_definition']['sensor_columns'] = sensor_columns
            results_summary['sample_definition']['time_columns'] = time_columns
        
        results_path = os.path.join(experiment_dir, "results.yaml")
        save_yaml(results_summary, results_path)
        logger.info(f"Results saved to: {results_path}")
    
    logger.info("Training pipeline completed successfully!")
    
    return {
        'experiment_name': experiment_name,
        'training_results': training_results,
        'test_metrics': test_metrics,
        'model_params': config['model']['xgboost'],
        'status': 'success'
    }


def main():
    """Entry point for standalone training."""
    parser = argparse.ArgumentParser(description='Train time series forecasting model')
    parser.add_argument('--config', type=str, default='src/mlett/config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name for the saved model')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    experiment_name = args.model_name or get_timestamp()
    
    return run_training(config, experiment_name)


if __name__ == "__main__":
    results = main()