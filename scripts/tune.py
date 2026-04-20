"""Grid search hyperparameter tuning script for MLETT."""

import argparse
import copy
import itertools
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_config, run_training
from mlett.utils.seed import set_random_seed
from mlett.utils.io import save_yaml
from mlett.utils.logger import setup_logger, get_timestamp


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a nested dictionary with values from update_dict.
    
    Parameters:
        base_dict (dict): Base dictionary to update
        update_dict (dict): Dictionary with new values
    
    Returns:
        dict: Updated dictionary
    """
    result = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def generate_grid(param_space: dict) -> list:
    """
    Generate all combinations of parameters from a grid search space.
    
    Parameters:
        param_space (dict): Dictionary of parameter names to lists of values
    
    Returns:
        list: List of parameter dictionaries (one per combination)
    """
    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def format_param_short(param_name: str, value) -> str:
    """
    Format a parameter value into a short readable string for experiment names.
    
    Parameters:
        param_name (str): Parameter name
        value: Parameter value
    
    Returns:
        str: Short formatted string
    """
    short_names = {
        'max_depth': 'd',
        'learning_rate': 'lr',
        'n_estimators': 'ne',
        'min_child_weight': 'mcw',
        'subsample': 'ss',
        'colsample_bytree': 'cbt',
        'colsample_bylevel': 'cbl',
        'reg_alpha': 'ra',
        'reg_lambda': 'rl',
    }
    short = short_names.get(param_name, param_name[:4])
    if isinstance(value, float):
        return f"{short}{str(value).replace('.', '')}"
    return f"{short}{value}"


def main():
    parser = argparse.ArgumentParser(description='Grid search hyperparameter tuning')
    parser.add_argument('--tune-config', type=str, default='src/mlett/config/tune_config.yaml',
                        help='Path to tuning configuration file')
    parser.add_argument('--max-experiments', type=int, default=None,
                        help='Maximum number of experiments to run (for testing)')
    
    args = parser.parse_args()
    
    # Load tuning configuration
    with open(args.tune_config, 'r') as f:
        tune_config = yaml.safe_load(f)
    
    # Load base configuration
    base_config = load_config(tune_config['base_config'])
    
    # Create timestamped tune directory
    tune_dir_name = f"tune_{get_timestamp()}"
    tune_dir = os.path.join(base_config['paths']['results_dir'], tune_dir_name)
    os.makedirs(tune_dir, exist_ok=True)
    
    # Generate parameter grid
    param_space = tune_config['param_space']
    param_combos = generate_grid(param_space)
    
    total = len(param_combos)
    if args.max_experiments:
        total = min(total, args.max_experiments)
        param_combos = param_combos[:total]
    
    print("=" * 70)
    print("MLETT Hyperparameter Grid Search")
    print("=" * 70)
    print(f"Tuning method: {tune_config['tune_method']}")
    print(f"Base config: {tune_config['base_config']}")
    print(f"Parameter space:")
    for k, v in param_space.items():
        print(f"  {k}: {v}")
    print(f"Total experiments: {total}")
    print(f"Results directory: {tune_dir}")
    print("=" * 70)
    
    # Run experiments
    all_results = []
    best_metric = float('inf')
    best_experiment = None
    metric_name = 'RMSE'
    
    for i, param_update in enumerate(param_combos):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i + 1}/{total}")
        print(f"Parameters: {param_update}")
        print(f"{'=' * 70}")
        
        # Update config with current parameters and route results to tune directory
        config = deep_update(copy.deepcopy(base_config), {
            'model': {'xgboost': param_update}
        })
        config['paths']['results_dir'] = tune_dir
        
        # Reset random seed for fair comparison
        set_random_seed(tune_config.get('random_seed', 42))
        
        # Generate experiment name
        param_suffix = "_".join(format_param_short(k, v) for k, v in sorted(param_update.items()))
        experiment_name = f"tune_{i:03d}_{param_suffix}"
        
        try:
            result = run_training(config, experiment_name)
            result['param_update'] = param_update
            all_results.append(result)
            
            # Track best experiment by validation metrics
            val_metrics = result.get('training_results', {}).get('validation_metrics', {})
            if metric_name in val_metrics:
                metric_value = val_metrics[metric_name]
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_experiment = result
            
            print(f"\nExperiment {i + 1} completed: val_{metric_name}={val_metrics.get(metric_name, 'N/A')}")
            
        except Exception as e:
            print(f"\nExperiment {i + 1} FAILED: {str(e)}")
            all_results.append({
                'experiment_name': experiment_name,
                'param_update': param_update,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save best results only
    if best_experiment:
        best_val_metrics = best_experiment.get('training_results', {}).get('validation_metrics', {})
        comparison = {
            'best_experiment': best_experiment['experiment_name'],
            'best_params': best_experiment.get('param_update', {}),
            'best_validation_metrics': best_val_metrics,
            'optimization_metric': f"val_{metric_name}",
            'best_metric_value': float(best_metric),
            'total_experiments': total,
            'timestamp': get_timestamp()
        }
    else:
        comparison = {
            'best_experiment': None,
            'best_params': None,
            'best_validation_metrics': None,
            'optimization_metric': f"val_{metric_name}",
            'best_metric_value': None,
            'total_experiments': total,
            'timestamp': get_timestamp()
        }
    
    comparison_path = os.path.join(tune_dir, "tune_comparison.yaml")
    save_yaml(comparison, comparison_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {total}")
    print(f"Optimization metric: val_{metric_name} (lower is better)")
    
    if best_experiment:
        best_val_metrics = best_experiment.get('training_results', {}).get('validation_metrics', {})
        print(f"\nBest experiment: {best_experiment['experiment_name']}")
        print(f"Best val_{metric_name}: {best_metric:.4f}")
        print(f"Best params: {best_experiment.get('param_update', {})}")
        print("Full validation metrics:")
        for k, v in best_val_metrics.items():
            print(f"  {k}: {v}")
    
    print(f"\nAll results saved to: {tune_dir}")
    print(f"Best results summary: {comparison_path}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()