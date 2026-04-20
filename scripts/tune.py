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
    print("=" * 70)
    
    # Run experiments
    all_results = []
    best_metric = float('inf')
    best_experiment = None
    metric_name = 'RMSE'  # Metric to optimize (lower is better)
    
    for i, param_update in enumerate(param_combos):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i + 1}/{total}")
        print(f"Parameters: {param_update}")
        print(f"{'=' * 70}")
        
        # Update config with current parameters
        config = deep_update(copy.deepcopy(base_config), {
            'model': {'xgboost': param_update}
        })
        
        # Reset random seed for fair comparison
        set_random_seed(tune_config.get('random_seed', 42))
        
        # Generate experiment name
        param_suffix = "_".join(format_param_short(k, v) for k, v in sorted(param_update.items()))
        experiment_name = f"tune_{i:03d}_{param_suffix}"
        
        try:
            result = run_training(config, experiment_name)
            result['param_update'] = param_update
            all_results.append(result)
            
            # Track best experiment
            if metric_name in result.get('test_metrics', {}):
                metric_value = result['test_metrics'][metric_name]
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_experiment = result
            
            print(f"\nExperiment {i + 1} completed: {metric_name}={metric_value:.4f}")
            
        except Exception as e:
            print(f"\nExperiment {i + 1} FAILED: {str(e)}")
            all_results.append({
                'experiment_name': experiment_name,
                'param_update': param_update,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save comparison results
    comparison = {
        'tune_method': tune_config['tune_method'],
        'base_config': tune_config['base_config'],
        'param_space': param_space,
        'total_experiments': total,
        'optimization_metric': metric_name,
        'best_experiment': best_experiment['experiment_name'] if best_experiment else None,
        'best_metric_value': float(best_metric) if best_metric != float('inf') else None,
        'experiments': []
    }
    
    for result in all_results:
        entry = {
            'name': result.get('experiment_name', 'unknown'),
            'params': result.get('param_update', {}),
            'status': result.get('status', 'unknown')
        }
        if 'test_metrics' in result:
            entry['test_metrics'] = result['test_metrics']
        if 'error' in result:
            entry['error'] = result['error']
        comparison['experiments'].append(entry)
    
    # Save comparison file
    results_dir = base_config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    comparison_path = os.path.join(results_dir, "tune_comparison.yaml")
    save_yaml(comparison, comparison_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {total}")
    print(f"Optimization metric: {metric_name} (lower is better)")
    
    if best_experiment:
        print(f"\nBest experiment: {best_experiment['experiment_name']}")
        print(f"Best {metric_name}: {best_metric:.4f}")
        print(f"Best params: {best_experiment.get('param_update', {})}")
    
    print(f"\nComparison saved to: {comparison_path}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()