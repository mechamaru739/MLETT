"""Hyperparameter tuning script for MLETT.

Supports two tuning methods:
  - "grid": Cartesian product of param_space values (exhaustive, predictable)
  - "optuna": Bayesian optimization with TPE sampler and MedianPruner (efficient)

Results are saved to a timestamped directory under results/, with a
tune_comparison.yaml containing the best validation metrics and parameters.
"""

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


# =============================================================================
# Grid Search
# =============================================================================

def run_grid(tune_config: dict, base_config: dict, tune_dir: str, metric_name: str = 'RMSE'):
    """
    Run grid search hyperparameter tuning (exhaustive cartesian product).
    
    Parameters:
        tune_config (dict): Tuning configuration with param_space
        base_config (dict): Base configuration to override
        tune_dir (str): Directory to save results
        metric_name (str): Validation metric to optimize (default: 'RMSE')
    
    Returns:
        list: All experiment results
    """
    param_space = tune_config['param_space']
    param_combos = generate_grid(param_space)
    
    max_exp = tune_config.get('max_experiments', None)
    total = min(len(param_combos), max_exp) if max_exp else len(param_combos)
    param_combos = param_combos[:total]
    
    print("=" * 70)
    print("MLETT Grid Search")
    print("=" * 70)
    print(f"Base config: {tune_config['base_config']}")
    print(f"Parameter space:")
    for k, v in param_space.items():
        print(f"  {k}: {v}")
    print(f"Total experiments: {total}")
    print(f"Results directory: {tune_dir}")
    print("=" * 70)
    
    all_results = []
    best_metric = float('inf')
    best_experiment = None
    
    for i, param_update in enumerate(param_combos):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i + 1}/{total}")
        print(f"Parameters: {param_update}")
        print(f"{'=' * 70}")
        
        config = deep_update(copy.deepcopy(base_config), {
            'model': {'xgboost': param_update}
        })
        config['paths']['results_dir'] = tune_dir
        
        set_random_seed(tune_config.get('random_seed', 42))
        
        param_suffix = "_".join(format_param_short(k, v) for k, v in sorted(param_update.items()))
        experiment_name = f"grid_{i:03d}_{param_suffix}"
        
        try:
            result = run_training(config, experiment_name)
            result['param_update'] = param_update
            all_results.append(result)
            
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
    
    _save_comparison(all_results, best_experiment, best_metric, total, metric_name,
                     tune_dir, tune_method='grid')
    
    return all_results


# =============================================================================
# Optuna Optimization
# =============================================================================

def suggest_params(trial, search_space: dict) -> dict:
    """
    Sample hyperparameters from search_space using Optuna trial.
    
    Parameters:
        trial: Optuna trial object
        search_space (dict): YAML search space with type/low/high/log keys
    
    Returns:
        dict: Sampled parameter dictionary
    """
    params = {}
    for name, spec in search_space.items():
        ptype = spec['type']
        low = spec['low']
        high = spec['high']
        log = spec.get('log', False)
        
        if ptype == 'int':
            params[name] = trial.suggest_int(name, low, high)
        elif ptype == 'float':
            params[name] = trial.suggest_float(name, low, high, log=log)
        else:
            raise ValueError(f"Unknown search space type: {ptype}")
    
    return params


def get_sampler(sampler_name: str):
    """Get Optuna sampler by name."""
    import optuna
    
    samplers = {
        'TPESampler': optuna.samplers.TPESampler,
        'RandomSampler': optuna.samplers.RandomSampler,
        'CMAESampler': optuna.samplers.CmaEsSampler,
    }
    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_name}. Available: {list(samplers.keys())}")
    return samplers[sampler_name]()


def get_pruner(pruner_config):
    """Get Optuna pruner from config (None if disabled)."""
    import optuna
    
    if pruner_config is None:
        return optuna.pruners.NopPruner()
    
    pruner_name = pruner_config if isinstance(pruner_config, str) else pruner_config.get('name', 'MedianPruner')
    
    if pruner_name == 'MedianPruner':
        warmup_steps = None
        interval_steps = None
        if isinstance(pruner_config, dict):
            warmup_steps = pruner_config.get('warmup_steps', 10)
            interval_steps = pruner_config.get('interval_steps', 5)
        return optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=warmup_steps or 10,
            interval_steps=interval_steps or 5
        )
    
    return optuna.pruners.NopPruner()


def run_optuna(tune_config: dict, base_config: dict, tune_dir: str, metric_name: str = 'RMSE'):
    """
    Run Optuna Bayesian hyperparameter optimization with pruning.
    
    Parameters:
        tune_config (dict): Tuning configuration with search_space and optuna settings
        base_config (dict): Base configuration to override
        tune_dir (str): Directory to save results
        metric_name (str): Validation metric to optimize (default: 'RMSE')
    
    Returns:
        list: All completed trial results
    """
    import optuna
    
    search_space = tune_config['search_space']
    optuna_config = tune_config.get('optuna', {})
    n_trials = optuna_config.get('n_trials', 50)
    sampler_name = optuna_config.get('sampler', 'TPESampler')
    
    pruner_config = optuna_config.get('pruner', None)
    if isinstance(pruner_config, str):
        pruner_config_dict = {'name': pruner_config}
        if pruner_config == 'MedianPruner':
            pruner_config_dict['warmup_steps'] = optuna_config.get('pruner_warmup_steps', 10)
            pruner_config_dict['interval_steps'] = optuna_config.get('pruner_interval_steps', 5)
        pruner_config = pruner_config_dict
    
    sampler = get_sampler(sampler_name)
    pruner = get_pruner(pruner_config)
    
    print("=" * 70)
    print("MLETT Optuna Hyperparameter Optimization")
    print("=" * 70)
    print(f"Base config: {tune_config['base_config']}")
    print(f"Search space:")
    for name, spec in search_space.items():
        log_str = " (log)" if spec.get('log') else ""
        print(f"  {name}: {spec['type']} [{spec['low']}, {spec['high']}]{log_str}")
    print(f"Trials: {n_trials}")
    print(f"Sampler: {sampler_name}")
    print(f"Pruner: {optuna_config.get('pruner', 'None')}")
    print(f"Results directory: {tune_dir}")
    print("=" * 70)
    
    all_results = []
    
    def objective(trial):
        params = suggest_params(trial, search_space)
        
        print(f"\n--- Trial {trial.number + 1}/{n_trials} ---")
        print(f"Parameters: {params}")
        
        config = deep_update(copy.deepcopy(base_config), {
            'model': {'xgboost': params}
        })
        config['paths']['results_dir'] = tune_dir
        
        set_random_seed(tune_config.get('random_seed', 42))
        
        experiment_name = f"optuna_trial_{trial.number:03d}"
        
        try:
            result = run_training(config, experiment_name)
            result['param_update'] = params
            all_results.append(result)
            
            val_metrics = result.get('training_results', {}).get('validation_metrics', {})
            if metric_name not in val_metrics:
                print(f"  WARNING: {metric_name} not in validation metrics, returning inf")
                return float('inf')
            
            metric_value = val_metrics[metric_name]
            
            eval_history = result.get('training_results', {}).get('eval_history', [])
            for step, score in enumerate(eval_history):
                trial.report(score, step)
                if trial.should_prune():
                    print(f"  Trial {trial.number} pruned at step {step} (val_{metric_name}={score:.4f})")
                    raise optuna.exceptions.TrialPruned()
            
            print(f"  val_{metric_name}={metric_value:.4f}")
            return metric_value
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} FAILED: {str(e)}")
            all_results.append({
                'experiment_name': experiment_name,
                'param_update': params,
                'status': 'failed',
                'error': str(e)
            })
            return float('inf')
    
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    
    best_experiment = None
    best_metric = float('inf')
    for r in all_results:
        if r.get('status') == 'failed':
            continue
        val_metrics = r.get('training_results', {}).get('validation_metrics', {})
        if metric_name in val_metrics and val_metrics[metric_name] < best_metric:
            best_metric = val_metrics[metric_name]
            best_experiment = r
    
    _save_comparison(all_results, best_experiment, best_metric, n_trials, metric_name,
                     tune_dir, tune_method='optuna', study=study)
    
    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Completed trials: {len([r for r in all_results if r.get('status') != 'failed'])}")
    print(f"Pruned trials: {len(study.trials) - len([r for r in all_results if r.get('status') != 'failed'])}")
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
    print(f"Best results summary: {os.path.join(tune_dir, 'tune_comparison.yaml')}")
    print("=" * 70)
    
    return all_results


# =============================================================================
# Common utilities
# =============================================================================

def _save_comparison(all_results, best_experiment, best_metric, total_trials,
                     metric_name, tune_dir, tune_method='grid', study=None):
    """Save tune_comparison.yaml with best results."""
    if best_experiment:
        best_val_metrics = best_experiment.get('training_results', {}).get('validation_metrics', {})
        comparison = {
            'best_experiment': best_experiment['experiment_name'],
            'best_params': best_experiment.get('param_update', {}),
            'best_validation_metrics': best_val_metrics,
            'optimization_metric': f"val_{metric_name}",
            'best_metric_value': float(best_metric),
            'total_trials': total_trials,
            'tune_method': tune_method,
            'timestamp': get_timestamp()
        }
    else:
        comparison = {
            'best_experiment': None,
            'best_params': None,
            'best_validation_metrics': None,
            'optimization_metric': f"val_{metric_name}",
            'best_metric_value': None,
            'total_trials': total_trials,
            'tune_method': tune_method,
            'timestamp': get_timestamp()
        }
    
    if study is not None:
        import optuna
        comparison['optuna_stats'] = {
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }
        comparison['best_params'] = dict(study.best_params)
        comparison['best_metric_value'] = float(study.best_value)
    
    comparison_path = os.path.join(tune_dir, "tune_comparison.yaml")
    save_yaml(comparison, comparison_path)
    print(f"\nComparison saved to: {comparison_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for MLETT')
    parser.add_argument('--tune-config', type=str, default='src/mlett/config/tune_config.yaml',
                        help='Path to tuning configuration file')
    parser.add_argument('--max-experiments', type=int, default=None,
                        help='Maximum number of experiments (grid search only)')
    
    args = parser.parse_args()
    
    with open(args.tune_config, 'r') as f:
        tune_config = yaml.safe_load(f)
    
    base_config = load_config(tune_config['base_config'])
    
    tune_dir_name = f"tune_{get_timestamp()}"
    tune_dir = os.path.join(base_config['paths']['results_dir'], tune_dir_name)
    os.makedirs(tune_dir, exist_ok=True)
    
    metric_name = 'RMSE'
    tune_method = tune_config.get('tune_method', 'grid')
    
    if tune_method == 'grid':
        if args.max_experiments:
            tune_config['max_experiments'] = args.max_experiments
        results = run_grid(tune_config, base_config, tune_dir, metric_name)
    elif tune_method == 'optuna':
        results = run_optuna(tune_config, base_config, tune_dir, metric_name)
    else:
        raise ValueError(f"Unknown tune_method: {tune_method}. Use 'grid' or 'optuna'.")
    
    return results


if __name__ == "__main__":
    results = main()