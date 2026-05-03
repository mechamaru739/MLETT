"""Visualization script for MLETT experiments."""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_yaml(filepath: str) -> dict:
    """Load YAML file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_experiment(experiment_dir: str) -> dict:
    """Load all artifacts from an experiment directory."""
    artifacts = {}
    
    results_path = os.path.join(experiment_dir, "results.yaml")
    if os.path.exists(results_path):
        artifacts['results'] = load_yaml(results_path)
    
    model_results_path = os.path.join(experiment_dir, "model_results.yaml")
    if os.path.exists(model_results_path):
        artifacts['model_results'] = load_yaml(model_results_path)
    
    config_path = os.path.join(experiment_dir, "config.yaml")
    if os.path.exists(config_path):
        artifacts['config'] = load_yaml(config_path)
    
    return artifacts


def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]], output_path: str):
    """Plot metrics comparison bar chart for train/val/test."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_names = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'SMAPE']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = []
        labels = []
        for split_name, split_metrics in metrics.items():
            if metric in split_metrics:
                values.append(split_metrics[metric])
                labels.append(split_name)
        
        if values:
            bars = ax.bar(labels, values, color=colors[:len(values)])
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel('Value')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Metrics Comparison (Train / Validation / Test)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_eval_history(eval_history: List[float], output_path: str, best_iteration: Optional[int] = None):
    """Plot validation loss curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(eval_history) + 1)
    ax.plot(iterations, eval_history, color='#3498db', linewidth=1.5, label='Validation Loss')
    
    if best_iteration and best_iteration < len(eval_history):
        ax.axvline(x=best_iteration, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Best Iteration ({best_iteration})')
        ax.scatter([best_iteration], [eval_history[best_iteration - 1]], 
                  color='#e74c3c', s=100, zorder=5)
    
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training History (Eval Loss)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(feature_importances: pd.DataFrame, output_path: str, top_k: int = 20):
    """Plot feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feature_importances.head(top_k).iloc[::-1]
    
    bars = ax.barh(top_features['feature'], top_features['importance'], color='#3498db')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, top_features['importance']):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2.,
               f'{val:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_params_scatter(trial_results: List[Dict], param_name: str, metric_name: str, output_path: str):
    """Plot parameter vs metric scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = [r['params'].get(param_name) for r in trial_results]
    metrics = [r['metrics'].get(metric_name) for r in trial_results]
    
    ax.scatter(params, metrics, color='#3498db', alpha=0.6, s=50)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f'{param_name} vs {metric_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trials_ranking(trial_results: List[Dict], metric_name: str, output_path: str):
    """Plot trials ranked by metric."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_trials = sorted(trial_results, key=lambda x: x['metrics'].get(metric_name, float('inf')))
    
    trial_names = [t['name'] for t in sorted_trials]
    metric_values = [t['metrics'].get(metric_name, 0) for t in sorted_trials]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_trials)))
    
    bars = ax.bar(range(len(sorted_trials)), metric_values, color=colors)
    ax.set_xticks(range(len(sorted_trials)))
    ax.set_xticklabels(trial_names, rotation=45, ha='right')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Trials Ranked by {metric_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_experiments_radar(experiments_metrics: Dict[str, Dict[str, float]], output_path: str):
    """Plot radar chart comparing multiple experiments."""
    metrics = ['RMSE', 'MAE', 'R2', 'SMAPE']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments_metrics)))
    
    for idx, (exp_name, exp_metrics) in enumerate(experiments_metrics.items()):
        values = [exp_metrics.get(m, 0) for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=exp_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Experiments Comparison (Radar)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_experiment(experiment_dir: str, output_dir: str):
    """Generate visualizations for a single experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading experiment: {experiment_dir}")
    artifacts = load_experiment(experiment_dir)
    
    if 'results' not in artifacts:
        print("Error: results.yaml not found")
        return
    
    results = artifacts['results']
    model_results = artifacts.get('model_results', {})
    
    # 1. Metrics comparison
    metrics_data = {}
    if 'training_metrics' in model_results:
        metrics_data['Train'] = model_results['training_metrics']
    if 'validation_metrics' in model_results:
        metrics_data['Validation'] = model_results['validation_metrics']
    if 'test_metrics' in results:
        metrics_data['Test'] = results['test_metrics']
    
    if metrics_data:
        plot_metrics_comparison(metrics_data, os.path.join(output_dir, "metrics_comparison.png"))
    
    # 2. Eval history
    eval_history = model_results.get('eval_history', results.get('training_results', {}).get('eval_history'))
    if eval_history:
        best_iteration = None
        if 'best_iteration' in model_results:
            best_iteration = model_results['best_iteration']
        plot_eval_history(eval_history, os.path.join(output_dir, "eval_history.png"), best_iteration)
    
    # 3. Feature importance
    try:
        import joblib
        model_path = os.path.join(experiment_dir, "model.pkl")
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            feature_importances = model_data.get('feature_importances_')
            feature_columns = model_data.get('feature_columns')
            
            if feature_importances is not None and feature_columns is not None:
                fi_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                plot_feature_importance(fi_df, os.path.join(output_dir, "feature_importance.png"))
    except Exception as e:
        print(f"Warning: Could not plot feature importance: {e}")
    
    # 4. Summary text
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        config = artifacts.get('config', {})
        f.write(f"Model Type: {config.get('model', {}).get('type', 'N/A')}\n")
        f.write(f"Feature Mode: {config.get('features', {}).get('feature_mode', 'N/A')}\n")
        f.write(f"Target Mode: {config.get('features', {}).get('target_mode', 'N/A')}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("TEST METRICS\n")
        f.write("-" * 40 + "\n")
        test_metrics = results.get('test_metrics', {})
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"Saved: {summary_path}")
    print(f"\nAll visualizations saved to: {output_dir}")


def visualize_tune(tune_dir: str, output_dir: str):
    """Generate visualizations for tuning experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading tuning results: {tune_dir}")
    
    comparison_path = os.path.join(tune_dir, "tune_comparison.yaml")
    if not os.path.exists(comparison_path):
        print("Error: tune_comparison.yaml not found")
        return
    
    comparison = load_yaml(comparison_path)
    
    # Load all trial results
    trial_results = []
    for trial_dir in sorted(Path(tune_dir).iterdir()):
        if trial_dir.is_dir() and trial_dir.name.startswith(('optuna_trial_', 'grid_')):
            trial_artifacts = load_experiment(str(trial_dir))
            if 'results' in trial_artifacts:
                results = trial_artifacts['results']
                config = trial_artifacts.get('config', {})
                model_params = config.get('model', {}).get(config.get('model', {}).get('type', 'xgboost'), {})
                
                trial_results.append({
                    'name': trial_dir.name,
                    'params': model_params,
                    'metrics': results.get('test_metrics', results.get('validation_metrics', {}))
                })
    
    if not trial_results:
        print("No trial results found")
        return
    
    # 1. Trials ranking
    metric_name = comparison.get('optimization_metric', 'val_RMSE').replace('val_', '')
    plot_trials_ranking(trial_results, metric_name, os.path.join(output_dir, "trials_ranking.png"))
    
    # 2. Parameter scatter plots
    if trial_results:
        sample_params = list(trial_results[0]['params'].keys())
        for param in sample_params[:4]:  # Plot top 4 parameters
            plot_params_scatter(trial_results, param, metric_name,
                              os.path.join(output_dir, f"param_{param}.png"))
    
    # 3. Summary text
    summary_path = os.path.join(output_dir, "tune_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TUNING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Best Experiment: {comparison.get('best_experiment', 'N/A')}\n")
        f.write(f"Best {metric_name}: {comparison.get('best_metric_value', 'N/A'):.4f}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("BEST PARAMETERS\n")
        f.write("-" * 40 + "\n")
        best_params = comparison.get('best_params', {})
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("BEST VALIDATION METRICS\n")
        f.write("-" * 40 + "\n")
        best_metrics = comparison.get('best_validation_metrics', {})
        for metric, value in best_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"Saved: {summary_path}")
    print(f"\nAll visualizations saved to: {output_dir}")


def visualize_experiments(experiment_dirs: List[str], output_dir: str):
    """Generate comparison visualizations for multiple experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading {len(experiment_dirs)} experiments...")
    
    experiments_metrics = {}
    experiments_history = {}
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        artifacts = load_experiment(exp_dir)
        
        if 'results' in artifacts:
            results = artifacts['results']
            test_metrics = results.get('test_metrics', {})
            if test_metrics:
                experiments_metrics[exp_name] = test_metrics
            
            eval_history = artifacts.get('model_results', {}).get('eval_history')
            if eval_history:
                experiments_history[exp_name] = eval_history
    
    # 1. Radar chart
    if len(experiments_metrics) >= 2:
        plot_experiments_radar(experiments_metrics, os.path.join(output_dir, "experiments_radar.png"))
    
    # 2. Overlaid eval history
    if experiments_history:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(experiments_history)))
        
        for idx, (exp_name, history) in enumerate(experiments_history.items()):
            ax.plot(range(1, len(history) + 1), history, label=exp_name, 
                   color=colors[idx], linewidth=1.5)
        
        ax.set_xlabel('Boosting Round')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Eval History Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_history_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/eval_history_comparison.png")
    
    # 3. Metrics comparison table
    if experiments_metrics:
        metrics_df = pd.DataFrame(experiments_metrics).T
        metrics_df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"))
        print(f"Saved: {output_dir}/metrics_comparison.csv")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize MLETT experiment results')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Path to single experiment directory')
    parser.add_argument('--tune-dir', type=str, default=None,
                        help='Path to tuning directory')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Paths to multiple experiment directories for comparison')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: experiment_dir/plots)')
    
    args = parser.parse_args()
    
    if args.experiment:
        output_dir = args.output or os.path.join(args.experiment, "plots")
        visualize_experiment(args.experiment, output_dir)
    elif args.tune_dir:
        output_dir = args.output or os.path.join(args.tune_dir, "plots")
        visualize_tune(args.tune_dir, output_dir)
    elif args.experiments:
        output_dir = args.output or "plots/comparison"
        visualize_experiments(args.experiments, output_dir)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/visualize.py --experiment results/my_experiment")
        print("  python scripts/visualize.py --tune-dir results/tune_20260424_183215")
        print("  python scripts/visualize.py --experiments results/exp1 results/exp2")


if __name__ == "__main__":
    main()
