# MLETT - ML Time Series Forecasting on ETT Dataset

A machine learning pipeline for time series forecasting using the ETT (Electricity Transformer Temperature) dataset.

## Project Structure

```
MLETT/
├── data/
│   ├── raw/                            # Raw data files
│   │   └── ETTh1.csv
│   └── processed/                      # Processed data files
├── src/mlett/                          # Source code (namespace package)
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.yaml                # Main configuration
│   │   └── tune_config.yaml           # Hyperparameter tuning configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py            # Data loading & cleaning
│   │   └── time_series_split.py        # Chronological split & sliding windows
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py             # FeatureTransformer (anti-leakage)
│   │   └── industrial_features.py     # Compact 42d industrial features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py              # Abstract base model
│   │   ├── xgboost_model.py           # XGBoost model
│   │   └── lightgbm_model.py          # LightGBM model
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                 # Trainer with inverse transform
│   └── utils/
│       ├── __init__.py
│       ├── io.py                       # I/O utilities (YAML, JSON, CSV)
│       ├── logger.py                   # Logging setup
│       ├── metrics.py                  # MSE, RMSE, MAE, R2, MAPE, SMAPE
│       └── seed.py                     # Random seed for reproducibility
├── scripts/
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation script
│   ├── predict.py                      # Prediction script
│   └── tune.py                         # Hyperparameter tuning (Grid/Optuna)
├── results/                            # Experiment outputs (one dir per run)
├── pyproject.toml                      # Package config (PEP 621)
├── .gitignore
├── README.md
└── README_CN.md
```

## Installation

### Prerequisites
- Python 3.8+
- Conda

### Setup

1. Create conda environment:
```bash
conda create -n data_science python=3.9
conda activate data_science
```

2. Install the package (includes all dependencies):
```bash
pip install -e .
```

> Uses `pyproject.toml` (PEP 621). The `-e` flag installs in editable mode so code changes take effect immediately.

## Usage

### Training

```bash
# Default configuration (XGBoost + industrial features)
python scripts/train.py

# Custom experiment name
python scripts/train.py --model-name my_experiment

# Custom config file
python scripts/train.py --config path/to/config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --experiment results/<experiment_name> --data data/raw/ETTh1.csv
```

### Prediction

```bash
python scripts/predict.py --experiment results/<experiment_name> --input data/raw/new_data.csv
```

### Hyperparameter Tuning

```bash
# Grid search
python scripts/tune.py --tune-config src/mlett/config/tune_config.yaml

# Optuna Bayesian optimization (default in tune_config.yaml)
python scripts/tune.py --tune-config src/mlett/config/tune_config.yaml
```

Results are saved to `results/tune_<timestamp>/` with a `tune_comparison.yaml` containing the best validation metrics and parameters.

## Training Pipeline

The training pipeline follows a strict order to prevent data leakage:

```
1. Load & clean raw data
2. Chronological split (train 70% / val 15% / test 15%) — BEFORE feature engineering
3. FeatureTransformer.fit_transform(train)  — fit scaler & target_scaler on train only
   FeatureTransformer.transform(val)        — use fitted scalers
   FeatureTransformer.transform(test)       — use fitted scalers
4. Build sliding windows for each split independently
5. Train model on standardized features & target
6. Inverse transform predictions to original scale for metrics
7. Save all artifacts to results/<experiment_name>/
```

### Feature Modes

| Mode | Dimensions | Description |
|------|-----------|-------------|
| `industrial` | 42 | Rolling stats (24) + recent values (12) + time features (6) |
| `flat` | 288 | Flatten all (24 steps × 12 features) |

**Industrial features** (default) compress each sliding window into a compact 42-dimensional vector:
- **Rolling Statistics (24d)**: mean/std/max/min of 6 sensors over full 24h window
- **Recent Values (12d)**: raw sensor readings at t-1 and t
- **Target-point Time Features (6d)**: hour_sin, hour_cos, month_sin, month_cos, day_of_week, is_weekend

### Target Mode

| Mode | Description |
|------|-------------|
| `delta` | Predict OT change: `OT[t+h] - OT[t+h-1]` (default) |
| `absolute` | Predict OT value directly |

Delta mode reduces target variance and often improves generalization. Metrics are always computed in original scale after reconstruction.

### Sample Definition (Sliding Window)

Each sample is defined as:
- **Input X**: past `window_size` hours of features (default: 24h)
- **Output y**: next `forecast_horizon` hours of OT value (default: 1h)

```
┌─────────────────────┐    ┌───┐
│  t-23 ... t-1, t    │ →  │t+1│
│  24h × features      │    │OT │
└─────────────────────┘    └───┘
```

### Target Standardization

Both features and target (OT) are standardized during training:
- Features: `StandardScaler` fitted on train set only
- Target: separate `target_scaler` fitted on train set only
- Metrics are computed in **original scale** after inverse transform
- The `target_scaler` is saved with `transformer.pkl` for inference

## Models

### XGBoost (default)

Gradient boosting with histogram-based splits. Uses `reg:absoluteerror` objective (MAE loss) for robustness against outliers.

### LightGBM

Alternative gradient boosting framework with leaf-wise tree growth, GOSS sampling, and exclusive feature bundling. Often trains faster than XGBoost.

Switch models via `config.yaml`:
```yaml
model:
  type: "lightgbm"  # or "xgboost"
```

### Model Comparison (ETT dataset, industrial features, delta mode)

| Model | Test RMSE | Test R² | Test SMAPE |
|-------|-----------|---------|------------|
| XGBoost | 0.6187 | 0.9524 | 6.27% |
| LightGBM | 0.6229 | 0.9518 | 6.31% |

## Hyperparameter Tuning

Two tuning methods are supported via `tune_config.yaml`:

### Grid Search
Exhaustive cartesian product of parameter values. Best for small search spaces.

### Optuna (Bayesian)
TPE sampler with MedianPruner for early termination of unpromising trials. Much more efficient for large search spaces.

Search spaces are model-specific — `tune_config.yaml` contains separate blocks for XGBoost (`search_space`) and LightGBM (`lightgbm_search_space`).

## Experiment Directory

Each training run produces a self-contained experiment directory:

```
results/
└── <experiment_name>/          ← timestamp or --model-name value
    ├── train.log               ← Main training log
    ├── trainer.log             ← Trainer module log
    ├── model.pkl               ← Trained model
    ├── model_results.yaml      ← Training/validation metrics
    ├── transformer.pkl          ← FeatureTransformer (with target_scaler)
    ├── config.yaml              ← Configuration snapshot
    └── results.yaml             ← Test metrics & sample definition
```

Tuning runs save all trials under a single timestamped directory:

```
results/
└── tune_<timestamp>/
    ├── tune_comparison.yaml     ← Best validation metrics & parameters
    ├── optuna_trial_000/        ← Each trial's artifacts
    ├── optuna_trial_001/
    └── ...
```

## Configuration

Main config: `src/mlett/config/config.yaml`
Tuning config: `src/mlett/config/tune_config.yaml`

| Section | Key Settings |
|---------|-------------|
| `data` | raw_data_path, target_column (OT), numerical_features |
| `split` | train/val/test ratios (0.7/0.15/0.15) |
| `features` | feature_mode, target_mode, window_size, forecast_horizon |
| `model.type` | "xgboost" or "lightgbm" |
| `model.xgboost` | objective, max_depth, learning_rate, n_estimators, regularization |
| `model.lightgbm` | objective, metric, num_leaves, max_depth, learning_rate, regularization |
| `training` | use_validation, save_model |
| `paths` | results_dir |
| `random_seed` | 42 (for reproducibility) |

## Data

ETT (Electricity Transformer Temperature) dataset:
- **Features**: HUFL, HULL, MUFL, MULL, LUFL, LULL (load types)
- **Target**: OT (Oil Temperature)
- **Granularity**: Hourly
- **Size**: 17,421 samples (2016-07 to 2018-06)

## Features

- Chronological data splitting (no random shuffle, preserves time order)
- Anti-leakage feature engineering (fit on train, transform on val/test)
- Industrial feature extraction (compact 42d representation)
- Delta target mode (predict temperature change, not absolute value)
- Multi-model support (XGBoost, LightGBM)
- Hyperparameter tuning (Grid Search + Optuna with pruning)
- Target standardization with inverse transform for interpretable metrics
- Comprehensive metrics: MSE, RMSE, MAE, R², MAPE, SMAPE
- Reproducibility via random seed (Python, NumPy, XGBoost, LightGBM)
- Self-contained experiment directories

## Package Import

```python
from mlett.data.preprocessing import load_data, clean_data
from mlett.data.time_series_split import time_series_split, create_sliding_windows
from mlett.features.engineering import FeatureTransformer
from mlett.features.industrial_features import create_industrial_windows
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger
from mlett.utils.seed import set_random_seed
```

## License

MIT License
