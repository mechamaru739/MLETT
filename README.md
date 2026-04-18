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
│   ├── __init__.py                     # Package entry
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.yaml                # Main configuration file
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py            # Data loading & cleaning
│   │   └── time_series_split.py        # Chronological split & sliding windows
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py             # FeatureTransformer (anti-leakage)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py              # Abstract base model
│   │   └── xgboost_model.py           # XGBoost model
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
│   └── predict.py                      # Prediction script
├── notebook/                           # Jupyter notebooks
├── results/                            # Experiment outputs (one dir per run)
├── pyproject.toml                      # Package config (PEP 621)
├── requirements.txt
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
# Default configuration
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

## Training Pipeline

The training pipeline follows a strict order to prevent data leakage:

```
1. Load & clean raw data
2. Chronological split (train 70% / val 15% / test 15%) — BEFORE feature engineering
3. FeatureTransformer.fit_transform(train)  — fit scaler & target_scaler on train only
   FeatureTransformer.transform(val)        — use fitted scalers
   FeatureTransformer.transform(test)       — use fitted scalers
4. Build sliding windows for each split independently
5. Train XGBoost on standardized features & target
6. Inverse transform predictions to original scale for metrics
7. Save all artifacts to results/<experiment_name>/
```

### Sample Definition (Sliding Window)

Each sample is defined as:
- **Input X**: past `window_size` hours of all feature columns (default: 24h × 6 features = 144 dimensions)
- **Output y**: next `forecast_horizon` hours of OT value (default: 1h)

```
┌─────────────────────┐    ┌───┐
│  t-23 ... t-1, t    │ →  │t+1│
│  24h × 6 features   │    │OT │
└─────────────────────┘    └───┘
```

### Target Standardization

Both features and target (OT) are standardized during training:
- Features: `StandardScaler` fitted on train set only
- Target: separate `target_scaler` fitted on train set only
- Metrics are computed in **original scale** after inverse transform
- The `target_scaler` is saved with `transformer.pkl` for inference

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

## Configuration

Main config file: `src/mlett/config/config.yaml`

| Section | Key Settings |
|---------|-------------|
| `data` | raw_data_path, target_column (OT), numerical_features |
| `split` | train/val/test ratios (0.7/0.15/0.15) |
| `features` | window_size (24), forecast_horizon (1), window_step (1) |
| `model.xgboost` | max_depth, learning_rate, n_estimators, etc. |
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
- Sliding window sample construction for time series forecasting
- Target standardization with inverse transform for interpretable metrics
- Comprehensive metrics: MSE, RMSE, MAE, R², MAPE, SMAPE
- Reproducibility via random seed (Python, NumPy, XGBoost)
- Self-contained experiment directories

## Package Import

```python
from mlett.data.preprocessing import load_data, clean_data
from mlett.data.time_series_split import time_series_split, create_sliding_windows
from mlett.features.engineering import FeatureTransformer
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger
from mlett.utils.seed import set_random_seed
```

## License

MIT License