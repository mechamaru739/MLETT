# MLETT Agent Instructions

## Project Overview

ML time series forecasting on ETT (Electricity Transformer Temperature) dataset using XGBoost or LightGBM. Python 3.8+, namespace package under `src/mlett/`.

## Setup

```bash
conda create -n data_science python=3.9
conda activate data_science
pip install -e .
```

Dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, pyyaml, joblib, optuna

## Key Commands

```bash
# Train (default: XGBoost + industrial features)
python scripts/train.py

# Train with custom experiment name
python scripts/train.py --model-name my_experiment

# Train with custom config
python scripts/train.py --config path/to/config.yaml

# Evaluate
python scripts/evaluate.py --experiment results/<experiment_name> --data data/raw/ETTh1.csv

# Predict
python scripts/predict.py --experiment results/<experiment_name> --input data/raw/new_data.csv

# Hyperparameter tuning (Grid or Optuna)
python scripts/tune.py --tune-config src/mlett/config/tune_config.yaml
```

## Architecture

```
src/mlett/
├── config/         # config.yaml (main), tune_config.yaml (tuning)
├── data/           # preprocessing.py (load/clean), time_series_split.py (split/windows)
├── features/       # engineering.py (FeatureTransformer), industrial_features.py (42d compact)
├── models/         # base_model.py, xgboost_model.py, lightgbm_model.py
├── training/       # trainer.py (with inverse transform)
└── utils/          # io.py, logger.py, metrics.py, seed.py
```

## Config Files

- `src/mlett/config/config.yaml` - Main config: data paths, split ratios, feature mode, model params
- `src/mlett/config/tune_config.yaml` - Tuning: grid/optuna spaces, inherits base_config

## Key Design Decisions

- **Feature modes**: `industrial` (42d compact: rolling stats + recent + time) vs `flat` (288d full flatten)
- **Target modes**: `delta` (predict OT change, default) vs `absolute` (predict OT directly)
- **Anti-leakage pipeline**: chronological split BEFORE feature engineering; fit scalers on train only
- **Sliding windows**: 24h input → 1h forecast (configurable)

## Output Structure

Each run creates `results/<experiment_name>/` with model.pkl, transformer.pkl, config.yaml, metrics YAMLs.
Tuning runs save to `results/tune_<timestamp>/` with tune_comparison.yaml.

## No Tests

No test files exist. If adding tests, use pytest (in dev dependencies).

## READMEs

- `README.md` - English
- `README_CN.md` - Chinese (互链)
