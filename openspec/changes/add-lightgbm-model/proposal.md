## Why

The MLETT project currently only supports XGBoost as the forecasting model. Adding LightGBM as an alternative gradient boosting framework provides a second tree-based baseline with different inductive biases (leaf-wise vs level-wise growth, GOSS sampling, exclusive feature bundling). This enables direct model comparison on the ETT dataset and is a prerequisite for future ensemble or model selection experiments.

## What Changes

- New `LightGBMModel` class inheriting from `BaseModel`, mirroring the `XGBoostModel` interface (`fit`, `predict`, `get_eval_scores`, `get_feature_importance`, `save_model`, `load_model`)
- `config.yaml` gains a `model.lightgbm` parameter block; `model.type` accepts `"lightgbm"`
- `trainer.py` adds an `elif model_type == "lightgbm"` branch in `train()`
- `tune_config.yaml` gains LightGBM-specific search spaces for both Grid and Optuna
- `pyproject.toml` adds `lightgbm>=4.0` dependency

## Capabilities

### New Capabilities
- `lightgbm-model`: LightGBM model class, training pipeline integration, and hyperparameter tuning support

### Modified Capabilities

## Impact

- **Code**: New file `src/mlett/models/lightgbm_model.py`; minor edits to `trainer.py`, `config.yaml`, `tune_config.yaml`, `pyproject.toml`
- **Dependencies**: Adds `lightgbm>=4.0` (pure Python wheel, no GPU required)
- **API**: `config.model.type` gains `"lightgbm"` option; all existing `"xgboost"` behavior unchanged
- **Breaking**: None — fully backward compatible
