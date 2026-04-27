## 1. Dependency

- [x] 1.1 Add `lightgbm>=4.0` to `pyproject.toml` dependencies array

## 2. LightGBMModel Class

- [x] 2.1 Create `src/mlett/models/lightgbm_model.py` with `LightGBMModel(BaseModel)` class
- [x] 2.2 Implement `__init__` with default parameters (`objective='regression'`, `metric='mae'`, `n_estimators=1000`, `learning_rate=0.01`, `max_depth=5`, `num_leaves=31`, `subsample=0.7`, `colsample_bytree=0.8`, `reg_alpha=0.5`, `reg_lambda=2.0`, `random_state=42`, `n_jobs=-1`, `verbose=-1`)
- [x] 2.3 Implement `fit(X, y, eval_set=None)` using `lgb.LGBMRegressor`, capture `evals_result_` for pruning
- [x] 2.4 Implement `predict(X)` returning flattened numpy array
- [x] 2.5 Implement `native_model` property returning `self._estimator`
- [x] 2.6 Implement `get_eval_scores(metric_name='mae')` extracting per-iteration validation scores
- [x] 2.7 Implement `get_feature_importance()` returning sorted DataFrame
- [x] 2.8 Implement `save_model` / `load_model` via joblib (same pattern as XGBoostModel)

## 3. Config Integration

- [x] 3.1 Add `model.lightgbm` parameter block to `config.yaml` with default parameters
- [x] 3.2 Update `model.type` comment to include `"lightgbm"` option

## 4. Trainer Integration

- [x] 4.1 Add `elif model_type == "lightgbm"` branch in `Trainer.train()` to instantiate `LightGBMModel`

## 5. Tuning Integration

- [x] 5.1 Add LightGBM Grid search space to `tune_config.yaml` under `lightgbm_param_space`
- [x] 5.2 Add LightGBM Optuna search space to `tune_config.yaml` under `lightgbm_search_space`
- [x] 5.3 Update `tune.py` to select search space based on `config['model']['type']`

## 6. Verification

- [x] 6.1 Run training with `model.type: "lightgbm"` and `feature_mode: "industrial"` end-to-end
- [x] 6.2 Verify model saves and loads correctly (predictions match after reload)
- [x] 6.3 Verify XGBoost workflow is unaffected (`model.type: "xgboost"` still works)
