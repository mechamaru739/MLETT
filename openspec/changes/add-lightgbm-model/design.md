## Context

MLETT is a time series forecasting project on the ETT dataset. It currently supports only XGBoost as the forecasting model via `XGBoostModel(BaseModel)`. The model pipeline is well-structured: `BaseModel` defines the abstract interface, `Trainer.train()` dispatches by `model_type`, and `config.yaml` holds model-specific parameters.

LightGBM is a gradient boosting framework that uses histogram-based split finding with leaf-wise tree growth (vs XGBoost's level-wise). It often trains faster and uses less memory. Adding it requires minimal changes because:
- The `BaseModel` interface (`fit`, `predict`, `save_model`, `load_model`) is model-agnostic
- `Trainer.train()` already has a `model_type` dispatch pattern
- Config and tuning infrastructure supports model-specific parameter blocks

## Goals / Non-Goals

**Goals:**
- Add `LightGBMModel(BaseModel)` class with identical interface to `XGBoostModel`
- Enable `model.type: "lightgbm"` in config.yaml
- Add LightGBM hyperparameter search spaces to tune_config.yaml (Grid + Optuna)
- Maintain full backward compatibility with existing XGBoost workflow

**Non-Goals:**
- Modifying `BaseModel` interface (it works as-is for both models)
- Adding LightGBM-specific features like categorical feature handling (not needed for ETT)
- Creating ensemble/stacking logic (future work)
- Changing training scripts (train.py already dispatches by `config['model']['type']`)

## Decisions

### Decision 1: Mirror XGBoostModel structure exactly
**Choice**: Create `LightGBMModel` as a near-copy of `XGBoostModel`, replacing `xgb.XGBRegressor` with `lgb.LGBMRegressor`.

**Rationale**: The two frameworks have nearly identical sklearn-style APIs. Mirroring the structure means:
- Same `fit(X, y, eval_set)` signature
- Same `eval_history` / `get_eval_scores()` pattern for Optuna pruning
- Same `save_model` / `load_model` via joblib
- Same `get_feature_importance()` for feature analysis

**Alternatives considered**:
- *Abstract factory pattern*: Over-engineered for two models; the simple `if/elif` in Trainer is clearer
- *Auto-detect from params*: Too implicit; explicit `model.type` is safer

### Decision 2: Default parameters tuned for ETT
**Choice**: LightGBM defaults include `num_leaves=31`, `max_depth=5`, `learning_rate=0.01`, `n_estimators=1000`, matching the anti-overfitting philosophy of the XGBoost config.

**Rationale**: ETT is a medium-sized dataset (~17K rows). Low learning rate + high estimators + regularization prevents overfitting. `num_leaves` (default 31) is the primary complexity knob in LightGBM, separate from `max_depth`.

### Decision 3: tune_config.yaml uses model-specific search space keys
**Choice**: Optuna search space for LightGBM is stored under `lightgbm_search_space` key. `tune.py` selects the correct search space based on `config['model']['type']`.

**Rationale**: XGBoost and LightGBM have different hyperparameters (e.g., LightGBM has `num_leaves` instead of `max_depth` as the primary depth control). Separate search spaces avoid confusion and allow independent tuning.

## Risks / Trade-offs

- **Risk**: LightGBM's `evals_result()` API differs slightly from XGBoost's. **Mitigation**: The `get_eval_scores()` method handles both formats by checking available metric keys.
- **Risk**: LightGBM may produce different feature importance rankings than XGBoost. **Mitigation**: Not a problem — this is expected and useful for comparison.
- **Trade-off**: Adding a second model increases config surface area. **Mitigation**: Each model's params are isolated under their own config key; no shared parameters.

## Migration Plan

No migration needed — purely additive change. Existing XGBoost workflows are unaffected:
1. `model.type: "xgboost"` continues to work unchanged
2. `model.type: "lightgbm"` activates the new model
3. All training/evaluation/prediction scripts work with both models via config

## Open Questions

None — implementation is straightforward given the existing architecture.
