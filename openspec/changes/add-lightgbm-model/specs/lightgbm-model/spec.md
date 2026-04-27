## ADDED Requirements

### Requirement: LightGBMModel class SHALL implement BaseModel interface
The `LightGBMModel` class SHALL inherit from `BaseModel` and implement all abstract methods (`fit`, `predict`, `native_model`). It SHALL use `lightgbm.LGBMRegressor` as the underlying estimator.

#### Scenario: LightGBMModel initialization with default parameters
- **WHEN** `LightGBMModel()` is instantiated with no parameters
- **THEN** the model SHALL use default parameters including `objective='regression'`, `metric='mae'`, `n_estimators=1000`, `learning_rate=0.01`, `max_depth=5`, `num_leaves=31`

#### Scenario: LightGBMModel initialization with custom parameters
- **WHEN** `LightGBMModel({'learning_rate': 0.05, 'n_estimators': 500})` is instantiated
- **THEN** the model SHALL merge custom params with defaults, overriding `learning_rate` to 0.05 and `n_estimators` to 500

### Requirement: LightGBMModel.fit SHALL train with optional validation set
The `fit` method SHALL accept an optional `eval_set` tuple `(X_val, y_val)` for validation-based early stopping and eval history tracking.

#### Scenario: Training with validation set
- **WHEN** `fit(X_train, y_train, eval_set=(X_val, y_val))` is called
- **THEN** the model SHALL train on training data, evaluate on validation data each iteration, store eval history, and set `is_fitted=True`

#### Scenario: Training without validation set
- **WHEN** `fit(X_train, y_train)` is called with no eval_set
- **THEN** the model SHALL train on training data only, `eval_history` SHALL be empty, and `is_fitted=True`

### Requirement: LightGBMModel.predict SHALL return flattened numpy array
The `predict` method SHALL accept a 2D feature array and return a 1D numpy array of predictions.

#### Scenario: Prediction on fitted model
- **WHEN** `predict(X)` is called on a fitted model
- **THEN** it SHALL return a 1D numpy array of shape `(n_samples,)`

#### Scenario: Prediction on unfitted model
- **WHEN** `predict(X)` is called on an unfitted model
- **THEN** it SHALL raise `RuntimeError`

### Requirement: LightGBMModel.save_model and load_model SHALL persist via joblib
The model SHALL save and load using joblib, storing the estimator state, model params, feature columns, and fitted status.

#### Scenario: Save and reload roundtrip
- **WHEN** a fitted model is saved to a file and then loaded into a new instance
- **THEN** the loaded model SHALL produce identical predictions to the original

### Requirement: Trainer SHALL support model_type "lightgbm"
`Trainer.train()` SHALL accept `model_type="lightgbm"` and instantiate `LightGBMModel` with the provided `model_params`.

#### Scenario: Training with model_type="lightgbm"
- **WHEN** `train(X_train, y_train, model_type="lightgbm")` is called
- **THEN** the trainer SHALL create a `LightGBMModel`, train it, compute validation metrics (if eval_set provided), and return results dict

#### Scenario: Training with model_type="xgboost" unchanged
- **WHEN** `train(X_train, y_train, model_type="xgboost")` is called
- **THEN** behavior SHALL be identical to current implementation

### Requirement: Config SHALL support lightgbm parameter block
`config.yaml` SHALL include a `model.lightgbm` section with default LightGBM parameters. `model.type` SHALL accept `"lightgbm"`.

#### Scenario: Config with model.type "lightgbm"
- **WHEN** `config.yaml` has `model.type: "lightgbm"` and `model.lightgbm: {...}`
- **THEN** the training pipeline SHALL use LightGBM with the specified parameters

### Requirement: Tune config SHALL include LightGBM search spaces
`tune_config.yaml` SHALL include Grid and Optuna search spaces for LightGBM hyperparameters.

#### Scenario: Optuna tuning with lightgbm
- **WHEN** `tune_method: "optuna"` and `model.type: "lightgbm"`
- **THEN** the tuning script SHALL use the LightGBM search space to suggest parameters via `trial.suggest_*`

### Requirement: pyproject.toml SHALL include lightgbm dependency
`pyproject.toml` SHALL list `lightgbm>=4.0` in the dependencies array.

#### Scenario: Package installation
- **WHEN** `pip install -e .` is run
- **THEN** lightgbm SHALL be installed alongside existing dependencies
