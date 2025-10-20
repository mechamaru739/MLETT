import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train, params=None):
    """
    Train an XGBoost model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    params (dict): Model parameters for XGBoost (optional).

    Returns:
    xgb.Booster: Trained XGBoost model.
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',  # 目标为回归
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'seed': 42
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model

def predict(model, X_test):
    """
    Make predictions using the trained XGBoost model.

    Parameters:
    model (xgb.Booster): Trained XGBoost model.
    X_test (pd.DataFrame): Test features.

    Returns:
    np.ndarray: Predicted values.
    """
    return model.predict(X_test)

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the model.

    Parameters:
    y_true (pd.Series): True target values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    dict: Evaluation metrics including MSE and R².
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {'MSE': mse, 'R²': r2}