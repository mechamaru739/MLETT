import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def select_features(data: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    从 DataFrame 中选择指定的特征。

    参数:
    data (pd.DataFrame): 包含特征的 DataFrame。
    feature_columns (list): 要选择的特征列名列表。

    返回:
    pd.DataFrame: 包含选定特征的 DataFrame。
    """
    return data[feature_columns]

def encode_features(X: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """
    对 DataFrame 中的分类特征进行编码。

    参数:
    X (pd.DataFrame): 包含特征的 DataFrame。
    categorical_features (list): 分类特征列名列表。

    返回:
    pd.DataFrame: 编码后的分类特征 DataFrame。
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(X[categorical_features])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    X = X.drop(columns=categorical_features).reset_index(drop=True)
    
    return pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

def scale_features(X: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
    """
    对 DataFrame 中的数值特征进行缩放。

    参数:
    X (pd.DataFrame): 包含特征的 DataFrame。
    numerical_features (list): 数值特征列名列表。

    返回:
    pd.DataFrame: 缩放后的数值特征 DataFrame。
    """
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    return X