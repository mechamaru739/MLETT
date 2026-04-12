import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗函数，填补缺失值和去除重复值


    Parameters:
    data (pd.DataFrame): 需要清洗的DataFrame

    Returns:
    pd.DataFrame: 清洗后的DataFrame
    """
    data = data.drop_duplicates()
    data = data.fillna(data.mean(numeric_only=True))
    return data


def select_features_and_target(data: pd.DataFrame, target_column: str):
    """
    Select features and target variable from the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing features and target.
    target_column (str): The name of the target variable column.

    Returns:
    (pd.DataFrame, pd.Series): Tuple containing features and target variable.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y