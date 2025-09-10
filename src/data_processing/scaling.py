import pandas as pd

def get_params_for_standardization(X: pd.DataFrame, columns: list[str]) -> dict:
    '''
    расчет mean и std для стандартизации признаков
    '''
    stand_info = dict()
    for column in columns:
        stand_info[column] = (X[column].mean(), X[column].std())

    return stand_info

def standardization(X: pd.DataFrame, stand_info: dict) -> pd.DataFrame:
    '''
    стандартизация признаков
    '''
    X = X.copy()
    for column, params in stand_info.items():
        mean, std = params
        X[column] = (X[column] - mean) / std

    return X