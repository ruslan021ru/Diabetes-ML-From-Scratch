import pandas as pd
import numpy as np

def get_median(X: pd.DataFrame, columns: list[str]) -> list[float]:
    '''
    Получение медианы ненулевых значений
    '''
    medians = [X[X[column] != 0][column].mean() for column in columns]
    return medians

def replacing_zero_values_median(X: pd.DataFrame, columns: list[str], median: list[float]) -> pd.DataFrame:
    '''
    В некоторых признаках имеются значения равные 0,
    хотя исходя из логики данных такого быть не может.
    Произведем замену 0-вых значений на
    среднее значение ненулевых значений признака
    '''
    X = X.copy()
    for ind, column in enumerate(columns):
        X[column] = np.where(X[column] == 0, median[ind], X[column])

    return X

def get_quantiles(X: pd.DataFrame, columns: list[str], quantiles_values: list[float]) -> list[float]:
    '''
    Получение quantiles по значению [0-1]
    '''
    quantiles = [X[column].quantile(quantiles_values[ind]) for ind, column in enumerate(columns)]
    return quantiles

def capping_outliers(X: pd.DataFrame, columns: list[str], quantiles: list[float], inequality: list[str]) -> pd.DataFrame:
    '''
    Обработает некоторые выбросы,
    не будем их убирать полностью, так как
    сама выборка маленькая и учитывая специфику задачи
    это могут быть реальные данные, где у пациента запредельные значения.
    Будем производить capping сверху по 0.99, снизу по 0.01.
    '''
    X = X.copy()
    for ind, column in enumerate(columns):
        if ord(inequality[ind]) == 62: X[column] = np.where(X[column] > quantiles[ind], quantiles[ind], X[column])
        elif ord(inequality[ind]) == 60: X[column] = np.where(X[column] < quantiles[ind], quantiles[ind], X[column])
        else: raise ValueError('Допустимые знаки неравенства > и <')

    return X

