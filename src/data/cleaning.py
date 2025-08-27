import pandas as pd
import numpy as np

def get_median(X: pd.DataFrame, columns: list[str]) -> list[float]:
    '''
    Получение медианы ненулевых значений
    '''
    medians = [X[X[column] != 0][column].mean() for column in columns]
    return medians

def replacing_zero_values_avg(X: pd.DataFrame, columns: list[str], median: list[float]) -> pd.DataFrame:
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
