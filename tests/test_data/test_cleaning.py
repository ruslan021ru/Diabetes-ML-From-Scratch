import sys
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.data import get_median, replacing_zero_values_median, get_quantiles, capping_outliers

def test_get_median():
    '''
    проверка корректности значения медианы
    '''
    test_data = pd.DataFrame({
            'Glucose': [125, 0, 118, 0],
            'Insulin': [0, 150, 0, 200]  
        })

    medians = get_median(test_data, ['Glucose', 'Insulin'])
    assert medians == [121.5, 175.0], 'Значение медианы посчиталось неверно'

def test_replacing_zero_values_median():
    '''
    проверка заполнения нулей значениями медианы
    '''
    test_data = pd.DataFrame({
            'Glucose': [125, 0, 118, 0],
            'Insulin': [0, 150, 0, 200]  
        })
    
    medians = get_median(test_data, ['Glucose', 'Insulin'])
    df = replacing_zero_values_median(test_data, ['Glucose', 'Insulin'], medians)

    df_correct = pd.DataFrame({
            'Glucose': [125, 121.5, 118, 121.5],
            'Insulin': [175, 150, 175.0, 200]  
        })
    
    assert df.equals(df_correct), 'Значения медианы заполнены неверно'

def test_get_quantiles():
    '''
    проверка получения quantiles по числу [0,1]
    '''
    test_data = pd.DataFrame({
            'Glucose': [125, 95, 118, 30, 26, 18, 70],
            'Insulin': [100, 200, 300, 400, 500, 600, 700]
        })
    columns = ['Glucose', 'Insulin']
    quantiles_values = [0.1, 0.9]

    assert get_quantiles(test_data, columns, quantiles_values) == [22.8, 640], 'Неверно посчитан перцентиль'

def test_capping_outliers():
    '''
    проверка обработки выбросов через capping
    '''
    test_data = pd.DataFrame({
            'Glucose': [125, 95, 118, 30, 26, 18, 70],
            'Insulin': [100, 200, 300, 400, 500, 600, 700] 
        })
    columns = ['Glucose', 'Insulin']
    quantiles_values = [0.1, 0.9]
    quantiles = get_quantiles(test_data, columns, quantiles_values)
    inequiluty = ['<', '>']

    df = capping_outliers(test_data, columns, quantiles, inequiluty)

    df_correct = pd.DataFrame({
            'Glucose': [125.0, 95.0, 118.0, 30.0, 26.0, 22.8, 70.0],
            'Insulin': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 640.0]
        })
    
    assert df.equals(df_correct), 'Неверно заменены выбросы на значение перцентиля'
