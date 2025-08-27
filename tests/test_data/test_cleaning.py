import sys
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.data import get_median, replacing_zero_values_median

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