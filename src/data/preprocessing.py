import pandas as pd
import numpy as np
from typing import Tuple

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.data.cleaning import get_median, replacing_zero_values_median, get_quantiles, capping_outliers
from src.features.engineering import create_medical_features, one_hot_encoding
from src.data.scaling import get_params_for_standardization, standardization

class DataPreprocessing:
    '''
    Класс для полной обработки данных:
    заполнение пропусков, очистка от выбросов,
    feature engineering, преобразование категориальных признаков,
    стандратизация
    '''
    def __init__(self):
        self.medians = None
        self.standard_data = None

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Полная обработка данных'''

        # заполнение пропусков
        columns_insert_medians = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.medians = get_median(X_train, columns_insert_medians)

        X_train = replacing_zero_values_median(X_train, columns_insert_medians, self.medians)
        X_test = replacing_zero_values_median(X_test, columns_insert_medians, self.medians)

        # capping(обрезание)
        columns_capping = ['Pregnancies', 'SkinThickness', 'Insulin']
        quantiles = [0.99] * 3
        quantiles_values = get_quantiles(X_train, columns_capping, quantiles)
        inequality = ['>'] * 3

        X_train = capping_outliers(X_train, columns_capping, quantiles_values, inequality)
        X_test = capping_outliers(X_test, columns_capping, quantiles_values, inequality)

        # feature engineering
        X_train = one_hot_encoding(create_medical_features(X_train))
        X_test = one_hot_encoding(create_medical_features(X_test))

        # standardization data
        columns_get_mean_std = ['Pregnancies', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age', 'HOMA_IR', 'BMI_AGE_BOND']
        self.standard_data = get_params_for_standardization(X_train, columns_get_mean_std)

        X_train = standardization(X_train, self.standard_data)
        X_test = standardization(X_test, self.standard_data)

        return X_train, X_test