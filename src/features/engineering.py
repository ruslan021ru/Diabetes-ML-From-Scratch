import pandas as pd

def create_medical_features(X: pd.DataFrame) -> pd.DataFrame:
    '''
    feature engineering
    '''
    X = X.copy()

    # Создание признаков на основе мед. знаний:
    # инсулинорезистентность
    X['HOMA_IR'] = (X['Glucose'] * X['Insulin']) / 22.5
    # связь возраста и ожирения
    X['BMI_AGE_BOND'] = X['Age'] * X['BMI']

    # Биннинг
    X['Glucose_category'] = pd.cut(X['Glucose'],
                                   bins=[0, 110, 125, 250],
                                   labels=['norm', 'prediabetes', 'diabetes'])
    
    X['BMI_category'] = pd.cut(X['BMI'],
                               bins=[0, 18.5, 24.9, 29.9, 100],
                               labels=['underweight', 'norm', 'overweight', 'obese'])
    
    X['Blood_Pressure_category'] = pd.cut(X['BloodPressure'],
                                          bins=[0, 79, 89, 140],
                                          labels=['norm', 'prehypertension', 'hypertension'])
    
    X.drop(['Glucose', 'BMI', 'BloodPressure'], axis=1, inplace=True)
    
    return X

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    '''
    one_hot_encoding
    '''
    return pd.get_dummies(X, columns=['Glucose_category', 'BMI_category', 'Blood_Pressure_category'])