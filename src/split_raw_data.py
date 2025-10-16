import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def data_split(df: pd.DataFrame, test_size=0.2, random_state=1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Разделение и сохранение данных на тестовую и обучающую выборку в data/processed
    '''

    # поиск пути и создание папки, если её нет
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'raw_split'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем пути к файлам
    train_file_path = processed_dir / "diabetes_train.csv"
    test_file_path = processed_dir / "diabetes_test.csv"
    
    # Проверяем и удаляем существующие файлы
    files_to_remove = [train_file_path, test_file_path]
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
    
    # Разделяем данные
    X = df.drop('Outcome', axis=1)  # Признаки
    y = df['Outcome']               # Целевая переменная

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # для сохранения распределения целевой переменной
    )
    
    # Создаем полные DataFrame для сохранения
    train_df = X_train.copy()
    train_df['Outcome'] = y_train
    
    test_df = X_test.copy()
    test_df['Outcome'] = y_test
    
    # Сохраняем данные
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    return X_train, X_test, y_train, y_test 