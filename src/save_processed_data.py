import pandas as pd
from pathlib import Path

def save_proc_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    '''
    Сохранение данных после препроцессинга
    '''

    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_file_path = processed_dir / "diabetes_train_proc.csv"
    test_file_path = processed_dir / "diabetes_test_proc.csv"

    files_to_remove = [train_file_path, test_file_path]
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()

    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False) 









