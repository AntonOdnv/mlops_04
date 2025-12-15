import pandas as pd
import numpy as np
import os


def main():
    # Загружаем данные 
    df = pd.read_csv('datasets/data_train.csv')
    # Заполняем пропуски tenure рандомными значениями из имеющихся в датасете `tenure`
    nan_idx = df[df['tenure'].isna()].index
    df.loc[nan_idx, 'tenure'] = np.random.choice(df.loc[~df['tenure'].isna(), 'tenure'].values, size=len(nan_idx))

    # Проверяем, существует ли каталог для датасетов. Если нет - создаем
    os.makedirs('datasets', exist_ok=True)

    # Записываем подготовленный датасет
    df.to_csv('datasets/data_train_prepared.csv', index=False)

if __name__ == '__main__':
    main()
