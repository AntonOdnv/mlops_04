import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             roc_auc_score, roc_curve, recall_score, precision_score)
import os
import yaml
import joblib


def main():
    # Читаем данные для обучения 
    df = pd.read_csv('datasets/data_train_prepared.csv')

    # Разделим данные на обучающую и тестовую выборки
    X = df.drop('exited', axis=1)
    y = df['exited']

    # Загружаем параметры
    with open("model/params.yaml") as f:
        yaml_params = yaml.safe_load(f)
    
    # категориальные признаки
    cat_features = yaml_params['cat_features']
    # инициализируем модель
    params = yaml_params["catboost_params"]

    # Создаём Pool для корректной работы с категориальными признаками
    train_pool = Pool(
        data=X,
        label=y,
        cat_features=cat_features
    )

    # Определяем фолды для CV
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    catb = CatBoostClassifier(**params) # инициализируем модель

    # Обучаем модель
    catb.fit(train_pool)

    # Проверяем, существует ли каталог. Если нет - создаем
    os.makedirs('model', exist_ok=True)

    # Сохраняем модель
    joblib.dump(catb, f"model/model.joblib")

    # получаем предсказания модели
    predicted_catb = catb.predict(X)

    # Выводим информацию
    print('-'*50)
    print('Метрика F1:', round(f1_score(y, predicted_catb), 10))
    print('Метрика AUC:',  round(roc_auc_score(y, predicted_catb), 10))
    print('Метрика Recall:',  round(recall_score(y, predicted_catb), 10))
    print('Метрика Precision:',  round(precision_score(y, predicted_catb), 10))
    print('Метрика Accuracy:',  round(accuracy_score(y, predicted_catb), 10))
    print('-'*50)

if __name__ == '__main__':
    main()
