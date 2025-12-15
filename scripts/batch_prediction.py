import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import csv
from io import StringIO
import joblib

import yaml
import logging
import logging.config

# Читаем конфиг для логера
with open("logger_cfg.yaml", 'r', encoding='utf-8') as f:
    yaml_cfg = yaml.safe_load(f)

# Инициализируем логгер из yaml конфига
logging.config.dictConfig(yaml_cfg)
logger = logging.getLogger('logger')


def select(sql, engine):
    try:
      with engine.begin() as con:
        query = text(sql)
        return pd.read_sql_query(query, con)
    except Exception as e:
        logger.error("Ошибка выполнения запроса к БД: ", e)


def psql_fast_insert(table, conn, keys, data_iter):

    '''
    **Что делает этот код?**
    Этот код реализует быструю загрузку данных из pandas DataFrame в таблицу PostgreSQL
    с помощью механизма COPY (самый быстрый способ вставки больших объёмов данных в PostgreSQL).

    **Зачем это нужно?**
    - Стандартный to_sql в pandas вставляет данные по одной строке (медленно).
    - С помощью COPY можно загрузить данные **в десятки и сотни раз быстрее** — особенно для больших DataFrame.

    **Когда использовать?**
    - Когда нужно быстро загрузить большой DataFrame в PostgreSQL.
    - Когда важна производительность при импорте данных.
    '''

    try:
        # Получение "сырых" соединений и курсора:
        dbapi_conn = conn.connection # создаем низкоуровневый коннектор с БД из SQLAlchemy
        with dbapi_conn.cursor() as cur: # оздаем курсор для выполнения SQL-команд
            s_buf = StringIO() # создаёт буфер в памяти (как файл, но в памяти)
            writer = csv.writer(s_buf) # создаёт CSV-писатель
            writer.writerows(data_iter) # записывает данные из DataFrame в буфер в формате CSV
            s_buf.seek(0) # возвращает указатель в начало буфера (чтобы читать с начала)
    
            # Формирование SQL-запроса:
            columns = ', '.join('"{}"'.format(k) for k in keys) # формирует строку с именами столбцов для SQL
            # определяем имя таблицы с учётом схемы (если существует)
            if table.schema:
                table_name = '{}.{}'.format(table.schema, table.name)
            else:
                table_name = table.name
    
            # Формируем команду COPY ... FROM STDIN WITH CSV — это специальная
            # команда PostgreSQL для быстрой загрузки данных из файла (или буфера)
            sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
                table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf) # выполняет команду COPY, загружая данные из буфера в таблицу
            logger.info("Данные успешно записаны в БД") 
    except Exception as e:
        logger.error("Ошибка записи данных в БД: ", e)  


def load_model():
    try:
        logger.info("Инициализируем модель из файла")
        loaded_model = joblib.load('model/model.joblib')
        logger.info("Инициализация модели успешна")
        return loaded_model
    except Exception as e:
        logger.error("Ошибка инициализации модели из файла: ", e)

def main():
    # Загружаем переменные окружения
    logger.info('Загружаем файл с кредами')
    if load_dotenv('keys.env'):
        # Загружаем переменные среды
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_NAME = os.getenv('DB_NAME')
        DB_USER = os.getenv('DB_USER')
        DB_PASSWORD = os.getenv('DB_PASSWORD')
        logger.info('Креды успешно инициализированы')
    else:
        logger.error("Файл с кредами не найден или не удалось загрузить переменные окружения.")
        raise RuntimeError("Ошибка загрузки файла с кредами: файл не найден или не удалось загрузить переменные окружения.")

    logger.info('Создаем подключение к БД')
    try:
        # Поднимаем engine для работы с БД
        engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        # Поднимаем сессию 
        # Рекомендуется работать через сессии и закрывать их по завершению операций
        # В противном случае сессия может зависнуть и заблокировать дотсуп к БД
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info('Читаем данные churn_predictions из БД')
        # Считываем данные из БД в df
        sql = '''
                select *
                from churn_predictions
            '''
        df = select(sql, engine)
        logger.info('Данные получены')
    except Exception as e:
        logger.error("Ошибка работы с БД: ", e)

    # В полученном датафрейме отделяем целевую переменную от фичей
    X = df.drop('exited', axis=1)
    y = df['exited']

    # Загружаем модель
    catb = load_model()

    try:
        logger.info('Получаем предсказания целевых значений')
        # Предсказываем целевые значения
        predicted_catb = catb.predict(X)
    except Exception as e:
        logger.error("Ошибка: ", e)

    logger.info('Записываем обработанные данные в БД')
    # Восстанавливаем таргет в df и добавляем предсказания
    df['exited'] = y
    df['churn'] = predicted_catb

    # Записываем данные обратно в БД
    df.to_sql('churn_predictions', engine, index=False, if_exists='replace', method=psql_fast_insert)

    # Обязательно рвем сессию по завершению работы 
    session.close()
    engine.dispose()
    logger.info('Скрипт batch_prediction.py выполнился успешно')
    
if __name__ == '__main__':
    main()
