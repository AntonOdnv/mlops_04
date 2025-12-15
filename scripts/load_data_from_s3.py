import pandas as pd
from dotenv import load_dotenv
import os
import boto3


def main():
    # Загружаем переменные окружения
    load_dotenv('keys.env')

    # Ваш API ключ DeepSeek
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
    AWS_PATH = 'https://storage.yandexcloud.net'

    # Поднимаем s3 клиент
    session = boto3.session.Session()
    s3 = session.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_PATH,
        region_name='ru-central1'
    )
    # Проверяем, существует ли каталог для датасетов. Если нет - создаем
    os.makedirs('datasets', exist_ok=True)

    # Загружаем датасеты из s3
    s3.download_file(
                    Bucket=AWS_BUCKET_NAME, 
                    Key='datasets/data_train.csv', # где файл лежит в бакете
                    Filename='datasets/data_train.csv' # куда сохраняем локально
                )

if __name__ == '__main__':
    main()
