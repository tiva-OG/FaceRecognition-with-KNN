import pickle
import sys

from src.exception import CustomException


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.loads(file.read())

        return obj

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    with open(file_path, 'wb') as file:
        file.write(pickle.dumps(obj))
