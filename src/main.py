import sys

from src.components.data_transformation import DataTransformationConfig
from src.components.preprocess import preprocess_data
from src.exception import CustomException
from src.pipeline.train import ModelTrainer


def run():
    try:
        data_path = DataTransformationConfig().encodings_path
        preprocess_data()
        trainer = ModelTrainer()
        trainer.train(data_path)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run()
