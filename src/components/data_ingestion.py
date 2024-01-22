import sys
from dataclasses import dataclass

from imutils import paths

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    data_path = "artifacts/images/"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("Data ingestion initiated.")

        try:
            img_paths = paths.list_images(self.ingestion_config.data_path)

            logging.info("Data ingestion completed.")

            return img_paths

        except Exception as e:
            raise CustomException(e, sys)
