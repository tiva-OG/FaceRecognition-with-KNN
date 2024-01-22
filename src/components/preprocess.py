from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def preprocess_data():
    ingester = DataIngestion()
    img_paths = ingester.initiate_ingestion()

    transformer = DataTransformation()
    transformer.encode(img_paths)


if __name__ == '__main__':
    preprocess_data()
