import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object


@dataclass
class ModelTrainerConfig:
    model_path = "artifacts/model.pkl"
    labels_path = "artifacts/labels.pkl"


class ModelTrainer:

    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def train(self, data_path):
        try:
            logging.info("Initiating training")
            model = KNeighborsClassifier(n_neighbors=3)
            label_encoder = LabelEncoder()

            data = load_object(data_path)
            # with open(data_file, 'rb') as data_obj:
            #     data = pickle.loads(data_obj.read())

            X = data['encodings']
            y = label_encoder.fit_transform(data['names'])

            save_object(self.trainer_config.labels_path, label_encoder.classes_)
            # with open(self.trainer_config.labels_path, 'wb') as labels:
            #     labels.write(pickle.dumps(label_encoder.classes_))
            logging.info("Image labels saved to file")

            model.fit(X, y)
            logging.info("Model training complete.")
            logging.info(f"Model accuracy: {model.score(X, y)}")

            save_object(self.trainer_config.model_path, model)
            logging.info("Model saved to disk successfully.")

        except Exception as e:
            raise CustomException(e, sys)
