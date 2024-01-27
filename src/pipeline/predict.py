import os
import sys

import cv2
import face_recognition
from imutils import paths

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

MODEL_PATH = "artifacts/model.pkl"
LABELS_PATH = "artifacts/labels.pkl"


def predict(img_path: str, save_path="artifacts/results", show=False):
    try:
        img_name = os.path.basename(img_path)

        model = load_object(MODEL_PATH)
        labels = load_object(LABELS_PATH)
        image = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img, model="hog")

        if len(face_locations) == 0:
            logging.info(f"No face found for {img_name}")
            return []

        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        predictions = model.predict_proba(face_encodings)
        preds_max = predictions.max(axis=1)
        names = []

        for i, pred in enumerate(preds_max):
            if pred > 0.8:
                names.append(labels[predictions.argmax(axis=1)[i]])
            else:
                names.append("UNKNOWN")

        for name, (top, right, bottom, left) in zip(names, face_locations):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
            top = (top - 10) if (top - 10) > 10 else (top + 10)
            cv2.putText(image, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, img_name), image)
        logging.info(f"Recognized and saved {img_name}.")

        if show:
            cv2.imshow(f"{img_path}", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    img_paths = "artifacts/data/test"

    for img_path in paths.list_images(img_paths):
        predict(img_path, show=True)
