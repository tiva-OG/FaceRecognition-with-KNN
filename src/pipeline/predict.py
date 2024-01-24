import os
import sys

import cv2
import face_recognition
from imutils import paths

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


def predict(img_path: str, model_path: str, labels_path: str, save_path="artifacts/results", show=False):
    """
    :type img_path: str: path to the image
    :param model_path: str: path to the saved model
    :param labels_path: str: path to the saved labels
    :param save_path: str: path to save the resulting image
    :type show: bool: display image after recognizing faces
    """

    try:
        img_name = os.path.basename(img_path)

        model = load_object(model_path)
        labels = load_object(labels_path)
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
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            top = (top - 15) if (top - 15) > 15 else (top + 15)
            cv2.putText(image, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

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
    model_path = "artifacts/pickles/model.pkl"
    labels_path = "artifacts/pickles/labels.pkl"
    img_paths = "artifacts/images/test"

    for img_path in paths.list_images(img_paths):
        predict(img_path, model_path, labels_path, show=True)
