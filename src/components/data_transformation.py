import logging
import os
import pickle
import sys
from dataclasses import dataclass

import cv2
import face_recognition
import imutils

from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    encodings_path = "artifacts/pickles/encodings.pkl"


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    @staticmethod
    def augment_image(img_path):
        try:
            image = cv2.imread(img_path)

            augmented_imgs = []

            # convert original image from BGR to RGB
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # add RGB to augmented images
            augmented_imgs.append(rgb_img)
            # rotate image by 10_deg and add to augmented images
            augmented_imgs.append(imutils.rotate_bound(rgb_img, 10))
            # rotate image by -10_deg and add to augmented images
            augmented_imgs.append(imutils.rotate_bound(rgb_img, -10))
            # rotate image by 30_deg and add to augmented images
            augmented_imgs.append(imutils.rotate_bound(rgb_img, 30))
            # rotate image by -30_deg and add to augmented images
            augmented_imgs.append(imutils.rotate_bound(rgb_img, -30))
            # flip image horizontally and add to augmented images
            augmented_imgs.append(cv2.flip(rgb_img, 1))
            # increase image brightness and add to augmented images
            augmented_imgs.append(cv2.convertScaleAbs(rgb_img, alpha=1.5, beta=0))
            # decrease image brightness and add to augmented images
            augmented_imgs.append(cv2.convertScaleAbs(rgb_img, alpha=0.5, beta=0))

            return augmented_imgs

        except Exception as e:
            raise CustomException(e, sys)

    def encode(self, img_paths):
        try:
            img_encodings = []
            img_names = []

            img_paths = list(img_paths)

            for i, img_path in enumerate(img_paths):
                augmented_imgs = self.augment_image(img_path)
                img_name = os.path.normpath(img_path).split(os.path.sep)[-2]

                logging.info(f"Processing {img_path} ----- {i + 1}/{len(list(img_paths))}")

                for img in augmented_imgs:
                    face_location = face_recognition.face_locations(img, model="hog")

                    if len(face_location) == 1:
                        face_encoding = face_recognition.face_encodings(img, face_location)[0]
                        img_encodings.append(face_encoding)
                        img_names.append(img_name)

            data = {"encodings": img_encodings, "names": img_names}
            with open(self.transformation_config.encodings_path, "wb") as encodings_file:
                encodings_file.write(pickle.dumps(data))

            logging.info("Data transformation completed.")

        except Exception as e:
            raise CustomException(e, sys)
