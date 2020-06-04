from pathlib import Path

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from ageAndGenderEstimation.wide_resnet import WideResNet
from keras.utils.data_utils import get_file

from preprocess.face_detector import get_face


class AgeGenderDetector:
    '''
    Class to determine Age and Gender of a Face Image
    based on https://github.com/yu4u/age-gender-estimation/
    '''
    def __init__(self):
        pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash="fbe63257a054c1c5466cfd7bf14646d6",
                               cache_dir=str(Path(__file__).resolve().parent))
        # load model and weights
        img_size = 64
        self.model = WideResNet(img_size, depth=16, k=8)()
        self.model.load_weights(weight_file)

    def get_age_gender_from_face(self, face_img):
        # predict ages and genders of the detected faces
        img_size = 64
        faces = np.empty((1, img_size, img_size, 3))
        faces[0, :, :, :] = cv2.resize(face_img, (img_size, img_size))

        results = self.model.predict(faces)
        predicted_gender = results[0]
        age_range = np.arange(0, 101).reshape(101, 1)
        predicted_age = results[1].dot(age_range).flatten()

        return predicted_gender, predicted_age


if __name__ == '__main__':
    face = get_face('/home/mo/experiments/masterthesis/face_generation/test_images/raw/sebi.jpg')
    for key in face.keys():
        face_img = face[key]["face_img"]
    model = AgeGenderDetector()
    a, b = model.get_age_gender_from_face(face_img)
    print(a, b)
