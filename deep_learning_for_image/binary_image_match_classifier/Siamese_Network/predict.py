import argparse
import cv2
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils import get_custom_objects
import numpy as np
from .config import IMAGE_HEIGTH_WIDTH, MODEL_FILEPATH
from .model import initialize_bias, initialize_weights


class CustomBiasInitializer:
    def __call__(self, shape):
        return initialize_bias(shape)


class CustomWeightInitializer:
    def __call__(self, shape):
        return initialize_weights(shape)


get_custom_objects().update(
    {
        "initialize_weights": CustomWeightInitializer,
        "initialize_bias": CustomBiasInitializer,
    }
)


def classify_images(image_path1: str, image_path2: str) -> None:
    model = load_model(MODEL_FILEPATH)
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)

    image1 = image1.astype("float") / 255.0
    image2 = image2.astype("float") / 255.0

    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)

    print("predicting")
    proba = model.predict([image1, image2])
    print(proba[0][0])
    print(proba)


def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_HEIGTH_WIDTH, IMAGE_HEIGTH_WIDTH))
    image = img_to_array(image)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_1", help="Source Image", type=str)
    parser.add_argument("--image_2", help="Target Image", type=str)
    args = parser.parse_args()

    image_1 = args.image_1
    image_2 = args.image_2
    classify_images(image_1, image_1)
