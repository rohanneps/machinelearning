import argparse
import os
import time
from typing import List, Tuple
from imutils import paths
from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd


BATCH_SIZE = 16
CONV_BASE = VGG16
IMAGE_DIMENSION = (224, 224)
IMAGE_DIR = "cat"
OUTPUT_FILENAME = "predictions"


class BatchPredictor:
    def __init__(self):
        self._model = CONV_BASE(weights="imagenet")

        self._filename_list: List = []
        self._image_batch_list: List = []
        self._prediction_list: List = []
        self._preprocess = imagenet_utils.preprocess_input
        self._output_df: pd.DataFrame = pd.DataFrame(
            columns=["file_name", "prediction"]
        )

    def output(self) -> None:
        self._output_df["file_name"] = self._filename_list
        self._output_df["prediction"] = self._prediction_list
        self._output_df.to_csv("{}.tsv".format(OUTPUT_FILENAME), sep="\t", index=False)

    def process(self, image_directory: str) -> None:

        cnt = 0
        image_paths = list(paths.list_images(image_directory))

        start = time.time()
        file_count = len(image_paths)
        # print("Images count {}".format(file_count))

        for image_path in image_paths:

            self._filename_list.append(image_path)
            self._image_batch_list.append(image_path)

            if cnt % BATCH_SIZE == 0 or cnt == file_count:
                self._classify_batch_images()
                # print(time.time() - start)
                self._image_batch_list = []

            # Linear Classification
            # prediction, probability = self._classify_single_image(image_full_path)

    def _classify_single_image(self, image_path: str) -> Tuple[str, float]:
        try:
            image = self._preprocess_image(image_path)
        except:
            return "error image", 0.00

        image = np.expand_dims(image, axis=0)
        prediction = self._model.predict(image)

        prediction = imagenet_utils.decode_predictions(prediction)
        max_prob = 0.0
        max_label = ""

        for i, (imagenet_id, label, prob) in enumerate(prediction[0]):
            current_prob = prob * 100
            if current_prob > max_prob:
                max_prob = current_prob
                max_label = label

        # max_prob = "%.4f" % max_prob
        return max_label, max_prob

    def _classify_batch_images(self) -> None:
        batch_images = []

        for image_path in self._image_batch_list:
            image = self._preprocess_image(image_path)
            batch_images.append(image)

        batch_images = np.array(batch_images)
        pred_list = self._model.predict(batch_images)
        downscaled_pred_list = imagenet_utils.decode_predictions(pred_list)

        for item in downscaled_pred_list:
            max_prob_item_tuple = item[0]
            max_label = max_prob_item_tuple[1]
            max_prob = max_prob_item_tuple[2]
            self._prediction_list.append(max_label)

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = load_img(image_path)
        image = image.resize(IMAGE_DIMENSION)
        image = img_to_array(image)
        image = self._preprocess(image)
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imagenet image(s) prediction"
    )
    parser.add_argument("--imgs_dir_path", help="Image Directory path here", type=str)
    args = parser.parse_args()

    image_dir = args.imgs_dir_path

    predictor = BatchPredictor()
    predictor.process(args.imgs_dir_path)
    predictor.output()
