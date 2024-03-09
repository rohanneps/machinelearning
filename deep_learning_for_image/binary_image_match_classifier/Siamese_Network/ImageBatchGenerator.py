import os
import random
from typing import List, Tuple
import cv2
import imutils
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from config import (
    IMAGE_HEIGTH_WIDTH,
    NUM_MATCH_NOT_MATCH_PAIR_CNT,
    ROTATION_RANGE,
    ROTATION_IMAGES_CNT,
    TRAIN_IMAGE_PATH,
)


class ImageBatchGenerator:
    def __init__(self, image_path: str):
        self._image_dir: str = image_path
        self._image_paths: List[str] = sorted(list(paths.list_images(self._image_dir)))
        self._datagen = ImageDataGenerator(
            height_shift_range=0.3,
            horizontal_flip=True,
            shear_range=0.3,
            width_shift_range=0.3,
            zoom_range=0.3,
        )

    def get_image_batch(self, batch_size: int = 32):
        """
        Return batches of images to be used during Memory Output of Error
        Here batch_size indicated number of images as input
        Output is batch_size * (NUM_MATCH_NOT_MATCH_PAIR_CNT*2)
        """

        while True:
            batch_paths = np.random.choice(a=self._image_paths, size=batch_size)
            batch_input_x = []
            batch_input_xx = []
            batch_output_label = []

            # For each image of this batch
            for image_path in batch_paths:
                try:
                    image = self._load_image(image_path)
                except:
                    continue

                # one copy of same image
                batch_input_x.append(image)
                batch_input_xx.append(image)
                batch_output_label.append(1)

                # some rotation augmentation
                for _ in range(0, ROTATION_IMAGES_CNT):
                    random_rotation_angle = random.randint(
                        -ROTATION_RANGE, ROTATION_RANGE
                    )
                    batch_input_x.append(image)
                    batch_input_xx.append(imutils.rotate(image, random_rotation_angle))
                    batch_output_label.append(1)

                # get some matching images through augmentation
                single_image_list = np.expand_dims(image, axis=0)
                i = 0
                for batch in self._datagen.flow(
                    single_image_list, batch_size=NUM_MATCH_NOT_MATCH_PAIR_CNT
                ):
                    i += 1
                    batch_input_x.append(image)  # original image in 3-dimension
                    batch_input_xx.append(batch[0])  # augmentated image in 3-dimension
                    batch_output_label.append(1)  # indicating match
                    if i >= NUM_MATCH_NOT_MATCH_PAIR_CNT:
                        break

                # for not match image pairs
                current_image_category = image_path.split("/")[-2]
                current_image_path = image_path.split("/")[-1]
                orig_image_batch, not_match_image_batch, not_match_output_labels = (
                    self.get_image_not_match_batch(
                        current_image_category, current_image_path, image
                    )
                )
                batch_input_x += orig_image_batch  # original image
                batch_input_xx += not_match_image_batch  # not match image
                batch_output_label += not_match_output_labels  # indicating not match

            batch_x = np.array(batch_input_x, dtype="float") / 255.0
            batch_xx = np.array(batch_input_xx, dtype="float") / 255.0
            batch_y = np.array(batch_output_label)
            yield [batch_x, batch_xx], batch_y

    def get_image_not_match_batch(
        self,
        current_category: str,
        current_image_path: str,
        current_image_vector: np.ndarray,
    ) -> Tuple[List, List, List]:
        original_image_batch = []
        not_match_image_batch = []

        # For not match image of same category
        image_dir_path = os.path.join(self._image_dir, current_category)
        category_image_list = os.listdir(image_dir_path)
        category_image_list.remove(current_image_path)
        random.shuffle(category_image_list)

        # number of not match images from same category is half of the total number
        for cnt in range(0, NUM_MATCH_NOT_MATCH_PAIR_CNT):
            img = category_image_list[cnt]
            image_path = os.path.join(image_dir_path, img)
            try:
                same_cat_not_match_img_vector = self._load_image(image_path)
                original_image_batch.append(current_image_vector)
                not_match_image_batch.append(same_cat_not_match_img_vector)
            except:
                pass

        # For not match image of diff category
        category_list = os.listdir(self._image_dir)
        category_list.remove(current_category)

        for cnt in range(0, NUM_MATCH_NOT_MATCH_PAIR_CNT):
            random_cat = random.choice(category_list)
            random_cat_images = os.listdir(os.path.join(self._image_dir, random_cat))
            random_cat_images = self._check_size_of_random_cat_images(
                category_list, random_cat_images
            )
            random_cat_image = random.choice(random_cat_images)
            random_cat_image_path = os.path.join(
                self._image_dir, random_cat, random_cat_image
            )
            try:
                diff_cat_not_match_img_vector = self._load_image(random_cat_image_path)
                original_image_batch.append(current_image_vector)
                not_match_image_batch.append(diff_cat_not_match_img_vector)
            except:
                pass

        return original_image_batch, not_match_image_batch, len(original_image_batch)

    def _check_size_of_random_cat_images(
        self, category_list: List[str], random_cat_images: List[str]
    ) -> List[str]:
        if len(random_cat_images) == 0:
            random_cat = random.choice(category_list)
            random_cat_images = os.listdir(os.path.join(self._image_dir, random_cat))
            return self._check_size_of_random_cat_images(random_cat_images, random_cat)
        else:
            return random_cat_images

    def _load_image(self, img_path: str) -> np.ndarray:
        image = cv2.imread(img_path)
        image = cv2.resize(image, (IMAGE_HEIGTH_WIDTH, IMAGE_HEIGTH_WIDTH))
        image = img_to_array(image)

        return image


if __name__ == "__main__":
    a = ImageBatchGenerator(TRAIN_IMAGE_PATH).get_image_batch(2)
    x, xx, y = next(a)

    import pandas as pd

    df = pd.DataFrame(columns=["SourceImg", "CompImg", "Label"])
    df["SourceImg"] = x
    df["CompImg"] = xx
    df["Label"] = y
    df.to_csv("Img.tsv", sep="\t", index=False)
