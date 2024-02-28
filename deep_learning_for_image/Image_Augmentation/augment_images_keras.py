import argparse
import os
import random
import cv2
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np

OUTPUT_DIR = "./augmented_images"
IMAGE_DIMENSION = (512, 512)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment Images for deep learning Model training. Images are generated in the same folder as input."
    )
    parser.add_argument("--imgs_dir_path", help="Image Directory path here", type=str)
    parser.add_argument(
        "--num_images", help="Number of images to augment for each category", type=int
    )
    args = parser.parse_args()

    image_dir = args.imgs_dir_path
    num_aug_images = args.num_images

    # grab the image paths and randomly shuffle them
    image_paths = list(paths.list_images(image_dir))
    random.seed(42)
    random.shuffle(image_paths)

    datagen = ImageDataGenerator(
        height_shift_range=0.2,
        horizontal_flip=True,
        rotation_range=40,
        shear_range=0.2,
        vertical_flip=True,
        width_shift_range=0.2,
        zoom_range=0.2,
    )

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, IMAGE_DIMENSION)
        image = img_to_array(image)
        image = image.reshape((1, *image.shape))

        datagen.fit(image)
        label = image_path.split("/")[-2]
        dir_path = os.path.join(OUTPUT_DIR, label)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        batch_cntr = 0

        for batch in datagen.flow(
            image,
            batch_size=32,
            save_format="jpg",
            save_prefix="img",
            save_to_dir=dir_path,
        ):
            batch_cntr += 1

            if batch_cntr > num_aug_images:
                break
