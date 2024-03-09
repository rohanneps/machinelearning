import argparse
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from binary_trainer import MODEL_FILEPATH, TRAINING_IMAGE_DIMENSIONS


def merge_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    result_width = width1 + width2
    result_height = max(height1, height2)
    result = Image.new("RGB", (result_width, result_height), (255, 255, 255))

    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))

    return result


def test_images(image_path_1: str, image_path_2: str):
    img1 = Image.open(image_path_1)
    img2 = Image.open(image_path_2)
    merged_image = merge_images(img1, img2)

    print("[INFO] loading network...")
    model = load_model(MODEL_FILEPATH)

    image = np.array(merged_image)

    # pre-process the image for classification
    image = cv2.resize(image, TRAINING_IMAGE_DIMENSIONS)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prob = model.predict(image)[0]

    idx = np.argmax(prob)  # Getting position of max Probability
    label = "Matched" if idx == 1 else "UnMatched"  # Getting Label

    print("Predicted Label: {}".format(label))
    print("Confidence Score: {0:.2f}%".format(prob[idx] * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_1", help="Source Image", type=str)
    parser.add_argument("--image_2", help="Target Image", type=str)
    args = parser.parse_args()

    image_1 = args.image_1
    image_2 = args.image_2

    test_images(image_1, image_2)
