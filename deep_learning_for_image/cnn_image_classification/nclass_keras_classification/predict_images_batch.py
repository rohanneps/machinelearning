import os
import cv2
from imutils import paths
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from train_model import IMAGE_DIMENSION, IMAGE_DIR, MODEL_FILENAME


PRED_IMAGES_DIR = "predict"


def predict():
    root_dir = os.getcwd()
    labels = []
    image_list = []
    image_paths = sorted(list(paths.list_images(os.path.join(root_dir, IMAGE_DIR))))

    # Labels
    for image_path in image_paths:
        label = image_path.split("/")[-2]
        labels.append(label)

    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # Images to predict
    image_paths = sorted(
        list(paths.list_images(os.path.join(root_dir, PRED_IMAGES_DIR)))
    )
    # Loading trained model
    model = load_model(os.path.join(root_dir, MODEL_FILENAME))

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(
            image, (IMAGE_DIMENSION[0], IMAGE_DIMENSION[0])
        )  # Resizing images to trainned images shape
        image = image.astype("float") / 255.0  # Normalization
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image_list.append(image)
    image_list = np.array(image_list, dtype="float")

    image_list = (y for y in image_list)  # converting list to generator
    predict = model.predict_generator(image_list, steps=4)
    for cnt, pred in enumerate(predict):
        print("----------------------------------------------")
        print(image_paths[cnt])
        idx = np.argmax(pred)
        label = lb.classes_[idx]
        print("predicted label: {}".format(label))
        print("confidence score: {}%".format(pred[idx] * 100))


if __name__ == "__main__":
    predict()
