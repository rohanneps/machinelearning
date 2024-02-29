import os
from keras.models import load_model
from imutils import paths
import numpy as np
from train_model import CnnModel, LABELS_FILE, MODEL_FILENAME

PRED_IMAGES_DIR = "predict"


def predict():
    root_dir = os.getcwd()

    # Loading trained model
    model = load_model(os.path.join(root_dir, MODEL_FILENAME))
    class_labels = np.load(LABELS_FILE, allow_pickle=True)

    image_paths = sorted(list(paths.list_images(os.path.join(root_dir, PRED_IMAGES_DIR))))
    for image_path in image_paths:
        image = CnnModel.preprocess_image(image_path)
        image = image.astype("float") / 255.0  # Normalization
        image = np.expand_dims(image, axis=0)

        proba = model.predict(image)[0]
        idx = np.argmax(proba)

        label = class_labels[idx]
        print("----------------------------------------------")
        print(image_path.split("/")[-1])
        print("Predicted label: {}".format(label))
        print("confidence score: {}%".format(proba[idx] * 100))


if __name__ == "__main__":
    predict()
