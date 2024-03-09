import os
from imutils import paths
from keras.models import load_model
import numpy as np
from train_model import CnnModel, LABELS_FILE, MODEL_FILENAME


PRED_IMAGES_DIR = "predict"


def predict():
    root_dir = os.getcwd()
    image_list = []

    # Loading trained model
    model = load_model(os.path.join(root_dir, MODEL_FILENAME))
    class_labels = np.load(LABELS_FILE, allow_pickle=True)

    image_paths = list(paths.list_images(os.path.join(root_dir, PRED_IMAGES_DIR)))
    for image_path in image_paths:
        image = CnnModel.preprocess_image(image_path)
        image = image.astype("float") / 255.0  # Normalization
        image = np.expand_dims(image, axis=0)
        image_list.append(image)

    image_list = np.array(image_list, dtype="float")

    image_list = (img for img in image_list)  # converting list to generator
    predict = model.predict_generator(image_list, steps=4)

    for cnt, pred in enumerate(predict):
        print("----------------------------------------------")
        print(image_paths[cnt])
        idx = np.argmax(pred)
        label = class_labels[idx]
        print("Predicted label: {}".format(label))
        print("confidence score: {}%".format(pred[idx] * 100))


if __name__ == "__main__":
    predict()
