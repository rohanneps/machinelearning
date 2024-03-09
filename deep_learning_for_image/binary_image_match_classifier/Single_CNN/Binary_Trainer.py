import argparse
import random
from typing import Tuple
import cv2
from imutils import paths
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BATCH_SIZE = 32
EPOCHS = 100
INIT_LR = 1e-3
MODEL_FILEPATH = "binary_model_best.hdf5"
TENSORBOARD_LOG_DIR = "logs"
TRAINING_IMAGE_DIMENSIONS = (56, 56)
TRAINING_SUMMARY_FILE = "training.tsv"


def build_model(classes: int, depth: int, height: int, width: int) -> Sequential:
    # initialize the model
    model = Sequential()
    input_shape = (height, width, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)

    # first set of CONV => RELU => POOL layers
    model.add(
        Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu")
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(500, activation="relu"))

    # softmax classifier
    model.add(Dense(classes, activation="softmax"))
    return model


def preprocess_image(
    image_path: str, image_dimension: Tuple[int, int] = (56, 56)
) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_dimension)
    image = img_to_array(image)
    return image


def train_images(train_directory: str) -> None:

    image_paths = list(paths.list_images(train_directory))
    file_count = len(image_paths)

    if file_count < 100:
        print(
            f"Cannot get good Images Training with less than {file_count} Images. Please Add"
        )
        exit(0)

    random.seed(2)
    random.shuffle(image_paths)

    data = []
    labels = []

    for image_path in image_paths:
        label = image_path.split("/")[-2]
        image = preprocess_image(image_path, TRAINING_IMAGE_DIMENSIONS)
        data.append(image)
        label = 1 if label == "Matched" else 0
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    train_y = to_categorical(train_y, num_classes=2)
    test_y = to_categorical(test_y, num_classes=2)

    aug = ImageDataGenerator(
        fill_mode="nearest",
        height_shift_range=0.15,
        horizontal_flip=True,
        rotation_range=30,
        shear_range=0.2,
        width_shift_range=0.15,
        zoom_range=0.2,
    )

    print("[INFO] compiling model...")
    model = build_model(classes=2, depth=3, height=TRAINING_IMAGE_DIMENSIONS[0], width=TRAINING_IMAGE_DIMENSIONS[1])

    opt = Adam(lr=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    model.compile(
        loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"]
    )

    print("[INFO] training network...")
    reduce_lr_on_plateau = ReduceLROnPlateau(
        cooldown=0,
        factor=0.2,
        min_delta=0.0001,
        min_lr=0.00001,
        mode="auto",
        monitor="val_loss",
        patience=5,
        verbose=0,
    )
    tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_DIR, write_graph=True)
    model_check_point = ModelCheckpoint(
        MODEL_FILEPATH,
        monitor="val_loss",
        period=3,
        save_best_only=True,
        verbose=0,
    )

    # Model summary
    print(model.summary())

    history = model.fit_generator(
        aug.flow(train_x, train_y, batch_size=BATCH_SIZE),
        validation_data=(test_x, test_y),
        steps_per_epoch=len(train_x) // BATCH_SIZE,
        callbacks=[reduce_lr_on_plateau, model_check_point, tensorboard],
        epochs=EPOCHS,
        verbose=1,
    )

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(TRAINING_SUMMARY_FILE, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_img_folder", help="Image directory for training", type=str
    )
    args = parser.parse_args()

    train_directory = args.input_img_folder
    train_images(train_directory)
