import os
from typing import List, Tuple
import cv2
import random
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    MaxPooling2D,
    Flatten,
)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# initialize the number of epochs to train for, initial learning rate and batch size
BATCH_SIZE = 16
EPOCHS = 50
LABELS_FILE = "labels.npy"
IMAGE_DIMENSION = (56, 56)
IMAGE_DIR = "images"
INIT_LR = 1e-3
MODEL_FILENAME = "image_model.best.hdf5"
TRAINING_IMAGES_CUTOFF = 50
TRAINING_SUMMARY_FILE = "training.tsv"
TENSORBOARD_LOG_DIR = "tf_logs"


class CnnModel:
    _IMAGE_DIMENSION: Tuple[int, int] = IMAGE_DIMENSION

    def __init__(self, model_file_path: str, batch_size: int=16, epochs: int=10):
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._model_file_path: str = os.path.join(os.getcwd(), model_file_path)

    @classmethod
    def preprocess_image(cls, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.resize(image, cls._IMAGE_DIMENSION)
        image = img_to_array(image)
        return image

    def train(self) -> None:
        image_paths = sorted(
            list(paths.list_images(os.path.join(os.getcwd(), IMAGE_DIR)))
        )

        # initialize the data and labels
        print("[INFO] loading images...")
        data = []
        labels = []

        file_count = len(image_paths)
        print(f"Num images : {file_count}")

        if file_count < TRAINING_IMAGES_CUTOFF:
            print(f"Cannot get good Images Training with less than file_count image, Please Add")
            exit(0)

        random.seed(50)
        random.shuffle(image_paths)

        for image_path in image_paths[:50]:
            data.append(self.preprocess_image(image_path))
            label = image_path.split("/")[-2]
            # label = 1 if label == "Matched" else 0
            labels.append(label)

        data = np.array(data, dtype="float") / 255.0
        labels = self._encode_labels(labels)
        
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

        # construct the image generator for data augmentation
        aug = ImageDataGenerator(
            fill_mode="nearest",
            height_shift_range=0.15,
            horizontal_flip=True,
            rotation_range=45,
            shear_range=0.2,
            width_shift_range=0.15,
            zoom_range=0.2,
        )

        # initialize the model
        print("[INFO] compiling model...")
        model = self._build_model(depth=3, num_classes=len(np.unique(labels)), height=self._IMAGE_DIMENSION[0], width=self._IMAGE_DIMENSION[0])

        opt = Adam(learning_rate=INIT_LR, weight_decay=INIT_LR / self._epochs)
        model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

        # train the network
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
            self._model_file_path,
            monitor="val_loss",
            period=3,
            save_best_only=True,
            verbose=0,
        )

        # Model summary
        print(model.summary())

        history = model.fit_generator(
            aug.flow(trainX, trainY, batch_size=self._batch_size),
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // self._batch_size,
            callbacks=[reduce_lr_on_plateau, model_check_point, tensorboard],
            epochs=self._epochs,
            verbose=1,
        )
        history_frame = pd.DataFrame(history.history)
        history_frame.to_csv(TRAINING_SUMMARY_FILE, sep="\t", index=False)

    def _build_model(self, depth: int, height: int, num_classes: int, width: int) -> Sequential:
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape,))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(num_classes, activation="softmax"))

        # return the constructed network architecture
        return model

    def _encode_labels(self, labels: List[str]) -> np.ndarray:
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        unique_labels = lb.classes_
        np.save(LABELS_FILE, unique_labels)
        labels = to_categorical(labels)
        print(type(labels))

        return labels

if __name__ == "__main__":
    CnnModel(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        model_file_path=MODEL_FILENAME
    ).train()
