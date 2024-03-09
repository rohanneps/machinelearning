import os
import random
import model as model_arch
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.optimizers import Adam
from config import (
    BATCH_SIZE,
    EPOCHS,
    INIT_LR,
    IMAGE_HEIGTH_WIDTH,
    LOG_DIR,
    MODEL_FILEPATH,
    TEST_IMAGE_PATH,
    TRAIN_IMAGE_PATH,
)
from ImageBatchGenerator import ImageBatchGenerator


if __name__ == "__main__":
    random.seed(42)

    print("[INFO] compiling model...")
    model = model_arch.create_cnn(3, IMAGE_HEIGTH_WIDTH, IMAGE_HEIGTH_WIDTH)

    # sgd = SGD(lr=0.01, clipvalue=0.5)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(
        loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"]
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

    tensorboard = TensorBoard(log_dir=LOG_DIR, write_graph=True)
    model_check_point = ModelCheckpoint(
        MODEL_FILEPATH,
        monitor="val_loss",
        period=3,
        save_best_only=True,
        verbose=0,
    )
    earlystopping = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=20)

    # Model summary
    print(model.summary())

    train_image_batch_generator = ImageBatchGenerator(TRAIN_IMAGE_PATH)
    test_image_batch_generator = ImageBatchGenerator(TEST_IMAGE_PATH)

    model.fit_generator(
        train_image_batch_generator.get_image_batch(BATCH_SIZE),
        callbacks=[reduce_lr_on_plateau, model_check_point, tensorboard, earlystopping],
        validation_data=test_image_batch_generator.get_image_batch(BATCH_SIZE),
        validation_steps=BATCH_SIZE,
        steps_per_epoch=BATCH_SIZE,
        epochs=EPOCHS,
    )
