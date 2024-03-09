import os


BATCH_SIZE = 32
EPOCHS = 100
IMAGE_DIR = os.path.join("images")
IMAGE_HEIGTH_WIDTH = 128
INIT_LR = 1e-3
LOG_DIR = "Logs"
MODEL_FILEPATH = os.path.join("models", "binary_aug_google_apparel.hdf5")
NUM_MATCH_NOT_MATCH_PAIR_CNT = 8
ROTATION_IMAGES_CNT = 5
ROTATION_RANGE = 90
TRAIN_IMAGE_PATH = os.path.join("images", "train")
TEST_IMAGE_PATH = os.path.join("images", "test")
