from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
import model
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint


# for initial random weigths and bais reproducibility
np.random.seed(7)

dataset = './images'

EPOCHS = 50
INIT_LR = 1e-3
BS = 32
 
# initialize the data and labels
print("[INFO] loading images...")
data_list1 = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# print(len(imagePaths))

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (56, 56))
	image = img_to_array(image)
	data_list1.append(image)	


labels = ['match' for i in range(0,len(data_list1))]
labels += ['not_match' for i in range(0,len(data_list1))]

data_list2 = data_list1.copy()
data_list1 += data_list1
data_list2 += data_list2[::-1]

data_list1 = np.array(data_list1, dtype="float") / 255.0
data_list2 = np.array(data_list2, dtype="float") / 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# 0 is match and 1 becomes not_match

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 80% for testing
print(len(data_list1))
(trainX, testX, trainXX, testXX, trainY, testY) = train_test_split(data_list1,data_list2,labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
cnn1 = model.create_cnn(56, 56, 3)
cnn2 = model.create_cnn(56, 56, 3)

# combining tensor output of the two cnn models
combinedInput = concatenate([cnn1.output, cnn2.output])


x = Dense(16, activation="relu")(combinedInput)

# binary predictor
y = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[cnn1.input, cnn2.input], outputs=y)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,	metrics=["accuracy"])

# train the network
print("[INFO] training network...")

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
tensorboard = TensorBoard(log_dir="logs",write_graph=True)
filepath="binary_classifier_multi_input.best_val_acc.hdf5"
model_check_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)


model.fit([trainX, trainXX], trainY, callbacks = [reduce_lr_on_plateau,model_check_point,tensorboard],validation_data=([testX, testXX], testY),	epochs=EPOCHS, batch_size=BS)

# the model is serialized by the callback function model_check_point