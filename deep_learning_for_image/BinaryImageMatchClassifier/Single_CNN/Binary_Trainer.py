from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from IMG_lenet import LeNet
from imutils import paths
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint
import numpy as np
import argparse
import random
import cv2
import os
import sys


def train_images(train_directory):
	EPOCHS = 50
	INIT_LR = 1e-3
	BS = 32
	
	# imagePaths = sorted(list(paths.list_images(download_base_directory+'/'+img_folder+'/'+MainCat)))
	imagePaths = sorted(list(paths.list_images(train_directory)))
		
	
	file_count = len(imagePaths)
	print(file_count)
	if file_count<100:
		print('Cannot get good Images Training with less than '+str(file_count)+' Images, Please Add')
		exit(0)

	random.seed(2)
	random.shuffle(imagePaths)

	
	
	data = []
	labels = []
	for imagePath in imagePaths:
		label = imagePath.split('/')[-2]
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (28, 28))
		image = img_to_array(image)
		data.append(image)						
		label = 1 if label == "Matched" else 0
		labels.append(label)

		
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	
	(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
	
	trainY = to_categorical(trainY, num_classes=2)
	testY = to_categorical(testY, num_classes=2)
	
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
	
	print("[INFO] compiling model...")
	model = LeNet.build(width=28, height=28, depth=3, classes=2)
	
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
	
	print("[INFO] training network...")
	reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
	tensorboard = TensorBoard(log_dir="logs",write_graph=True)
	filepath="binary_model.best.hdf5"
	model_check_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)

	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), callbacks = [reduce_lr_on_plateau,model_check_point,tensorboard],
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, 
		epochs=EPOCHS, verbose=1)
	
	# print("[INFO] serializing network...")
	# model.save('binary_model.h5')
	# print(model.summary())

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_img_folder", help="Image Directory for training", type=str)
	args = parser.parse_args()

	train_directory = args.input_img_folder
	train_images(train_directory)

