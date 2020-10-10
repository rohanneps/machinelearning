import pandas as pd
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from imutils import paths

model = load_model('binary_classifier_multi_input.best_val_acc.hdf5')

def classifyImages(image_path1, image_path2):
	image1 = cv2.imread(image_path1)
	image1 = cv2.resize(image1, (56, 56))
	image2 = cv2.imread(image_path2)
	image2 = cv2.resize(image2, (56, 56))
	image1 = image1.astype("float") / 255.0
	image2 = image2.astype("float") / 255.0
	image1 = img_to_array(image1)
	image2 = img_to_array(image2)
	image1 = np.expand_dims(image1, axis=0)
	image2 = np.expand_dims(image2, axis=0)
	proba = model.predict([image1,image2])
	print(proba)
	if proba > 0.5:	
		print('Not match')
	else:
		print('Match')



if __name__ == '__main__':
	input_image_1 = os.path.join('images','00038130257.jpg')
	input_image_2 = os.path.join('images','00648520016.jpg')
	classifyImages(input_image_1,input_image_2)