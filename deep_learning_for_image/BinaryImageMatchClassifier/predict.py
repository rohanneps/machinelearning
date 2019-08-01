import pandas as pd
import os
import numpy as np
import cv2
import keras
from keras.preprocessing.image import  img_to_array, load_img
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

def initialize_weights(shape, name=None):
	"""
		The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
		suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
	"""
	return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


def initialize_bias(shape, name=None):
	"""
		The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
		suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
	"""
	return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

class CustomBiasInitializer:
    def __call__(self, shape):
        return initialize_bias(shape)

class CustomWeightInitializer:
    def __call__(self, shape):
        return initialize_weights(shape)

get_custom_objects().update({'initialize_weights': CustomWeightInitializer,'initialize_bias': CustomBiasInitializer})


model = load_model(os.path.join('models','binary_aug_google_apparel.hdf5'))
print('b')
IMG_HEIGHT_WIDTH = 128


def classifyImages(image_path1, image_path2):
	image1 = cv2.imread(image_path1)
	image1 = cv2.resize(image1, (IMG_HEIGHT_WIDTH, IMG_HEIGHT_WIDTH))
	image2 = cv2.imread(image_path2)
	image2 = cv2.resize(image2, (IMG_HEIGHT_WIDTH, IMG_HEIGHT_WIDTH))
	image1 = image1.astype("float") / 255.0
	image2 = image2.astype("float") / 255.0
	image1 = img_to_array(image1)
	image2 = img_to_array(image2)
	image1 = np.expand_dims(image1, axis=0)
	image2 = np.expand_dims(image2, axis=0)
	print('predicting')
	proba = model.predict([image1,image2])
	print(proba[0][0])
	print(proba)




if __name__ == '__main__':
	input_image_1 = os.path.join('1.jpg')
	input_image_2 = os.path.join('2.jpeg')
	classifyImages(input_image_1,input_image_2)