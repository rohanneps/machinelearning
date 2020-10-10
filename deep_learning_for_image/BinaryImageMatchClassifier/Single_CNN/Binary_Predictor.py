from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
from imutils import paths
from PIL import Image

def merge_images(image1,image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    result_width = width1+ width2
    result_height = max(height1, height2)
    result = Image.new("RGB", (result_width, result_height), (255, 255, 255))

    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    
    return (result)



def test_images(image_1, image_2, model_name):
	
	img1 = Image.open(image_1)
	img2 = Image.open(image_2)
	merged_image = merge_images(img1,img2)
	print("[INFO] loading network...")
	model = load_model(model_name)
	
	image =  np.array(merged_image)
	
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	prob = model.predict(image)[0]
	
	idx = np.argmax(prob)		# Getting position of max Probability
	label = 'Matched' if idx==1 else 'UnMatched'  # Getting Label
	
	print("Predicted Label: {}".format(label))
	print("Confidence Score: {0:.2f}%".format(prob[idx]*100))
	
	

if __name__ =='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_1", help="Source Image", type=str)
	parser.add_argument("--image_2", help="Target Image", type=str) 
	args = parser.parse_args()
		
	image_1 = args.image_1
	image_2 = args.image_2

	
	model_name = 'binary_model.best.hdf5'
	test_images(image_1,image_2,model_name)