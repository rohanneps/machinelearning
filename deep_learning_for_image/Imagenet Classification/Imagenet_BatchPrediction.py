import pandas as pd
import os
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import time

FOLDER = 'search'
print(FOLDER)

ROOT_DIR = os.path.join('..','images',FOLDER)
BATCH_SIZE = 16
output_df = pd.DataFrame(columns=['filename','prediction'])
filename_list = []
predction_list = []

IMAGENET_MODEL = VGG19
Network = IMAGENET_MODEL
model = Network(weights="imagenet")
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

def classifyImage(image_path):
	global model
	try:
		# image = load_img(image_path)
		image = load_img(image_path, target_size=(128,128))
		image = image.resize(inputShape)
	except:
		# print(image_path)
		return 'error image',0.00

	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess(image)
	preds = model.predict(image)
	P = imagenet_utils.decode_predictions(preds)
	max_prob = 0.0
	max_label = ''
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		current_prob= prob * 100
		if current_prob>max_prob:
			max_prob = current_prob
			max_label = label
	max_prob = "%.4f" % max_prob
	print(max_label,max_prob)
	return max_label,max_prob

def classifyImageList(image_location_list):
	image_list = []
	for img in image_location_list:
		image = load_img(img)
		image = image.resize(inputShape)
		image = img_to_array(image)
		image = preprocess(image)
		image_list.append(image)

	image_list = np.array(image_list)
	pred_list = model.predict(image_list)
	downscaled_pred_list = imagenet_utils.decode_predictions(pred_list)
	
	for item in downscaled_pred_list:
		max_prob_item_tuple = item[0]
		max_label = max_prob_item_tuple[1]
		max_prob = max_prob_item_tuple[2]
		predction_list.append(max_label)
		# print(max_label,max_prob)

if __name__=='__main__':
	cnt = 0
	image_location_list = []
	start = time.time()
	for roots, dirs, files in os.walk(ROOT_DIR):
		file_count = len(files)
		print('File count {}'.format(file_count))
		for file in files:
			cnt += 1
			
			# print(file)
			filename_list.append(file)
			image_full_path = os.path.join(ROOT_DIR, file)
			
			# predction_list.append(prediction)
			image_location_list.append(image_full_path)
			print(cnt)
			if cnt%BATCH_SIZE ==0 or cnt==file_count:
				#classify
				print('predicting')
				classifyImageList(image_location_list)
				print(time.time()-start)
				image_location_list = []
			
			# Linear Classification
			# prediction, probability = classifyImage(image_full_path)
			# if cnt==BATCH_SIZE:
			# 	print(time.time()-start)
			# 	exit(1)

	output_df['filename'] = filename_list
	output_df['prediction'] = predction_list
	output_df.to_csv('{}.tsv'.format(FOLDER), sep='\t',index=False)


