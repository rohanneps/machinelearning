from imutils import paths
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
from random import shuffle, choice
from config import (
					IMG_HEIGTH_WIDTH,
					NUM_MATCH_NOT_MATCH_PAIR_CNT,
					ROTATION_RANGE,
					ROTATION_IMAGES_CNT
					)
import imutils
import random

class ImageBatchGenerator:

	def __init__(self, image_path):
		self.IMAGE_DIR = image_path
		self.IMAGE_PATHS =  sorted(list(paths.list_images(self.IMAGE_DIR)))
		self.datagen = datagen = ImageDataGenerator(width_shift_range=0.3,
													height_shift_range=0.3,
													shear_range=0.3,
													zoom_range=0.3,
													horizontal_flip=True)


	def get_image_not_match_batch(self,
								  current_category,
								  current_image,
								  current_image_vector):
		orig_image_batch = []
		not_match_image_batch = []
		output_labels = []
		# For not match image of same category
		image_dir_path = os.path.join(self.IMAGE_DIR, current_category)
		category_image_list = os.listdir(image_dir_path)
		category_image_list.remove(current_image)
		shuffle(category_image_list)


		# number of not match images from same category is half of the total number	
		for cnt in range(0, NUM_MATCH_NOT_MATCH_PAIR_CNT):
			img = category_image_list[cnt]
			image_path = os.path.join(image_dir_path, img)
			try:
				same_cat_not_match_img_vector = self.load_image(image_path)
				orig_image_batch.append(current_image_vector)			
				not_match_image_batch.append(same_cat_not_match_img_vector)		
				output_labels.append(0)
			except:
				pass

		# For not match image of diff category
		category_list = os.listdir(self.IMAGE_DIR)
		category_list.remove(current_category)

		for cnt in range(0, NUM_MATCH_NOT_MATCH_PAIR_CNT):
			random_cat = choice(category_list)
			random_cat_images = os.listdir(os.path.join(self.IMAGE_DIR,random_cat))
			random_cat_images = self.check_size_of_random_cat_images(random_cat_images,category_list)
			random_cat_image = choice(random_cat_images)
			random_cat_image_path = os.path.join(self.IMAGE_DIR,random_cat,random_cat_image)
			try:
				diff_cat_not_match_img_vector = self.load_image(random_cat_image_path)
				orig_image_batch.append(current_image_vector)			
				not_match_image_batch.append(diff_cat_not_match_img_vector)		
				output_labels.append(0)	
			except:
				pass
		return orig_image_batch, not_match_image_batch, output_labels

	def check_size_of_random_cat_images(self,random_cat_images,category_list):
		if len(random_cat_images)==0:
			random_cat = choice(category_list)
			random_cat_images = os.listdir(os.path.join(self.IMAGE_DIR,random_cat))
			return self.check_size_of_random_cat_images(random_cat_images,random_cat)
		else:
			return random_cat_images

	def load_image(self,imgpath):
		image = cv2.imread(imgpath)
		# cv2.imwrite('test.jpg',image)
		image = cv2.resize(image, (IMG_HEIGTH_WIDTH, IMG_HEIGTH_WIDTH))
		image = img_to_array(image)
		return image

	def get_image_batch(self,batch_size = 32):
		# Return batches of images to be used during Memory Output of Error
		# Here batch_size indicated number of images as input
		# Output is batch_size * (NUM_MATCH_NOT_MATCH_PAIR_CNT*2)
		while True:
			batch_paths = np.random.choice(a=self.IMAGE_PATHS,size = batch_size)
			batch_input_X = []
			batch_input_XX = []
			batch_output = []
			# For each image of this batch
			for image_path in batch_paths:
				try:
					image = self.load_image(image_path)
				except:
					continue

				# One copy of same image
				batch_input_X.append(image)
				batch_input_XX.append(image)
				batch_output.append(1)
				
				# Some Rotational Images
				for rot_img_cnt in range(0, ROTATION_IMAGES_CNT):
					random_rotation_angle = random.randint(-ROTATION_RANGE, ROTATION_RANGE)
					batch_input_X.append(image)
					batch_input_XX.append(imutils.rotate(image, random_rotation_angle))
					batch_output.append(1)



				# get some matching images through augmentation
				single_image_list = np.expand_dims(image, axis=0)
				i = 0
				for batch in self.datagen.flow(single_image_list,batch_size=NUM_MATCH_NOT_MATCH_PAIR_CNT):
					i += 1
					batch_input_X.append(image)			# original image in 3-dimension
					batch_input_XX.append(batch[0])		# augmentated image in 3-dimension
					batch_output.append(1)				# indicating match
					if i >= NUM_MATCH_NOT_MATCH_PAIR_CNT:
						break
				# get not matching
				current_image_category = image_path.split('/')[-2]
				current_image = image_path.split('/')[-1]
				orig_image_batch, not_match_image_batch, output_labels = self.get_image_not_match_batch(current_image_category, current_image, image)
				batch_input_X += orig_image_batch			# original image
				batch_input_XX += not_match_image_batch		# not match image
				batch_output += output_labels				# indicating not match
			batch_x = np.array(batch_input_X, dtype="float") / 255.0
			batch_xx = np.array(batch_input_XX, dtype="float") / 255.0
			batch_y = np.array(batch_output)
			yield [batch_x,batch_xx],batch_y

		
if __name__=='__main__':		
	a = get_image_batch(2)
	x,xx,y=next(a)

	import pandas as pd

	df = pd.DataFrame(columns=['SourceImg','CompImg','Label'])
	df['SourceImg']=x
	df['CompImg']=xx
	df['Label']=y
	df.to_csv('Img.tsv',sep='\t',index=False)