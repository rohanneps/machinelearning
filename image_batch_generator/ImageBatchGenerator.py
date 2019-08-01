from imutils import paths
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

imagedir = 'downloads'
imagePaths = sorted(list(paths.list_images(imagedir)))


img_heigth_width=128
def load_image(imgpath):
	image = cv2.imread(imgpath)
	image = cv2.resize(image, (img_heigth_width, img_heigth_width))
	image = img_to_array(image)
	return image

def get_image_batch(batch_size = 32):
	# Return batches of images to be used during Memory Output of Error
	while True:
		batch_paths = np.random.choice(a=imagePaths,size = batch_size)
		batch_input = []
		batch_output = []
		for imagePath in batch_paths:
			print(imagePath)
			img = load_image(imagePath)
			batch_input.append(img)
			batch_output.append(1)
		batch_x = np.array(batch_input, dtype="float") / 255.0
		batch_y = np.array(batch_output)
		yield (batch_x,batch_y)		

def get_and_save_image_batch(batch_size = 32):
	# Return batches of images to be used during Memory Output of Error
	while True:
		batch_paths = np.random.choice(a=imagePaths,size = batch_size)
		batch_input = []
		batch_output = []
		for imagePath in batch_paths:
			print(imagePath)
			current_category = imagePath.split('/')[-2]
			current_image = imagePath.split('/')[-1]
			image = load_image(imagePath)
			batch_input.append(image)
			batch_output.append(1)
			image = np.expand_dims(image, axis=0)
			i = 0
			for batch in datagen.flow(image,batch_size=number_of_match_none_match_pair):
				print(i)
				cv2.imwrite(os.path.join('test','{}.jpg'.format(i)), batch[0])
				i += 1
				if i > number_of_match_none_match_pair:
					break
		batch_x = np.array(batch_input, dtype="float") / 255.0
		batch_y = np.array(batch_output)
		yield (batch_x,batch_y)
