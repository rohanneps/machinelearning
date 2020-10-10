import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(7)											# prevent different random number generation each run instance to compare models.
image_dir = 'images'

def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]
 
	# return our set of features
	return features


imagePaths = paths.list_images(image_dir)
data = []
labels = []
 

for imagePath in imagePaths:
	image = Image.open(imagePath)
	features = extract_color_stats(image)					# extract RBG channel feature
	data.append(features)									
 
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)									# output class list


le = LabelEncoder()
labels = le.fit_transform(labels)							# convert the string output class to numeric vector

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25)
model = RandomForestClassifier(n_estimators=100)
model.fit(trainX, trainY)									# classifier learns here
predictions = model.predict(testX)					
print(model.score(testX, testY))							# get quality of fit
print(classification_report(testY, predictions,target_names=le.classes_))