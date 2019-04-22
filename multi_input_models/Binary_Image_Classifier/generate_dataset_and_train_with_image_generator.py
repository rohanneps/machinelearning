from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
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
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


subcategory = './image_dataset'
dataset = os.path.join('..','images',subcategory)
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
img_heigth_width = 56 
# initialize the data and labels
print("[INFO] loading images...")
data_list1 = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

imagePaths = imagePaths[:6000]
# print(len(imagePaths))

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (img_heigth_width, img_heigth_width))
	image = img_to_array(image)
	data_list1.append(image)	


# creating match list
labels = ['match' for i in range(0,len(data_list1))]
data_list2 = data_list1.copy()

# creating not_match list
labels += ['not_match' for i in range(0,len(data_list1))]
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
(trainX, testX, trainXX, testXX, trainY, testY) = train_test_split(data_list1,data_list2,labels, test_size=0.3, random_state=42)

image_generator = ImageDataGenerator(horizontal_flip = False,
                         vertical_flip = False,
                         width_shift_range = 0.2,
                         height_shift_range = 0.2,
                         zoom_range = 0.2,
                         rotation_range = 180,fill_mode='nearest')

def gen_flow_for_two_inputs(X1, X2, y):
	genX1 = image_generator.flow(X1,y,  batch_size=BS,seed=666)
	genX2 = image_generator.flow(X2,y, batch_size=BS,seed=666)
	while True:
		X1i = genX1.next()
		X2i = genX2.next()
		#Assert arrays are equal - this was for peace of mind, but slows down training
		#np.testing.assert_array_equal(X1i[0],X2i[0])
		yield [X1i[0], X2i[0]], X1i[1]

gen_flow = gen_flow_for_two_inputs(trainX, trainXX, trainY)


# initialize the model
print("[INFO] compiling model...")
cnn1 = model.create_cnn(img_heigth_width, img_heigth_width, 3)
cnn2 = model.create_cnn(img_heigth_width, img_heigth_width, 3)

# combining tensor output of the two cnn models
combinedInput = concatenate([cnn1.output, cnn2.output])


x = Dense(16, activation="relu")(combinedInput)

# binary predictor
y = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[cnn1.input, cnn2.input], outputs=y)
sgd = SGD(lr=0.01, clipvalue=0.5)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,	metrics=["accuracy"])

# train the network
print("[INFO] training network...")

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
tensorboard = TensorBoard(log_dir="logs",write_graph=True)
filepath="{}_binary_aug.hdf5".format(subcategory)
model_check_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
earlystopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=15)

# model.fit([trainX, trainXX], trainY, callbacks = [reduce_lr_on_plateau,model_check_point,tensorboard, earlystopping],validation_data=([testX, testXX], testY),	epochs=EPOCHS, batch_size=BS)
model.fit_generator(gen_flow,callbacks = [reduce_lr_on_plateau,model_check_point,tensorboard, earlystopping], validation_data=([testX, testXX], testY),
                    steps_per_epoch=len(trainX) / BS, epochs=EPOCHS)