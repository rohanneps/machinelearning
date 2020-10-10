from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
import cv2

from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

# def create_cnn(width, height, depth):
# 	inputShape = (height, width, depth)
# 	model = Sequential()
# 	model.add(Conv2D(32, (2, 2), padding="same",input_shape=inputShape))
# 	model.add(Activation("relu"))
# 	model.add(Conv2D(32, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Activation("relu"))
# 	model.add(Dropout(0.25))

# 	model.add(Conv2D(64, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(Conv2D(64, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Activation("relu"))
# 	model.add(Dropout(0.3))

# 	model.add(Conv2D(128, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(Conv2D(128, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Activation("relu"))

# 	model.add(Conv2D(128, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(Conv2D(128, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Activation("relu"))
# 	model.add(Dropout(0.2))

# 	model.add(Conv2D(128, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(Conv2D(128, (2, 2), padding="same"))
# 	model.add(Activation("relu"))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Activation("relu"))
# 	model.add(Dropout(0.3))

# 	model.add(Flatten())
# 	model.add(Dense(64, activation='relu'))
# 	model.add(Dense(32, activation='relu'))

	# return model


def create_cnn(width, height, depth):
	inputShape = (height, width, depth)
	inputs = Input(shape=inputShape)
	x = Conv2D(32, (3, 3), padding="same")(inputs)
	x = Activation("relu")(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Activation("relu")(x)
	x = Dropout(0.3)(x)

	x = Conv2D(64, (3, 3), padding="same")(x)
	x = Activation("relu")(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)	
	x = Activation("relu")(x)
	x = Dropout(0.5)(x)

	x = Conv2D(128, (3, 3), padding="same")(x)
	x = Activation("relu")(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)	
	x = Activation("relu")(x)
	x = Dropout(0.2)(x)

	x = Conv2D(128, (3, 3), padding="same")(x)
	x = Activation("relu")(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)	
	x = Activation("relu")(x)
	x = Dropout(0.2)(x)


	x = Conv2D(64, (3, 3), padding="same")(x)
	x = Activation("relu")(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)	
	x = Activation("relu")(x)
	x = Dropout(0.5)(x)

	x = Flatten()(x)
	x = Dense(64)(x)
	x = Activation("relu")(x)
	x = Dense(32)(x)
	x = Activation("relu")(x)

	model = Model(inputs, x)
	return model