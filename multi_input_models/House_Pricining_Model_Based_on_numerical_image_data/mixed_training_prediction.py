import datasets
import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import os
 
image_input_dir = 'Houses Dataset'
file_path = os.path.join('.', "HousesInfo.txt")
df = datasets.load_house_attributes(file_path)
images = datasets.load_house_images(df, image_input_dir)
images = images / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainAttrX, testAttrX, trainImagesX, testImagesX  = train_test_split(df, images, test_size=0.25, random_state=42)

 
# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice


trainAttrX, testAttrX = datasets.process_house_attributes(df,trainAttrX, testAttrX)

mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['acc'])

model.fit([trainAttrX, trainImagesX], trainY,validation_data=([testAttrX, testImagesX], testY),	epochs=200, batch_size=8)


# model Prediction
testAttrXSample = np.expand_dims(testAttrX[0],axis=0)
testImagesXSample = np.expand_dims(testImagesX[0],axis=0)
p=model.predict([testAttrXSample,testImagesXSample])

print('Sample Test')
print('Predicted Value is :{}'.format(p[0][0]*maxPrice))
print('Actual Value is :{}'.format(testY.iloc[0]*maxPrice))