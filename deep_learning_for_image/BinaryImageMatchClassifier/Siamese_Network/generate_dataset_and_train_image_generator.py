from keras.optimizers import Adam, SGD
import numpy as np
import random
import os
import model as model_arch
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint, EarlyStopping
from ImageBatchGenerator import ImageBatchGenerator
from config import EPOCHS,INIT_LR,BS,IMG_HEIGTH_WIDTH,NUM_MATCH_NOT_MATCH_PAIR_CNT,TRAIN_IMAGE_PATH, TEST_IMAGE_PATH




if __name__=='__main__':
	random.seed(42)

	print("[INFO] compiling model...")
	model = model_arch.create_cnn(IMG_HEIGTH_WIDTH, IMG_HEIGTH_WIDTH, 3)

	# sgd = SGD(lr=0.01, clipvalue=0.5)
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,	metrics=["accuracy"])

	# train the network
	print("[INFO] training network...")

	reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_acc',
											factor=0.2,
											patience=2,
											verbose=0,
											mode='auto',
											min_delta=0.0001,
											cooldown=0,
											min_lr=0.001)

	tensorboard = TensorBoard(log_dir="logs",write_graph=True)
	filepath= os.path.join('models',"binary_aug_google_apparel.hdf5")
	model_check_point = ModelCheckpoint(filepath,
										monitor='val_acc',
										verbose=0,
										save_best_only=True,
										mode='max',
										period=1)
	earlystopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=20)

	train_image_batch_generator = ImageBatchGenerator(TRAIN_IMAGE_PATH)
	test_image_batch_generator = ImageBatchGenerator(TEST_IMAGE_PATH)

	model.fit_generator(train_image_batch_generator.get_image_batch(BS),
						callbacks = [reduce_lr_on_plateau,model_check_point,tensorboard, earlystopping], 
						validation_data = test_image_batch_generator.get_image_batch(BS),
						validation_steps = BS,
	                    steps_per_epoch= BS, epochs=EPOCHS)

	print('------------------------------------------------------------------')