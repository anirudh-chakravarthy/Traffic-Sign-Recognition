import os
import numpy as np
import cv2
from PIL import Image

from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical


# get training images from directory
def input_preprocess(base_dir, input_shape, num_classes=43):
	x, y = [], []
	train_dir = os.path.join(base_dir, 'Train')
	for i in range(num_classes):
		class_path = os.path.join(train_dir, str(i))
		images = os.listdir(class_path)
		for image in images:
			img = cv2.imread(os.path.join(class_path, image))
			img = Image.fromarray(img, 'RGB')
			resized_img = img.resize((input_shape[0], input_shape[1]))
			x.append(np.array(resized_img))
			y.append(i)
	x, y = np.asarray(x), np.asarray(y)
	return x, y


# one-hot encoding of labels
def one_hot(labels, num_classes=43):
	return to_categorical(labels, num_classes)


# set keras callbacks
def get_callbacks(ckpt_dir, model_name):
	ckpt_path = os.path.join(ckpt_dir, str.lower(model_name) + '.hdf5')
	earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=False)
	reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
	modelckpt = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	tb = TensorBoard()
	return [earlystop, reducelr, modelckpt, tb]
