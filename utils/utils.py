import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd

from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
import matplotlib.pyplot as plt


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
	x = x / 255.0
	return x, y


# one-hot encoding of labels
def one_hot(labels, num_classes=43):
	return to_categorical(labels, num_classes)


# set keras callbacks
def get_callbacks(ckpt_dir, model_name):
	ckpt_path = os.path.join(ckpt_dir, model_name + '.hdf5')
	earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=1, restore_best_weights=False)
	reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
	modelckpt = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	tb = TensorBoard()
	return [earlystop, reducelr, modelckpt, tb]


# visualize train and validation accuracy
def visualize(model_history):
	# Training plots
	epochs = [i for i in range(1, len(model_history.history['loss'])+1)]

	plt.plot(epochs, model_history.history['loss'], color='blue', label="training_loss")
	plt.plot(epochs, model_history.history['val_loss'], color='red', label="validation_loss")
	plt.legend(loc='best')
	plt.title('loss')
	plt.xlabel('epoch')
	plt.show()

	plt.plot(epochs, model_history.history['acc'], color='blue', label="training_accuracy")
	plt.plot(epochs, model_history.history['val_acc'], color='red',label="validation_accuracy")
	plt.legend(loc='best')
	plt.title('accuracy')
	plt.xlabel('epoch')
	plt.show()


# load test set images
def load_test(base_dir, input_shape):
	x, y = [], []
	test_df = pd.read_csv(os.path.join(base_dir, 'Test.csv')) 
	for path in test_df['Path']:
		img = cv2.imread(os.path.join(base_dir, path))
		img = Image.fromarray(img, 'RGB')
		resized_img = img.resize((input_shape[0], input_shape[1]))
		x.append(np.array(resized_img))
		label = test_df.loc[test_df['Path'] == path, ['ClassId']]
		# print(type(label['ClassId'].iloc[0]))
		y.append(label['ClassId'].iloc[0])
	x, y = np.asarray(x), np.asarray(y)
	x = x / 255.0
	return x, y
