import os 
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.optimizers import Adam

from models.lenet import LeNet_baseline, LeNet_modified
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from utils.utils import *

# set random seed for result comparison
seed = 4192
np.random.seed(seed)

def train(args):
	base_dir = args.base_dir
	# get training dataset
	x, y = input_preprocess(base_dir, (32,32,3))
	# split dataset into train and val set
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
	print("Train images loaded")

	# one-hot encoding for each label
	y_train = one_hot(y_train)
	y_val = one_hot(y_val)

	model = LeNet_modified(input_shape=(32,32,3), num_classes=43)
	model.compile(optimizer=Adam(lr=1e-3),
				  loss='categorical_crossentropy',
				  metrics=['acc'])
	model.summary()

	# train the model
	callbacks = get_callbacks(args.checkpoint)
	model.fit(x_train, y_train,
			  batch_size=args.batch_size,
			  validation_data=(x_val, y_val),
			  epochs=args.epochs,
			  callbacks=callbacks)
	return model


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='German Traffic Sign Recognition')
    parser.add_argument('-d', '--base_dir', default='', type=str, help='path to dataset (default: None)')
    parser.add_argument('-c', '--checkpoint', default='./checkpoint', help='path to checkpoint directory')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size (default: 1)')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='number of epochs to run (default: 20)')
    args = parser.parse_args()

    model = train(args)
