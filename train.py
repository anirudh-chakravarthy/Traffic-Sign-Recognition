import os 
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.optimizers import Adam

from models.lenet import LeNet_baseline, LeNet_modified
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from models.resnet import ResNet18
from utils.utils import *

# set random seed for result comparison
seed = 4192
np.random.seed(seed)


def train(args):
	base_dir = args.base_dir
	num_classes = args.num_classes
	model_name = str.lower(args.model)

	if model_name== 'lenet_baseline':
		input_shape = (32, 32, 3)
		model = LeNet_baseline(input_shape=input_shape, num_classes=num_classes)
	elif model_name == 'lenet_modified':
		input_shape = (32, 32, 3)
		model = LeNet_modified(input_shape=input_shape, num_classes=num_classes)
	elif model_name == 'alexnet':
		input_shape = (227, 227, 3)
		model = AlexNet(input_shape=input_shape, num_classes=num_classes)
	elif model_name == 'vgg16':
		input_shape = (224, 224, 3)
		model = VGG16(input_shape=input_shape, num_classes=num_classes)
	elif model_name == 'vgg19':
		input_shape = (224, 224, 3)
		model = VGG19(input_shape=input_shape, num_classes=num_classes)
	elif model_name == 'resnet18':
		input_shape = (112, 112, 3)
		model = ResNet18(input_shape=input_shape, num_classes=num_classes)
	else:
		print('Please choose an implemented model!')
		sys.exit()

	# get training dataset
	x, y = input_preprocess(base_dir, input_shape, num_classes)
	# split dataset into train and val set
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
	print('Train images loaded')

	# one-hot encoding for each label
	y_train = one_hot(y_train, num_classes)
	y_val = one_hot(y_val, num_classes)

	# train the model
	callbacks = get_callbacks(args.checkpoint, model_name)
	model.compile(optimizer=Adam(lr=args.lr),
				  loss='categorical_crossentropy',
				  metrics=['acc'])
	model.summary()
	history = model.fit(x_train, y_train,
			  batch_size=args.batch_size,
			  validation_data=(x_val, y_val),
			  epochs=args.epochs,
			  callbacks=callbacks)
	visualize(history)
	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='German Traffic Sign Recognition')
	parser.add_argument('-d', '--base_dir', default='', type=str, help='path to dataset (default: None)')
	parser.add_argument('-m', '--model', default='LeNet_baseline', type=str, help='Model to use\n\
					Choose between LeNet_baseline, LeNet_modified, AlexNet, VGG16, VGG19 and ResNet18 (default: LeNet_baseline)')
	parser.add_argument('-e', '--epochs', default=25, type=int, help='number of epochs to run (default: 25)')
	parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size (default: 1)')
	parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate (default: 1e-3)')
	parser.add_argument('--num_classes', default=43, type=int, help='Number of classes (default: 43 for GTSRB)')
	parser.add_argument('-c', '--checkpoint', default='./checkpoint', help='path to checkpoint directory')
	args = parser.parse_args()

	print(args)
	model = train(args)
