import os 
import sys
import argparse
import numpy as np

from tensorflow import keras
from sklearn.metrics import accuracy_score

from models.lenet import LeNet_baseline, LeNet_modified
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from utils.utils import *


def test(args):
	base_dir = args.base_dir
	
	if str.lower(args.model) == 'lenet_baseline':
		input_shape = (32, 32, 3)
		model = LeNet_baseline(input_shape=input_shape, num_classes=43)
	elif str.lower(args.model) == 'lenet_modified':
		input_shape = (32, 32, 3)
		model = LeNet_modified(input_shape=input_shape, num_classes=43)
	elif str.lower(args.model) == 'alexnet':
		input_shape = (227, 227, 3)
		model = AlexNet(input_shape=input_shape, num_classes=43)
	elif str.lower(args.model) == 'vgg16':
		input_shape = (224, 224, 3)
		model = VGG16(input_shape=input_shape, num_classes=43)
	elif str.lower(args.model) == 'vgg19':
		input_shape = (224, 224, 3)
		model = VGG19(input_shape=input_shape, num_classes=43)
	else:
		print('Please choose an implemented model!')
		sys.exit()

	# load test set
	x, y = load_test(base_dir, input_shape)
	print('Test images loaded')

	model.load_weights(args.pretrained_model)
	pred = model.predict_classes(x)
	accuracy = accuracy_score(y, pred)
	print('Accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='German Traffic Sign Recognition')
    parser.add_argument('-m', '--model', default='', type=str, help='model to use (default: None)')
    parser.add_argument('--pretrained_model', default='', type=str, help='path to pretrained model (default: None)')
    parser.add_argument('-d', '--base_dir', default='', type=str, help='path to dataset (default: None)')
    args = parser.parse_args()

    test(args)