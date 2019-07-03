import os 
import sys
import argparse
import numpy as np

from sklearn.metrics import accuracy_score, classification_report

from models.lenet import LeNet_baseline, LeNet_modified
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from models.resnet import ResNet18
from utils.utils import *


def test(args):
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

	# load test set
	x, y = load_test(base_dir, input_shape, num_classes)
	print('Test images loaded')

	model.load_weights(args.pretrained_model)
	pred = model.predict_classes(x)
	accuracy = accuracy_score(y, pred)
	print('Accuracy: {}'.format(accuracy))
	print(classification_report(y, pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='German Traffic Sign Recognition')
    parser.add_argument('-m', '--model', default='', type=str, help='model to use (default: None)')
    parser.add_argument('--pretrained_model', default='', type=str, help='path to pretrained model (default: None)')
    parser.add_argument('-d', '--base_dir', default='', type=str, help='path to dataset (default: None)')
    parser.add_argument('--num_classes', default=43, type=int, help='Number of classes (43 for GTSRB)')
    args = parser.parse_args()

    test(args)