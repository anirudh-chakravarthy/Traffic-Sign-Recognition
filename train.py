from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from models.lenet import LeNet_baseline, LeNet_modified
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19


def train():
	