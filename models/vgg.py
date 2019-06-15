from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def VGG16(input_shape=(224,224,3), num_classes):
	model = Sequential()
	
	# first block, output=(112,112,64)
	model.add(Conv2D(input_shape=input_shape,
					 filters=64,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=64,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# second block, output=(56,56,128)
	model.add(Conv2D(filters=128,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=128,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# third block, output=(28,28,256)
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# fourth block, output=(14,14,512)
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# fifth block, output=(7,7,512)
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# FC layers
	model.add(Flatten())
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dense(units=num_classes, activation='softmax'))
	return model


def VGG19(input_shape=(224,224,3), num_classes):
	model = Sequential()

	# first block, output=(112,112,64)
	model.add(Conv2D(input_shape=input_shape,
					 filters=64,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=64,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# second block, output=(56,56,128)
	model.add(Conv2D(filters=128,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=128,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# third block, output=(28,28,256)
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# fourth block, output=(14,14,512)
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# fifth block, output=(7,7,512)
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(Conv2D(filters=512,
					 kernel_size=(3,3),
					 padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),
						   stride=2))

	# FC layers
	model.add(Flatten())
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dense(units=num_classes, activation='softmax'))
	return model

	