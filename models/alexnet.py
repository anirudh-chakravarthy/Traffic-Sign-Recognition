from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def AlexNet(num_classes, input_shape=(227,227,3)):
	'''
	Standard AlexNet implementation as per the paper
	NOTE: Input size is (227,227,3), not (224,224,3) as Andrej Karpathy mentioned in his CS231n course.
	'''
	model = Sequential()
	kernel_initializer='he_normal'

	# first block
	model.add(Conv2D(input_shape=input_shape,
					 filters=96,
					 kernel_size=(11,11), strides=4,
					 kernel_initializer=kernel_initializer,
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3),
						   strides=2))
	model.add(BatchNormalization())

	# second block
	model.add(Conv2D(filters=256,
					 kernel_size=(5,5),
					 strides=1, padding='same',
					 kernel_initializer=kernel_initializer,
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3),
						   strides=2))
	model.add(BatchNormalization())

	model.add(Conv2D(filters=384,
					 kernel_size=(3,3),
					 strides=1, padding='same',
					 kernel_initializer=kernel_initializer,
					 activation='relu'))

	model.add(Conv2D(filters=384,
					 kernel_size=(3,3),
					 strides=1, padding='same',
					 kernel_initializer=kernel_initializer,
					 activation='relu'))

	model.add(Conv2D(filters=256,
					 kernel_size=(3,3),
					 strides=1, padding='same',
					 kernel_initializer=kernel_initializer,
					 activation='relu'))

	model.add(MaxPooling2D(pool_size=(3,3),
						   strides=2))
	model.add(Dropout(0.5))
	
	# FC layers
	model.add(Flatten())
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=4096, activation='relu'))
	model.add(Dense(units=num_classes, activation='softmax'))
	return model
	