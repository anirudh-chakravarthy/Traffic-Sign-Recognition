from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense,\
							 Dropout, BatchNormalization, MaxPooling2D


def LeNet_baseline(num_classes, input_shape=(32,32,1)):
	'''
	Standard LeNet-5 architecture implementation as per the paper
	'''
	model = Sequential()
	
	# first conv layer, output = (28,28,6)
	model.add(Conv2D(input_shape=input_shape,
					 filters=6, 
					 kernel_size=(5,5),
					 activation='tanh'))
	# Average Pooling, output = (14,14,6)
	model.add(AveragePooling2D(pool_size=(2,2),
							   strides=2))
	# second conv layer, output=(10,10,16)
	model.add(Conv2D(filters=16, 
					 kernel_size=(5,5),
					 activation='tanh'))
	# Average Pooling, output=(5,5,16)
	model.add(AveragePooling2D(pool_size=(2,2),
							   strides=2))
	# third conv layer, output=(1,1,120)
	model.add(Conv2D(filters=120, 
					 kernel_size=(5,5),
					 activation='tanh'))

	# FC layers
	model.add(Flatten())
	model.add(Dense(units=120, activation='tanh'))
	model.add(Dense(units=84, activation='tanh'))
	model.add(Dense(units=num_classes, activation='softmax'))
	return model


def LeNet_modified(num_classes, input_shape=(32,32,1)):
	'''
	Modified implementation using LeNet as the backbone architecture
		1. Uses Relu activation instead of tanh
		2. Use Maxpooling instead of average pooling
		3. Added BatchNormalization after conv layers
		4. Introduced Dropout after FC layer
		5. Added He initialization for conv layers
	'''
	model = Sequential()
	kernel_initializer = 'he_normal'
	
	# first conv layer, output = (28,28,6)
	model.add(Conv2D(input_shape=input_shape,
					 filters=6, 
					 kernel_size=(5,5),
					 kernel_initializer=kernel_initializer,
					 activation='relu'))
	model.add(BatchNormalization())
	# Average Pooling, output = (14,14,6)
	model.add(MaxPooling2D(pool_size=(2,2),
							   strides=2))
	# second conv layer, output=(10,10,16)
	model.add(Conv2D(filters=16, 
					 kernel_size=(5,5),
					 kernel_initializer=kernel_initializer,
					 activation='relu'))
	model.add(BatchNormalization())
	# Average Pooling, output=(5,5,16)
	model.add(MaxPooling2D(pool_size=(2,2),
							   strides=2))
	# third conv layer, output=(1,1,120)
	model.add(Conv2D(filters=120, 
					 kernel_size=(5,5),
					 kernel_initializer=kernel_initializer,
					 activation='relu'))
	model.add(BatchNormalization())

	# FC Layers
	model.add(Flatten())
	model.add(Dense(units=120, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(units=84, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(units=num_classes, activation='softmax'))
	return model