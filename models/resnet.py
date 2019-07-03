from tensorflow import keras
from keras import Model
from keras.layers import Input, add, Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, \
			Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l2


def _residual_block(residual, input):
	'''
	Adds the input and residual connection and returns sum
	Shortcut layer is of a larger size than input layer
	'''
	input_shape = input.get_shape().as_list()
	residual_shape = residual.get_shape().as_list()
	filters = input_shape[-1]
	strides = 2

	# downsizing to enable additions
	shortcut = Conv2D(filters=filters, 
					  kernel_size=(1,1),	
					  strides=strides,
					  padding='valid')(residual)
	shortcut = pad(shortcut, input_shape)
	return add([input, shortcut])


def _bn_relu(input):
	'''
	Helper function to build a BN -> relu block
	'''
	bn = BatchNormalization()(input)
	return Activation(activation='relu')(bn)


def _conv_bn_relu(input, filters, kernel_size, strides, padding='same', kernel_initializer='he_normal'):
	'''
	Helper function to build a conv -> BN -> relu block
	'''
	kernel_regularizer = l2(1.e-4)
	conv = Conv2D(filters=filters, kernel_size=kernel_size,
				  strides=strides, padding='same',
				  kernel_initializer=kernel_initializer,
				  kernel_regularizer=kernel_regularizer)(input)
	if padding == 'same':
		# print(conv.get_shape(), input.get_shape())
		conv = pad(conv, input.get_shape().as_list())
		# print(conv.get_shape(), input.get_shape())
	return _bn_relu(conv)


def pad(input, shape):
	'''
	Helper function to pad input layer to given shape
	'''
	input_shape = input.get_shape().as_list()
	in_height, in_width = input_shape[1], input_shape[2]
	out_height, out_width = shape[1], shape[2]

	pad_width = out_width - in_width
	pad_height = out_height - in_height

	pad_top = pad_height // 2
	pad_bottom = pad_height - pad_top
	pad_left = pad_width // 2
	pad_right = pad_width - pad_left

	padding = ((pad_top, pad_bottom), (pad_left, pad_right))
	return ZeroPadding2D(padding=padding)(input)


def ResNet18(num_classes, input_shape=(112,112,3)):
	input = Input(shape=input_shape)

	# first layer
	conv_1 = _conv_bn_relu(input=input,
						   filters=64,
						   kernel_size=(7,7),
						   strides=2,
						   padding='same')
	pool_1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv_1)

	# second layer
	conv2_1 = _conv_bn_relu(input=pool_1,
							filters=64,
					 		kernel_size=(3,3),
							strides=1,
							padding='same')
	conv2_2 = _conv_bn_relu(input=conv2_1,
							filters=64,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	res2_1 = _residual_block(pool_1, conv2_2)

	conv2_3 = _conv_bn_relu(input=res2_1,
							filters=64,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv2_4 = _conv_bn_relu(input=conv2_3,
							filters=64,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='valid')
	res2_2 = _residual_block(res2_1, conv2_4)

	# third layer
	conv3_1 = _conv_bn_relu(input=res2_2,
							filters=128,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv3_2 = _conv_bn_relu(input=conv3_1,
							filters=128,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	res3_1 = _residual_block(res2_2, conv3_2)

	conv3_3 = _conv_bn_relu(input=res3_1,
							filters=128,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv3_4 = _conv_bn_relu(input=conv3_3,
							filters=128,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='valid')
	res3_2 = _residual_block(res3_1, conv3_4)

	# fourth layer
	conv4_1 = _conv_bn_relu(input=res3_2,
							filters=256,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv4_2 = _conv_bn_relu(input=conv4_1,
							filters=256,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	res4_1 = _residual_block(res3_2, conv4_2)

	conv4_3 = _conv_bn_relu(input=res4_1,
							filters=256,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv4_4 = _conv_bn_relu(input=conv4_3,
							filters=256,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='valid')
	res4_2 = _residual_block(res4_1, conv4_4)

	# fifth layer
	conv5_1 = _conv_bn_relu(input=res4_2,
							filters=512,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv5_2 = _conv_bn_relu(input=conv5_1,
							filters=512,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	res5_1 = _residual_block(res4_2, conv5_2)

	conv5_3 = _conv_bn_relu(input=res5_1,
							filters=512,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='same')
	conv5_4 = _conv_bn_relu(input=conv5_3,
							filters=512,
					 		kernel_size=(3,3),
					 		strides=2,
					 		padding='valid')
	res5_2 = _residual_block(res5_1, conv5_4)

	# fully connected layers
	avgpool_1 = GlobalAveragePooling2D()(res5_2)
	dropout_1 = Dropout(0.5)(avgpool_1)
	dense_1 = Dense(units=1000, activation='relu')(dropout_1)

	dropout_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(units=num_classes, activation='softmax')(dropout_2)

	model = Model(inputs=input, outputs=dense_2)
	return model
