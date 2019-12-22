#coding=utf-8
from keras.models import *
from keras.layers import *


def FCN32(n_classes,  input_height=224, input_width=224):

	img_input = Input(shape=(input_height, input_width, 3))

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f1 = x
	# 112 x 112
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f2 = x

	# 64 x 64
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f3 = x
 
	# 32 x 32
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f4 = x

	# 16 x 16
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f5 = x
	# 8 x 8	

	o = f5

	o = (Conv2D(256, (7, 7), activation='relu', padding='same'))(o)
	x = BatchNormalization()(x)

	o = (Conv2D(n_classes,  (1, 1)))(o)
	o = Conv2DTranspose(n_classes, kernel_size=(64, 64),  strides=(32, 32))(o)
	o_shape = Model(img_input, o).output_shape
	
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
	o = (Activation('softmax'))(o)
	model = Model(img_input, o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model
