#coding=utf-8
from keras.models import *
from keras.layers import *
import os

def FCN16(nClasses, input_height=224, input_width=224):

	img_input = Input(shape=(input_height, input_width, 3))

	x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f1 = x
	# 112 x 112
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f2 = x

	# 56 x 56
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f3 = x

	# 28 x 28
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f4 = x

	# 14 x 14
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f5 = x
	# 7 x 7        

	o = f5

	o = (Conv2D(256, (7, 7), activation='relu', padding='same'))(o)
	o = BatchNormalization()(o)


	o = (Conv2D(nClasses, (1, 1)))(o)
	# W = (N - 1) * S - 2P + F = 6 * 2 - 0 + 2 = 14 x 14
	o = UpSampling2D(size=(2, 2))(o)
	o = (Conv2D(nClasses, kernel_size=(1, 1)))(o)
	#o = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid")(o)
	# 14 x 14

	o2 = f4
	o2 = (Conv2D(nClasses, (1, 1)))(o2)
	
	# (28 x 28) (28 x 28)

	o = Add()([o, o2])
      
	# 224 x 224
	# W = (N - 1) * S - 2P + F = 27 * 8 + 8 = 224
	o = UpSampling2D(size=(16, 16))(o)
	o = Conv2D(nClasses, kernel_size=(1, 1))(o)	
	#o = Conv2DTranspose(nClasses , kernel_size=(16,16),  strides=(16,16), padding="valid")(o)
	
	o_shape = Model(img_input, o).output_shape
	
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
	o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model
