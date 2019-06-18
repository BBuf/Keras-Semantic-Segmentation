# coding=utf-8
from keras.models import *
from keras.layers import *

def Segnet(n_classes, input_height=512, input_width=512):
	img_input = Input(shape=(input_height, input_width, 3))
	#Encoder
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = (BatchNormalization())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = (BatchNormalization())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = (BatchNormalization())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = (BatchNormalization())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	#decoder
	o = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2)))(o)
	o = (ZeroPadding2D((1, 1)))(o)
	o = (Conv2D(256, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2)))(o)
	o = (ZeroPadding2D((1, 1)))(o)
	o = (Conv2D(128, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2)))(o)
	o = (ZeroPadding2D((1, 1)))(o)
	o = (Conv2D(64, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2)))(o)
	o = (ZeroPadding2D((1, 1)))(o)
	o = (Conv2D(32, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)
	
	o = Conv2D(n_classes, (3, 3), padding='same')(o)
	o_shape = Model(img_input, o).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]
	print(outputHeight)
	print(outputWidth)
	o = Reshape((n_classes, input_height * input_width))(o)
	o = Permute((2, 1))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	
	return model

