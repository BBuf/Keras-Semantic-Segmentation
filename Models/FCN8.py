#coding=utf-8
from keras.models import *
from keras.layers import *
import os

# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape


    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape

    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0),  (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0),  (0, cx)))(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy),  (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy),  (0, 0)))(o2)

    return o1, o2

def FCN8(nClasses, input_height=224, input_width=224):

	img_input = Input(shape=(input_height, input_width, 3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f1 = x
	# 112 x 112
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	f2 = x

	# 56 x 56
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
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
	# W = (N - 1) * S - 2P + F = 6 * 2 - 0 + 4 = 16
	o = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), padding="valid")(o)


	o2 = f4
	o2 = (Conv2D(nClasses, (1, 1)))(o2)
	
	
	o, o2 = crop(o, o2, img_input)
	o = Add()([o, o2])
        # W = (N - 1) * S - 2P + F = 13 * 2 - 0 + 2 = 28
	o = Conv2DTranspose(nClasses, kernel_size=(4, 4),  strides=(2, 2), padding="valid")(o)
	o2 = f3 
	o2 = (Conv2D(nClasses,  (1, 1)))(o2)
        
	o2, o = crop(o2, o, img_input)
	o = Add()([o2, o])

	# W = (N - 1) * S - 2P + F = 27 * 8 + 8 = 224
	o = Conv2DTranspose(nClasses , kernel_size=(16,16),  strides=(8,8), padding="valid")(o)
	
	o_shape = Model(img_input, o).output_shape
	
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
	o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model