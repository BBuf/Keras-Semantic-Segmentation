#coding=utf-8
from keras.models import *
from keras.layers import *

def Unet (nClasses, input_width=512, input_height=512):
    input = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2), )(conv4), conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    outputHeight = Model(input, conv5).output_shape[1]
    outputWidth = Model(input, conv5).output_shape[2]
    o = Conv2D(nClasses, (1, 1), padding='same')(conv5)
    o = Reshape((nClasses, input_height * input_width))(o)
    o = Permute((2, 1))(o)
    o = (Activation('softmax'))(o)

    model = Model(input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    return model

