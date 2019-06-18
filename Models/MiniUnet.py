#coding=utf-8
from keras.models import *
from keras.layers import *

def MiniUnet(nClasses, optimizer=None, input_width=512, input_height=512):
    input_size = (input_height, input_width, 3)
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(inputs)
    conv1 = (BatchNormalization())(conv1)
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = (BatchNormalization())(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding = 'same')(pool2)
    conv3 = (BatchNormalization())(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding = 'same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis = 3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding = 'same')(up1)
    conv4 = (BatchNormalization())(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding = 'same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis = 3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding = 'same')(up2)
    conv5 = (BatchNormalization())(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding = 'same')(conv5)
    
    conv6 = Conv2D(nClasses, (1, 1), activation='relu', padding = 'same')(conv5)

    o_shape = Model(inputs, conv6).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    conv6 = Reshape((nClasses, input_height * input_width))(conv6)
    conv6 = Permute((2, 1))(conv6)

    conv7 = (Activation('softmax'))(conv6)

    model = Model(inputs, conv7)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    return model

