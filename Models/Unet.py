#coding=utf-8
from keras.models import *
from keras.layers import *

def Unet(nClasses, input_width=224, input_height=224):
    input = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    o = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    o = BatchNormalization()(o)

    # decoder
    o = concatenate([UpSampling2D((2, 2))(o), conv4], axis=3)
    o = Conv2D(128, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)

    o = concatenate([UpSampling2D((2, 2), )(o), conv3], axis=3)
    o = Conv2D(64, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)

    o = concatenate([UpSampling2D((2, 2), )(o), conv2], axis=3)
    o = Conv2D(32, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)

    o = concatenate([UpSampling2D((2, 2), )(o), conv1], axis=3)
    o = Conv2D(16, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)

    outputHeight = Model(input, o).output_shape[1]
    outputWidth = Model(input, o).output_shape[2]
    o = Conv2D(nClasses, (1, 1), padding='same')(o)
    o = Reshape((nClasses, input_height * input_width))(o)
    o = Permute((2, 1))(o)
    o = (Activation('softmax'))(o)

    model = Model(input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    return model

