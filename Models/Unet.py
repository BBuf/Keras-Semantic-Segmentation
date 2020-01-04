#coding=utf-8
from keras.models import *
from keras.layers import *

def Unet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    # encode
    # 224x224
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = (Activation('relu'))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 112x112
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = (Activation('relu'))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 56x56
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = (Activation('relu'))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 28x28
    conv4 = Conv2D(256, (3, 3),  padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv1 = (Activation('relu'))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 14x14
    
    o = Conv2D(512, (3, 3), padding='same')(pool4)
    o = BatchNormalization()(o)
    # decode
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, pool3], axis=-1))
    o = (Conv2D(256, (3, 3), padding='same'))(o)
    o = (BatchNormalization())(o)
    o = (Activation('relu'))(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, pool2], axis=-1))
    o = (Conv2D(128, (3, 3), padding='same'))(o)
    o = (BatchNormalization())(o)
    o = (Activation('relu'))(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, pool1], axis=-1))

    o = (Conv2D(64, (3, 3), padding='same'))(o)
    o = (BatchNormalization())(o)
    o = (Activation('relu'))(o)

    o = Conv2D(nClasses, (3, 3), padding='same')(o)
    
    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]
    o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
    o = Activation('softmax')(o)

    model = Model(input=inputs, output=o)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth
    return model