#coding=utf-8
from keras.models import *
from keras.layers import *
import keras.backend as K

def resize_image(inp, s):
    return Lambda(lambda  x: K.resize_images(x, height_factor=s[0], width_factor=s[1], data_format='channels_last',
                                             ))(inp)

def pool_block(inp, pool_factor):
    h = K.int_shape(inp)[1]
    w = K.int_shape(inp)[2]
    pool_size = strides = [int(np.round( float(h) / pool_factor)), int(np.round( float(w )/ pool_factor))]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(inp)
    x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print(strides)
    x = resize_image(x, strides)
    return x

def PSPNet(nClasses, optimizer=None, input_width=384, input_height=576):

    assert input_height%192 == 0
    assert input_width%192 == 0
    
    input_size = (input_height, input_width, 3)
    inputs = Input(input_size)
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    x = inputs
    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(filter_size, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    f1 = x

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(128, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    f2 = x

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    f3 = x

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    f4 = x

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    f5 = x

    o = f3
    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(nClasses, (3, 3), padding='same')(o)
    o = resize_image(o, (8, 8))
    o_shape = Model(inputs, o).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    print(outputHeight)
    print(outputWidth)
    o = Reshape((nClasses, input_height * input_width))(o)
    o = Permute((2, 1))(o)
    model = Model(inputs, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

