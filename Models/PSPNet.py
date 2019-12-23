#coding=utf-8
from keras.models import *
from keras.layers import *
import keras.backend as K

def resize_image(inp, s):
    return Lambda(lambda x: K.resize_images(x, height_factor=s[0], width_factor=s[1], data_format='channels_last', interpolation='bilinear'))(inp)

def pool_block(inp, pool_factor):
    h = K.int_shape(inp)[1]
    w = K.int_shape(inp)[2]
    print(h)
    print(w)
    pool_size = strides = [int(np.round( float(h) / pool_factor)), int(np.round( float(w)/ pool_factor))]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(inp)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = resize_image(x, strides)
    return x

def PSPNet(nClasses, input_width=384, input_height=384):
    assert input_height % 192 == 0
    assert input_width % 192 == 0
    inputs = Input(shape=(input_height, input_width, 3))

    x = (Conv2D(64, (3, 3), padding='same'))(inputs)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    f1 = x
    # 192 x 192

    x = (Conv2D(128, (3, 3), padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    f2 = x
    # 96 x 96
    x = (Conv2D(256, (3, 3), padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    f3 = x
    # 48 x 48
    x = (Conv2D(256, (3, 3), padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    f4 = x
    # 24 x 24
    x = (Conv2D(256, (3, 3), padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    f5 = x
    # 12 x 12
    o = f4
    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(256, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = resize_image(o, (8, 8))
    o = Conv2D(nClasses, (1, 1), padding='same')(o)
    
    o_shape = Model(inputs, o).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    print(outputHeight)
    print(outputWidth)
    o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
    o = (Activation('softmax'))(o)
    model = Model(inputs, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

