#coding=utf-8
from keras.models import *
from keras.layers import *
import keras
import keras.backend as K

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')


def relu6(x):
    return K.relu(x, max_value=6)

# Width Multiplier: Thinner Models
def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs(outputWidth1 - outputWidth2)
    cy = abs(outputHeight2 - outputHeight1)

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if outputHeight1 > outputHeight2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2

def MobileNetFCN32 (nClasses, optimizer=None, input_width=512, input_height=512,  pretrained='imagenet'):
    input_size = (input_height, input_width, 3)
    img_input = Input(input_size)
    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3
    x = conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x

    o = f5
    
    o = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu', padding='same'))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(nClasses,  (1, 1), kernel_initializer='he_normal'))(o)
    o = Conv2DTranspose(nClasses, kernel_size=(64, 64),  strides=(32, 32), use_bias=False)(o)
    o_shape = Model(img_input, o).output_shape
    
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((-1, outputHeight*outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model