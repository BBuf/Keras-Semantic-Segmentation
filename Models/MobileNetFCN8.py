#coding=utf-8
from keras.models import *
from keras.layers import *
import keras
import keras.backend as K

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


def MobileNetFCN8 (nClasses, optimizer=None, input_width=512, input_height=512,  pretrained='imagenet'):
    input_size = (input_height, input_width, 3)
    img_input = Input(input_size)
    alpha = 1.0
    depth_multiplier = 1
    x = conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=1)
    f1 = x
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=3)
    f2 = x
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=5)
    f3 = x
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=11)
    f4 = x
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=13)
    f5 = x

    o = f5

    o = (Conv2D(256, (7, 7), activation='relu', padding='same'))(o)
    o = BatchNormalization()(o)


    o = (Conv2D(nClasses, (1, 1)))(o)
    # W = (N - 1) * S - 2P + F = 6 * 2 - 0 + 2 = 14
    o = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid")(o)
    # 14 x 14

    o2 = f4
    o2 = (Conv2D(nClasses, (1, 1)))(o2)

    # (14 x 14) (14 x 14)

    o = Add()([o, o2])
    # W = (N - 1) * S - 2P + F = 13 * 2 - 0 + 2 = 28
    o = Conv2DTranspose(nClasses, kernel_size=(2, 2),  strides=(2, 2), padding="valid")(o)
    o2 = f3 
    o2 = (Conv2D(nClasses,  (1, 1)))(o2)
    # (28 x 28) (28 x 28)
    o = Add()([o2, o])

    # 224 x 224
    # W = (N - 1) * S - 2P + F = 27 * 8 + 8 = 224
    o = Conv2DTranspose(nClasses , kernel_size=(8,8),  strides=(8,8), padding="valid")(o)

    o_shape = Model(img_input, o).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model
