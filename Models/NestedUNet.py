#coding=utf-8
import tensorflow as tf
import keras 
from keras.models import *
from keras.layers import *

class conv_block_nested(Model):

    def __init__(self, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = Activation('relu')
        self.conv1 = Conv2D(mid_ch, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(out_ch, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class NestedUNet(nClasses, input_height=224, input_width=224):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    inputs = Input(shape=(input_height, input_width, 3))
    t = 2
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    x0_0 = conv_block_nested(filters[0], filters[0])(x)

    x1_0 = conv_block_nested(filters[1], filters[1])(MaxPooling2D(strides=2)(x0_0))
    x0_1 = conv_block_nested(filters[0], filters[0])(Concatenate()([x0_0, UpSampling2D()(x1_0)]))

    x2_0 = conv_block_nested(filters[2], filters[2])(MaxPooling2D(strides=2)(x1_0))
    x1_1 = conv_block_nested(filters[1], filters[1])(Concatenate()([x1_0, UpSampling2D()(x2_0)]))
    x0_2 = conv_block_nested(filters[0], filters[0])(Concatenate()([x0_0, x0_1, UpSampling2D()(x1_1)]))

    x3_0 = conv_block_nested(filters[3], filters[3])(MaxPooling2D(strides=2)(x2_0))
    x2_1 = conv_block_nested(filters[2], filters[2])(Concatenate()([x2_0, UpSampling2D()(x3_0)]))
    x1_2 = conv_block_nested(filters[1], filters[1])(Concatenate()([x1_0, x1_1, UpSampling2D()(x2_1)]))
    x0_3 = conv_block_nested(filters[0], filters[0])(Concatenate()([x0_0, x0_1, x0_2, UpSampling2D()(x1_2)]))

    x4_0 = conv_block_nested(filters[4], filters[4])(MaxPooling2D(strides=2)(x3_0))
    x3_1 = conv_block_nested(filters[3], filters[3])(Concatenate()([x3_0, UpSampling2D()(x4_0)]))
    x2_2 = conv_block_nested(filters[2], filters[2])(Concatenate()([x2_0, x2_1, UpSampling2D()(x3_1)]))
    x1_3 = conv_block_nested(filters[1], filters[1])(Concatenate()([x1_0, x1_1, x1_2, UpSampling2D()(x2_2)]))
    x0_4 = conv_block_nested(filters[0], filters[0])(Concatenate()([x0_0, x0_1, x0_2, x0_3, UpSampling2D()(x1_3)]))

    o = Conv2D(nClasses, (3, 3), padding='same')(x0_4)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model