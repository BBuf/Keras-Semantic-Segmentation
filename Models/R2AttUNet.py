#coding=utf-8
import tensorflow as tf
import keras 
from keras.models import *
from keras.layers import *

class conv_block(Model):
    def __init__(self, filters):
        super(conv_block, self).__init__()
        self.output_dim = filters

        self.conv = Sequential([
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.conv(x)
        return x
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        return (input_shape[0], space[0], space[1], self.output_dim)

class up_conv(Model):
    def __init__(self, filters):
        super(up_conv, self).__init__()
        self.output_dim = filters

        self.up = Sequential([
            UpSampling2D(),
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.up(x)
        return x
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        return (input_shape[0], space[0]*2, space[1]*2, self.output_dim)

class Recurrent_block(Model):
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()
        self.output_dim = out_ch
        self.t = t
        self.conv = Sequential([
            Conv2D(out_ch, kernel_size=(3, 3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        return (input_shape[0], space[0], space[1], self.output_dim)


class RRCNN_block(Model):
    def __init__(self, out_ch, t=2):
        super(RRCNN_block, self).__init__()
        self.output_dim = out_ch
        self.RCNN = Sequential([
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        ])
        self.Conv = Conv2D(out_ch, kernel_size=(1, 1), strides=1, padding='same')

    def call(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        return (input_shape[0], space[0], space[1], self.output_dim)

class Attention_block(Model):
    """
    Attention Block
    """

    def __init__(self, filters):
        super(Attention_block, self).__init__()
        self.output_dim = filters
        self.W_g = Sequential([
            Conv2D(filters, kernel_size=1, strides=1, padding='same'),
            BatchNormalization()
        ])

        self.W_x = Sequential([
            Conv2D(filters, kernel_size=1, strides=1, padding='same'),
            BatchNormalization()
        ])

        self.psi = Sequential([
            Conv2D(filters, kernel_size=1, strides=1, padding='same'),
            BatchNormalization(),
            Activation('sigmoid')
        ])

        self.relu = Activation('relu')

    def call(self, x):
        g1 = self.W_g(x[0])
        x1 = self.W_x(x[1])
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x[1] * psi
        return out
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        return (input_shape[0], space[0]*2, space[1]*2, self.output_dim)

def R2AttUNet(nClasses, input_height=224, input_width=224):
    # """
    #Residual Recuurent Block with attention Unet
    #Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    #"""
    inputs = Input(shape=(input_height, input_width, 3))
    t = 2
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = RRCNN_block(filters[0], t=t)(inputs)

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = RRCNN_block(filters[1], t=t)(e2)

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = RRCNN_block(filters[2], t=t)(e3)

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = RRCNN_block(filters[3], t=t)(e4)

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = RRCNN_block(filters[4], t=t)(e5)

    d5 = up_conv(filters[3])(e5)
    x4 =  Attention_block(filters[3])([d5, e4])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(filters[3])(d5)

    d4 = up_conv(filters[2])(d5)
    x3 =  Attention_block(filters[2])([d4, e3])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(filters[2])(d4)

    d3 = up_conv(filters[1])(d4)
    x2 =  Attention_block(filters[1])([d3, e2])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(filters[1])(d3)

    d2 = up_conv(filters[0])(d3)
    x1 =  Attention_block(filters[0])([d2, e1])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(filters[0])(d2)

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model
