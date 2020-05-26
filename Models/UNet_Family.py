#coding=utf-8
import tensorflow as tf
import keras 
from keras.models import *
from keras.layers import *

class conv_block(Model):
    def __init__(self, filters):
        super(conv_block, self).__init__()

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

class up_conv(Model):
    def __init__(self, filters):
        super(up_conv, self).__init__()
        self.up = Sequential([
            UpSampling2D(),
            Conv2D(filters, kernel_size=(3,3), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.up(x)
        return x

def UNet(nClasses, input_height=224, input_width=224):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    inputs = Input(shape=(input_height, input_width, 3))
    
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
    conv1 = conv_block(n1)(inputs)

    conv2 = MaxPooling2D(strides=2)(conv1)
    conv2 = conv_block(filters[1])(conv2)

    conv3 = MaxPooling2D(strides=2)(conv2)
    conv3 = conv_block(filters[2])(conv3)

    conv4 = MaxPooling2D(strides=2)(conv3)
    conv4 = conv_block(filters[3])(conv4)

    conv5 = MaxPooling2D(strides=2)(conv4)
    conv5 = conv_block(filters[4])(conv5)

    d5 = up_conv(filters[3])(conv5)
    d5 = Concatenate()([conv4, d5])

    d4 = up_conv(filters[2])(d5)
    d4 = Concatenate()([conv3, d4])
    d4 = conv_block(filters[2])(d4)

    d3 = up_conv(filters[1])(d4)
    d3 = Concatenate()([conv2, d3])
    d3 = conv_block(filters[1])(d3)

    d2 = up_conv(filters[0])(d3)
    d2 = Concatenate()([conv1, d2])
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


class Recurrent_blcok(Model):
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
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

class RRCNN_block(Model):
    def __init__(self, out_ch, t=2):
        super(RRCNN_block, self).__init__()

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

def R2UNet(nClasses, input_height=224, input_width=224):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
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
    d5 = Concatenate()([e4, d5])
    d5 = RRCNN_block(filters[3], t=t)(d5)

    d4 = up_conv(filters[2])(d5)
    d4 = Concatenate()([e3, d4])
    d4 = RRCNN_block(filters[2], t=t)(d4)

    d3 = up_conv(filters[1])(d4)
    d3 = Concatenate()([e2, d3])
    d3 = RRCNN_block(filters[1], t=t)(d3)

    d2 = up_conv(filters[0])(d3)
    d2 = Concatenate()([e1, d2])
    d2 = RRCNN_block(filters[0], t=t)(d2)


    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


class Attention_block(Model):
    """
    Attention Block
    """

    def __init__(self, filters):
        super(Attention_block, self).__init__()

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

def AttUNet(nClasses, input_height=224, input_width=224):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    inputs = Input(shape=(input_height, input_width, 3))
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = conv_block(filters[0])(inputs)

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = conv_block(filters[1])(e2)

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = conv_block(filters[2])(e3)

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = conv_block(filters[3])(e4)

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = conv_block(filters[4])(e5)

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


def R2AttUNet(nClasses, input_height=224, input_width=224):
     """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
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