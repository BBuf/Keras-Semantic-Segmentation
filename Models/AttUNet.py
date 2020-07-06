#coding=utf-8
import tensorflow as tf 
from keras.models import *
from keras.layers import *

def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out
    

def AttUNet(nClasses, input_height=224, input_width=224):
    
    inputs = Input(shape=(input_height, input_width, 3))
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = conv_block(e5, filters[4])

    d5 = up_conv(e5, filters[3])
    x4 =  Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 =  Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 =  Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 =  Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model