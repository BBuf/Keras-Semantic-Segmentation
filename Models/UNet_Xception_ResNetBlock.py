# coding=utf-8
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications.xception import Xception

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


def Unet_Xception_ResNetBlock(nClasses, input_height=224, input_width=224):
    
    backbone = Xception(input_shape=(input_height, input_width, 3), weights=None, include_top=False)
    
    inputs = backbone.input

    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)
    
     # Middle
    convm = Conv2D(16*32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, 16*32)
    convm = residual_block(convm, 16*32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    # 8 -> 16
    deconv4 = Conv2DTranspose(16*16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)
    
    uconv4 = Conv2D(16*16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, 16 * 16)
    uconv4 = residual_block(uconv4, 16*16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(16*8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(0.1)(uconv3)
    
    uconv3 = Conv2D(16*8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, 16*8)
    uconv3 = residual_block(uconv3, 16*8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(16*4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(16*4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, 16*4)
    uconv2 = residual_block(uconv2, 16*4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    # 64 -> 128
    deconv1 = Conv2DTranspose(16*2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(16*2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, 16*2)
    uconv1 = residual_block(uconv1, 16*2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    # 128 -> 256
    uconv0 = Conv2DTranspose(16*1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(16*1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, 16*1)
    uconv0 = residual_block(uconv0, 16*1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(0.1/2)(uconv0)

    out = Conv2D(nClasses, (3, 3), padding='same')(uconv0)
    out = BatchNormalization()(out)

    outputHeight = Model(inputs, out).output_shape[1]
    outputWidth = Model(inputs, out).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(out)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model
