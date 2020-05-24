# coding=utf-8
from keras.layers import *
from keras.models import *
from keras.optimizers import *

def identity_block(x, f_kernel_size, filters, dilation, pad):
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x

    # stage 1
    x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    # stage 2
    x = ZeroPadding2D(padding=pad)(x)
    x = Conv2D(filters=filters_2, kernel_size=f_kernel_size, strides=(1, 1),
               dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)

    # stage 4
    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x

def convolutional_block(x, f_kernel_size, filters, strides, dilation, pad):
    filters_1, filters_2, filters_3 = filters
    x_shortcut = x

    # stage 1
    x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=strides, padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    # stage 2
    x = ZeroPadding2D(padding=pad)(x)
    x = Conv2D(filters=filters_2, kernel_size=f_kernel_size, strides=(1, 1), dilation_rate=dilation,
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1),
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    # stage 4
    x_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=strides, padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(momentum=0.95, axis=-1)(x_shortcut)

    # stage 5
    x = Add()([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x

def ResNet50(inputs):
    # stage 1
    #conv1_1_
    x = ZeroPadding2D(padding=(1, 1))(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    #conv1_2
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    # conv1_3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    #pool1
    x = ZeroPadding2D(padding=(1, 1))(x)
    x_stage_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    # x_stage_1 = Dropout(0.25)(x_stage_1)

    # stage 2
    x = convolutional_block(x_stage_1, f_kernel_size=(3, 3), filters=[64, 64, 256], strides=1, pad=(1, 1), dilation=1)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)
    x_stage_2 = identity_block(x, f_kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)
    # x_stage_2 = Dropout(0.25)(x_stage_2)

    # stage 3
    x = convolutional_block(x_stage_2, f_kernel_size=(3, 3), filters=[128, 128, 512], strides=2, pad=(1, 1), dilation=1)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1), dilation=1)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1), dilation=1)
    x_stage_3 = identity_block(x, f_kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1), dilation=1)
    # x_stage_3 = Dropout(0.25)(x_stage_3)

    # stage 4
    x = convolutional_block(x_stage_3, f_kernel_size=(3, 3), filters=[256, 256, 1024], strides=1, pad=(2, 2), dilation=2)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
    x_stage_4 = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
    # x_stage_4 = Dropout(0.25)(x_stage_4)

    # stage 5
    x = convolutional_block(x_stage_4, f_kernel_size=(3, 3), filters=[512, 512, 2048], strides=1, pad=(4, 4), dilation=4)
    x = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 2048], pad=(4, 4), dilation=4)
    x_stage_5 = identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 2048], pad=(4, 4), dilation=4)
    # x_stage_5 = Dropout(0.25)(x_stage_5)

    return x_stage_5


def PSPNet_ResNet50(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    # ResNet50提取特征
    res_features = ResNet50(inputs)

    #金字塔池化
    x_c1 = AveragePooling2D(pool_size=60, strides=60, name='ave_c1')(res_features)
    x_c1 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c1')(x_c1)
    x_c1 = BatchNormalization(momentum=0.95, axis=-1)(x_c1)
    x_c1 = Activation(activation='relu')(x_c1)
    #x_c1 = Dropout(0.2)(x_c1)
    x_c1 = UpSampling2D(size=(60, 60), name='up_c1')(x_c1)

    x_c2 = AveragePooling2D(pool_size=30, strides=30, name='ave_c2')(res_features)
    x_c2 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c2')(x_c2)
    x_c2 = BatchNormalization(momentum=0.95, axis=-1)(x_c2)
    x_c2 = Activation(activation='relu')(x_c2)
    #x_c2 = Dropout(0.2)(x_c2)
    x_c2 = UpSampling2D(size=(30, 30), name='up_c2')(x_c2)

    x_c3 = AveragePooling2D(pool_size=20, strides=20, name='ave_c3')(res_features)
    x_c3 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c3')(x_c3)
    x_c3 = BatchNormalization(momentum=0.95, axis=-1)(x_c3)
    x_c3 = Activation(activation='relu')(x_c3)
    #x_c3 = Dropout(0.2)(x_c3)
    x_c3 = UpSampling2D(size=(20, 20), name='up_c3')(x_c3)

    x_c4 = AveragePooling2D(pool_size=10, strides=10, name='ave_c4')(res_features)
    x_c4 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c4')(x_c4)
    x_c4 = BatchNormalization(momentum=0.95, axis=-1)(x_c4)
    x_c4 = Activation(activation='relu')(x_c4)
    #x_c4 = Dropout(0.2)(x_c4)
    x_c4 = UpSampling2D(size=(10, 10), name='up_c4')(x_c4)

    x_c5 = Conv2D(filters=512, kernel_size=1, strides=1, name='conv_c5', padding='same')(res_features)
    x_c5 = BatchNormalization(momentum=0.95, axis=-1)(x_c5)
    x_c5 = Activation(activation='relu')(x_c5)
    #x_c5 = Dropout(0.2)(x_c5)

    x = Concatenate(axis=-1, name='concat')([x_c1, x_c2, x_c3, x_c4, x_c5])
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='sum_conv_1_11')(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)

    x = UpSampling2D(size=(4, 4))(x)
    # x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='sum_conv_1_21')(x)
    # x = BatchNormalization(momentum=0.95, axis=-1)(x)

    x = Conv2D(nClasses, (1, 1), padding='same')(x)

    outputHeight = Model(inputs, x).output_shape[1]
    outputWidth = Model(inputs, x).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(x)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model
