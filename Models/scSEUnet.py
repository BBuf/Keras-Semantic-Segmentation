import keras.backend as K
from keras.layers import *
from keras.models import Model

# spatial squeeze by mean and channel excitation


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)  # H W 求均值
    # K.int_shape() Returns the shape of tensor or variable as a tuple of int or None entries
    lin1 = Dense(K.int_shape(prevlayer)[
                 3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[
                 3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x

# channel squeeze and spatial excitation


def sse_block(prevlayer, prefix):
    # Bug? Should be 1 here?
    conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
                  activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv

# concurrent spatial and channel squeeze and channel excitation


def csse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579

    x = csse_block(x, prefix='csse_block_{}'.format(i))
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


def scSEUnet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv5))
    up6 = BatchNormalization()(up6)

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv6))
    up7 = BatchNormalization()(up7)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv7 = csse_block(conv7, prefix="conv7")

    up8 = Conv2D(32,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv7))
    up8 = BatchNormalization()(up8)

    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)

    conv8 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    conv8 = csse_block(conv8, prefix="conv8")

    up9 = Conv2D(16,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv8))
    up9 = BatchNormalization()(up9)

    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv9 = csse_block(conv9, prefix="conv9")

    conv10 = Conv2D(nClasses, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)

    outputHeight = Model(inputs, conv10).output_shape[1]
    outputWidth = Model(inputs, conv10).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(conv10)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model
