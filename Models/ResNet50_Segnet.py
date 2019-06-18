# coding=utf-8
import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K

# code taken from https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


def one_side_pad( x ):
    x = ZeroPadding2D((1, 1))(x)
    x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

bn_axis = 3

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1)  , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size  ,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3 , (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1)   , strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size   , padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1)   , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1)   , strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def Resnet_Segnet(n_classes, input_height=64, input_width=64):
        
        img_input = Input(shape=(input_height, input_width, 3))
        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        f1 = x

        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3)  , strides=(2, 2))(x)
    

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        f2 = one_side_pad(x )


        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        f3 = x 

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
        f4 = x 

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        f5 = x 
        print('****')
        print(x.get_shape)
       #Decoder
        o = Conv2DTranspose(2048, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        o = (BatchNormalization())(o)

        o = Conv2DTranspose(1024, (3, 3), strides=(2, 2), activation='relu', padding='same')(o) 
        o = (BatchNormalization())(o)

        o = Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        o = (BatchNormalization())(o)
        
        o = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        o = (BatchNormalization())(o)
        
        o = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        o = (BatchNormalization())(o)
        
        o = Conv2D(n_classes, (3, 3), padding='same')(o)
        o_shape = Model(img_input, o).output_shape
        outputHeight = o_shape[1]
        outputWidth = o_shape[2]
        print(outputHeight)
        print(outputWidth)
        o = Reshape((n_classes, input_height * input_width))(o)
        o = Permute((2, 1))(o)
        model = Model(img_input, o)
        model.outputWidth = outputWidth
        model.outputHeight = outputHeight

        return model

