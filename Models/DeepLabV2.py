# coding=utf-8
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *

class BilinearUpsampling(Layer):
    def __init__(self, upsampling, **kwargs):
        self.upsampling = upsampling
        super(BilinearUpsampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BilinearUpsampling, self).build(input_shape)

    def call(self, x, mask=None):
        new_size = [x.shape[1] * self.upsampling, x.shape[2] * self.upsampling]
        output = tf.image.resize_images(x, new_size)
        return output

def DeepLabV2(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    
     # Block 1
    x = ZeroPadding2D(padding=(1, 1))(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block 2
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_3')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block 4
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_3')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    # Block 5 
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(x)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(x)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    # branching for Atrous Spatial Pyramid Pooling - Until here -14 layers
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=nClasses, kernel_size=(1, 1), activation='relu', name='fc8_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=nClasses, kernel_size=(1, 1), activation='relu', name='fc8_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=nClasses, kernel_size=(1, 1), activation='relu', name='fc8_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=nClasses, kernel_size=(1, 1), activation='relu', name='fc8_4')(b4)

    s = Add()([b1, b2, b3, b4])
    logits = BilinearUpsampling(upsampling=8)(s)

    outputHeight = Model(inputs, logits).output_shape[1]
    outputWidth = Model(inputs, logits).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(logits)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model
