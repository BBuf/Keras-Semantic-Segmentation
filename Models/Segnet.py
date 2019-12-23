# coding=utf-8
from keras.models import *
from keras.layers import *
from keras.engine import Layer
import keras.backend as K

class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, up_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.up_size = up_size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.up_size[0],
                    input_shape[2] * self.up_size[1],
                    input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            # y = mask // (output_shape[2] * output_shape[3])
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range
            y = (mask - f) // (output_shape[2] * output_shape[3]) - b * output_shape[1]
            x = (mask // output_shape[3]) % output_shape[2]

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            # indices_return = K.reshape(indices, (input_shape[0], 12, 4))
            # indices_return = K.cast(indices_return, 'float32')
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            # ret = K.cast(ret, 'float32')
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.up_size[0],
            mask_shape[2] * self.up_size[1],
            mask_shape[3]
        )


def Segnet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    #Encoder
    # 224x224
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1, mask1 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv1)
    # 112x112
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2, mask2 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)
    # 56x56
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3, mask3 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv3)
    # 28x28
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4, mask4 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4)
    # 14x14

    # decode
    up7 = MaxUnpooling2D()([pool4, mask4])
    #up7 = concatenate([up7, conv4], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    #up8 = concatenate([up8, conv3], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = MaxUnpooling2D()([conv8, mask3])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    #up9 = concatenate([up9, conv2], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = MaxUnpooling2D()([conv9, mask2])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    #up10 = concatenate([up10, conv1], axis=-1)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = MaxUnpooling2D()([conv10, mask1])
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = BatchNormalization()(conv10)

    outputHeight = Model(inputs, conv10).output_shape[1]
    outputWidth = Model(inputs, conv10).output_shape[2]
    conv11 = Conv2D(nClasses, (1, 1), padding='same')(conv10)
    conv11 = (Reshape((outputHeight*outputWidth, nClasses)))(conv11)
    conv11 = Activation('softmax')(conv11)

    model = Model(inputs, conv11)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
	
    return model

