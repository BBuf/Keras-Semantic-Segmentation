"""
GhostNet https://arxiv.org/abs/1911.11907
"""
from keras.layers import *
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def ConvBnLayer(x, oup, kernel_size, stride, padding='valid'):
    y = Conv2D(filters=oup, kernel_size=kernel_size, strides=stride, padding=padding)(x)
    y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
    return y


def SELayer(x, reduction=4):
    batch, _, __, channel = x.shape
    y = GlobalAveragePooling2D()(x)
    y = Dense(units=channel // reduction, activation='relu')(y)
    y = Dense(units=channel, activation='sigmoid')(y)
    y = Reshape([1, 1, channel])(y)
    se_tensor = Multiply()([x, y])
    return se_tensor


def DepthWiseConv(x, kernel_size=3, stride=1, depth_multiplier=1, padding='same', relu=False):
    y = DepthwiseConv2D(kernel_size=kernel_size // 2, depth_multiplier=depth_multiplier,
                        strides=stride, padding=padding)(x)
    y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
    if relu:
        y = Activation('relu')(y)
    return y


def GhostModule(x, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
    init_channels = math.ceil(oup / ratio)
    new_channels = init_channels * (ratio - 1)

    multiplier = new_channels // init_channels

    primary_tensor = ConvBnLayer(x, init_channels, kernel_size=kernel_size, stride=stride, padding='same')
    if relu:
        primary_tensor = Activation('relu')(primary_tensor)

    cheap_tensor = DepthWiseConv(primary_tensor, kernel_size=dw_size,
                                 depth_multiplier=multiplier, padding='same', stride=1)
    if relu:
        cheap_tensor = Activation('relu')(cheap_tensor)

    out = Concatenate()([primary_tensor, cheap_tensor])
    # 使用Lambda进行切分
    return Lambda(lambda x: x[:, :, :, :oup])(out)


def GhostBottleneck(x, hidden_dim, oup, kernel_size, stride, use_se):
    assert stride in [1, 2]
    inp = x.shape[-1]
    if stride == 1 and inp == oup:
        shortcut = x
    else:
        shortcut = DepthWiseConv(x, kernel_size=3, stride=stride, relu=False)
        shortcut = ConvBnLayer(shortcut, oup, 1, 1, padding='same')

    x = GhostModule(x, hidden_dim, kernel_size=1, relu=True)
    if stride == 2:
        x = DepthWiseConv(x, kernel_size, stride, relu=False)
    if use_se:
        x = SELayer(x)
    x = GhostModule(x, oup, kernel_size=1, relu=False)
    return Add()([x, shortcut])


def GhostNet(x, num_classes=1000, width_mult=1.):
    cfgs = [
        # k, t, c, SE, s
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [5, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]

    output_channel = _make_divisible(16 * width_mult, 4)

    x = ConvBnLayer(x, output_channel, 3, 2, padding='same')
    for k, exp_size, c, use_se, s in cfgs:
        output_channel = _make_divisible(c * width_mult, 4)
        hidden_channel = _make_divisible(exp_size * width_mult, 4)
        x = GhostBottleneck(x, hidden_channel, output_channel, k, s, use_se)

    output_channel = _make_divisible(exp_size * width_mult, 4)
    x = ConvBnLayer(x, output_channel, kernel_size=1, stride=1, padding='same')
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output_channel = 1280
    x = Dense(output_channel)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes)(x)
    return x
