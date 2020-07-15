"""
MobileNext模型

https://arxiv.org/abs/2007.02269

知乎版解析：https://zhuanlan.zhihu.com/p/157878449
"""

from keras.layers import *


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def ConvBNReLU(x, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=False):
    """
    Conv + BN + RELU6
    Args:
        x: 输入张量
        out_planes: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        groups: 分组，当=1则为普通卷积，其他情况则是depthwise
        norm_layer: 若为False则不做BN，为True则过一层BN

    Returns:

    """
    if groups == 1:
        x = Conv2D(filters=out_planes, kernel_size=kernel_size, strides=stride, padding='same')(x)
    else:
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same')(x)

    if norm_layer:
        x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    # relu 6 , 限制最大输出在6
    x = ReLU(max_value=6)(x)
    return x


def CutOutLayer(input_tensor, oup):
    """
    在通道维上截取到 oup 的张量
    Args:
        input_tensor: 输入张量
        oup: 输出通道数

    Returns:

    """
    return Lambda(lambda x: x[:, :, :, :oup])(input_tensor)


def CutInLayer(input_tensor, oup):
    """
    在通道维上截取从 oup开始到后面的张量

    Args:
        input_tensor: 输入张量
        oup: 输出通道数
    Returns:

    """
    return Lambda(lambda x: x[:, :, :, oup:])(input_tensor)


def SandGlass(x, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=False):
    """
    SandGlass 模块
    Args:
        x: 输入张量
        oup: 输出通道数
        stride: 步长
        expand_ratio: 扩张系数
        identity_tensor_multiplier: 区间在0-1的浮点数，用于部分通道残差连接，
                                    默认为1，即原始残差连接
        norm_layer: 若为False则不做BN，为True则过一层BN

    Returns:

    """
    assert stride in [1, 2]
    # 残差连接
    residual = x

    inp = x.shape[-1]
    use_identity = False if identity_tensor_multiplier == 1.0 else True
    identity_tensor_channels = int(round(inp * identity_tensor_multiplier))

    hidden_dim = int(round(inp / expand_ratio))
    use_res_connect = stride == 1 and inp == oup

    # depthwise
    x = ConvBNReLU(x, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer)
    if expand_ratio != 1:
        x = Conv2D(hidden_dim, kernel_size=1, strides=1, padding='same')(x)
        x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    # pointwise
    x = ConvBNReLU(x, oup, kernel_size=1, stride=1, norm_layer=norm_layer)
    # depthwise linear
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(x)
    if norm_layer:
        x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)

    if use_res_connect:
        if use_identity:
            # 在0：identity_tensor_channels上做残差连接
            identity_tensor = Add()([CutOutLayer(x, identity_tensor_channels),
                                     CutOutLayer(residual, identity_tensor_channels)])
            # 进行拼接
            out = Concatenate()([identity_tensor, CutInLayer(x, identity_tensor_channels)])
        else:
            # 直接残差
            out = x + residual
        return out
    else:
        # 直接返回
        return x


def MobileNext(x, num_classes=1000,
               width_mult=1.0,
               identity_tensor_multiplier=1.0,
               sand_glass_setting=None,
               round_nearest=8,
               norm_layer=True):
    """
    MobileNext网络主体
    Args:
        x: 输入张量
        num_classes: 最后输出分类数目
        width_mult: 宽度扩张
        identity_tensor_multiplier: 参见SandGlass模块
        sand_glass_setting: 用于设置SandGlass模块的各个参数，若为None则采取论文的设置
        round_nearest:
        norm_layer:

    使用方法，这里以Input层作为演示

    input_shape = (224, 224, 3)

    inputs = Input(shape=input_shape, name="inputs")

    y = MobileNext(inputs, 1000, width_mult=1.0, identity_tensor_multiplier=0.5)

    Returns: (batch, num_classes)的张量

    """
    input_channel = 32
    last_channel = 1280
    lc = last_channel
    if sand_glass_setting is None:
        sand_glass_setting = [
            # t, c,  b, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 2, 1],
            [6, lc, 1, 1],
        ]
    # only check the first element, assuming user knows t,c,n,s are required
    if len(sand_glass_setting) == 0 or len(sand_glass_setting[0]) != 4:
        raise ValueError("sand_glass_setting should be non-empty "
                         "or a 4-element list, got {}".format(sand_glass_setting))

    # first layer
    input_channel = _make_divisible(input_channel * width_mult, round_nearest)
    x = ConvBNReLU(x, input_channel, stride=2, norm_layer=norm_layer)

    for t, c, b, s in sand_glass_setting:
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(b):
            stride = s if i == 0 else 1
            x = SandGlass(x, output_channel, stride, expand_ratio=t,
                          identity_tensor_multiplier=identity_tensor_multiplier, norm_layer=True)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    return x
