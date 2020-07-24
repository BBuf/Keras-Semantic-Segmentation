from keras.layers import *


def ConvBNLayer(x, out_channels, kernel_size, stride=1, dilation=1, act=True):
    """
    卷积+归一化层
    Args:
        x: 输入张量
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        dilation: 膨胀系数，默认为1
        act: 若为True，则最后会进行Relu激活，否则直接返回BN归一化的值

    Returns:

    """
    x = Conv2D(out_channels, kernel_size, strides=stride,
               padding='same', dilation_rate=dilation)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    if act:
        return Activation('relu')(x)
    else:
        return x


def ACBlock(x, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, deploy=False):
    """
    ACBlock搭建，这里没有采取源代码的Crop，而是直接通过same填充
    Args:
        x: 输入张量
        out_channels: 输出通道
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 扩张卷积参数，默认都用1
        groups: 分组数，暂未用到，只是为了与源代码保持统一，方便后续自己可修改为Depthwise
        deploy: 若deploy为True，则转用3x3，这是推理阶段。否则分解成1x3, 3x1, 3x3卷积

    Returns:

    """
    if deploy:
        return Conv2D(out_channels, (kernel_size, kernel_size), strides=stride,
                      dilation_rate=dilation, use_bias=True, padding='same')(x)
    else:
        square_outputs = Conv2D(out_channels, (kernel_size, kernel_size), strides=stride,
                                dilation_rate=dilation, use_bias=False, padding='same')(x)
        square_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(square_outputs)

        # 计算Crop
        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (padding, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, padding)

        if center_offset_from_origin_border >= 0:
            vertical_outputs = x
            ver_conv_padding = ver_pad_or_crop
            horizontal_outputs = x
            hor_conv_padding = hor_pad_or_crop
        else:
            vertical_outputs = x

            ver_conv_padding = (0, 0)
            horizontal_outputs = x

            hor_conv_padding = (0, 0)

        vertical_outputs = ZeroPadding2D(padding=ver_conv_padding)(vertical_outputs)
        vertical_outputs = Conv2D(out_channels, kernel_size=(kernel_size, 1),
                                  strides=stride, padding='same', use_bias=False,
                                  dilation_rate=dilation)(vertical_outputs)
        vertical_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(vertical_outputs)

        horizontal_outputs = ZeroPadding2D(padding=hor_conv_padding)(horizontal_outputs)
        horizontal_outputs = Conv2D(out_channels, kernel_size=(kernel_size, 1),
                                    strides=stride, padding='same', use_bias=False,
                                    dilation_rate=dilation)(horizontal_outputs)
        horizontal_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(horizontal_outputs)

        results = Add()([square_outputs, vertical_outputs, horizontal_outputs])

        return results


def BasicBlock(x, out_channels, stride=1, downsample=False):
    """
    BasicBlock搭建
    Args:
        x: 输入张量
        out_channels: 输出通道数
        stride: 步长
        downsample: 是否下采样

    Returns:
    """
    residual = x
    x = ACBlock(x, out_channels, kernel_size=3, stride=stride)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = ACBlock(x, out_channels, kernel_size=3)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)

    if downsample:
        shortcut = ConvBNLayer(residual, out_channels, kernel_size=1, stride=stride)
        outputs = Add()([x, shortcut])
    else:
        outputs = Add()([x, residual])

    return Activation('relu')(outputs)


def BottleNeckBlock(x, out_channels, stride=1, downsample=False):
    """
    BottleNeckBlock搭建
    Args:
        x: 输入张量
        out_channels: 输出通道数
        stride: 步长
        downsample: 是否下采样

    Returns:

    """
    expansion = 4

    residual = x

    x = ConvBNLayer(x, out_channels, kernel_size=1, act=True)

    x = ACBlock(x, out_channels, kernel_size=3, stride=stride)

    x = ConvBNLayer(x, out_channels * expansion, kernel_size=1, act=False)

    if downsample:
        shortcut_tensor = ConvBNLayer(residual, out_channels * 4, kernel_size=1, stride=stride)
    else:
        shortcut_tensor = residual

    outputs = Add()([x, shortcut_tensor])
    return Activation('relu')(outputs)


def ResNet(x, block_type, layers_repeat, class_dim=1000):
    """
    构造resnet
    Args:
        x:
        block_type: 根据不同的类型，选择BasicBlock或BottlenectBlock
        layers_repeat: 每一层重复的次数
        class_dim: 最后Dense出来的维度

    Returns:

    """
    num_filters = [64, 128, 256, 512]

    x = ConvBNLayer(x, 64, kernel_size=7, stride=2, act=True)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for block in range(4):
        # 第一个模块做下采样
        downsample = True
        for i in range(layers_repeat[block]):
            x = block_type(x, num_filters[block], stride=2 if i == 0 and block != 0 else 1, downsample=downsample)
            downsample = False

    pool = GlobalAveragePooling2D()(x)
    output = Dense(class_dim, activation='relu')(pool)
    return output


def ResACNet(x, class_dim=1000, depth=50):
    """
    返回各个系列的由ACBlock组成的resnet
    Args:
        x: 输入张量
        class_dim: 输出类别数
        depth: 网络深度，参加下面assert允许的resnet深度

    Returns:

    Usage：
        input_shape = (224, 224, 3)
        inputs = Input(shape=input_shape, name="inputs")
        y = ResACNet(inputs, depth=50)

    """
    assert depth in [10, 18, 34, 50, 101, 152, 200]

    if depth == 10:
        output = ResNet(x, BasicBlock, [1, 1, 1, 1], class_dim)
    elif depth == 18:
        output = ResNet(x, BasicBlock, [2, 2, 2, 2], class_dim)
    elif depth == 34:
        output = ResNet(x, BasicBlock, [3, 4, 6, 3], class_dim)
    elif depth == 50:
        output = ResNet(x, BottleNeckBlock, [3, 4, 6, 3], class_dim)
    elif depth == 101:
        output = ResNet(x, BottleNeckBlock, [3, 4, 23, 3], class_dim)
    elif depth == 152:
        output = ResNet(x, BottleNeckBlock, [3, 8, 36, 3], class_dim)
    elif depth == 200:
        output = ResNet(x, BottleNeckBlock, [3, 24, 36, 3], class_dim)
    return output
