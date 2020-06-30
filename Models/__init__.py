from __future__ import absolute_import

from .ENet import *
from .FCN8 import *
from .VGGFCN8 import *
from .ICNet import *
from .MobileNetFCN8 import *
from .MobileNetUnet import *
from .PSPNet import *
from .Segnet import *
from .SEUNet import *
from .Unet import *
from .scSEUnet import *
from .VGGUnet import *
from .DeepLabV2 import *
from .PSPNet_ResNet50 import *
from .UNet_Xception_ResNetBlock import *
from .HRNet import *


__model_factory = {
    'enet': ENet,
    'fcn8': FCN8,
    'mobilenet_fcn8': MobileNetFCN8,
    'vggfcn8': VGGFCN8,
    'unet': UNet,
    'vggunet': VGGUnet,
    'unet_xception_resnetblock': Unet_Xception_ResNetBlock,
    'mobilenet_unet': MobileNetUnet,
    'seunet': SEUnet,
    'scseunet': scSEUnet,
    'segnet': Segnet,
    'pspnet': PSPNet,
    'pspnet_resnet50': PSPNet_ResNet50,
    'icnet': ICNet,
    'deeplab_v2': DeepLabV2,
    'hrnet': HRNet
}


def show_avai_models():
    """Displays available models.
    Examples::
        >>> from Models import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, input_height, input_width):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(
            name, avai_models))
    return __model_factory[name](num_classes,
                                 input_height=input_height,
                                 input_width=input_width)
