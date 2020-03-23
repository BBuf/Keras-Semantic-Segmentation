from __future__ import absolute_import

from .ENet import *
from .FCN8 import *
from .ICNet import *
from .MobileNetFCN8 import *
from .MobileNetUnet import *
from .PSPNet import *
from .Segnet import *
from .SEUNet import *
from .Unet import *
from .scSEUnet import *

__model_factory = {
    'enet': ENet,
    'fcn8': FCN8,
    'unet': Unet,
    'segnet': Segnet,
    'pspnet': PSPNet,
    'icnet': ICNet,
    'mobilenet_unet': MobileNetUnet,
    'mobilenet_fcn8': MobileNetFCN8,
    'seunet': SEUnet,
    'scseunet': scSEUnet
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
