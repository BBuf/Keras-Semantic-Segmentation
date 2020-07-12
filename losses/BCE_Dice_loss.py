#coding=utf-8
from keras.losses import binary_crossentropy

from losses.Dice_loss import _dice_coef_loss


def _BCE_Dice_Loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + _dice_coef_loss(y_true, y_pred)
    return loss

def BCE_DiceLoss():
    return _BCE_Dice_Loss



