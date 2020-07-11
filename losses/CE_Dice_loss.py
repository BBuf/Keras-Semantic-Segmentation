#coding=utf-8
import tensorflow as tf
import keras
from keras.layers import Flatten
from keras.losses import categorical_crossentropy

def Dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + 
                    tf.reduce_sum(y_pred_f) + smooth)
    return score

def Dice_Loss(y_true, y_pred):
    loss = 1. - Dice_coeff(y_true, y_pred)
    return loss

def _CE_Dice_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + Dice_Loss(y_true, y_pred)
    return loss

def CE_DiceLoss():
    return _CE_Dice_loss



