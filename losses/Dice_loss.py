#coding=utf-8
import tensorflow as tf
import keras
from keras.layers import Flatten

def Dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + 
                    tf.reduce_sum(y_pred_f) + smooth)
    return score

def _dice_Loss(y_true, y_pred):
    loss = 1. - Dice_coeff(y_true, y_pred)
    return loss

def DiceLoss():
    return _dice_Loss





