#coding=utf-8
import tensorflow as tf
import keras
from keras.layers import Flatten
from keras.losses import binary_crossentropy

def Tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = Flatten()(y_true)
    y_pred_pos = Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def _Tversky_Loss(y_true, y_pred):
    return 1 - Tversky(y_true,y_pred)

def TverskyLoss():
    return _Tversky_Loss

# model.compile(optimizer=adam, loss=Tversky_loss,metrics=[dsc, tversky])