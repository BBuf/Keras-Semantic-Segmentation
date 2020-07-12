#coding=utf-8
import tensorflow as tf

import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K

WEIGHTS = {
    'DRIVE': [0.91, 0.09],
}


def Weighted_Categorical_CrossEntropy_Loss(dataset):
    weights = WEIGHTS[dataset]
    weights = tf.Variable(weights,dtype=tf.float32)
    def loss_(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        return loss
    return loss_
    

# 使用例子
# weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
# loss = weighted_categorical_crossentropy(weights)
# model.compile(loss=loss,optimizer='adam')