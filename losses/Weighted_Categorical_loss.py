#coding=utf-8
import tensorflow as tf


def Weighted_Categorical_CrossEntropy_Loss(weights):
     """
     Keras多元交叉熵函数带权版本
    变量:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.variable(weights)
    def loss_(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip(y_pred, tf.epsilon(), 1 - tf.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss_
    

# 使用例子
# weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
# loss = weighted_categorical_crossentropy(weights)
# model.compile(loss=loss,optimizer='adam')