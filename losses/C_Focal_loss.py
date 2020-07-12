#coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K

NUM_SAMPLES = {
    'DRIVE': [6029585, 569615],
}

def focal_loss_multiclasses(dataset, gamma=2., e=0.1):
    # classes_num包含每一个类别的样本数量
    classes_num = NUM_SAMPLES[dataset]
    def focal_loss_fixed(y_pred, y_true):
        '''
        y_pred 是输出Tensor，形状类似[None, 10]，其中10是类别数
        y_true 是标签Tensor
        '''
        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        one_minus_p = array_ops.where(tf.greater(y_true,zeros), y_true - y_pred, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=y_pred.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(y_true, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)

        return fianal_loss
    return focal_loss_fixed