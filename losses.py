#coding=utf-8
#https://blog.csdn.net/m0_37477175/article/details/83004746
from keras import backend as K

# model.compile(optimizer=optimizer, loss=[focal_loss(alpha=.25, gamma=2)])
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_(y_true, y_pred):
        pt_1 = K.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = K.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha))
    return focal_loss_



