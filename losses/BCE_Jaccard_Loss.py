import tensorflow as tf
from keras.losses import binary_crossentropy
from losses.Jaccard_loss import _Jaccard_Loss
SMOOTH = 1.


def _BCE_Jaccard_Loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True):
    """Sum of binary crossentropy and jaccard losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + jaccard_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch (only for jaccard loss)
    Returns:
        loss
    
    """
    bce = tf.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + Jaccard_Loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss

def BCE_JaccardLoss():
    return _BCE_Jaccard_Loss
