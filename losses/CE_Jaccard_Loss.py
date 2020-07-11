import tensorflow as tf
from keras.losses import categorical_crossentropy
from losses.Jaccard_loss import _Jaccard_Loss
SMOOTH = 1.

def _CE_Jaccard_Loss(gt, pr, cce_weight=1., smooth=SMOOTH, per_image=True):
    """Sum of categorical crossentropy and jaccard losses:
    
    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + jaccard_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch
    Returns:
        loss
    
    """
    cce = tf.mean(categorical_crossentropy(gt, pr))
    loss = cce_weight * cce + Jaccard_Loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss


def CE_JaccardLoss():
    return _CE_Jaccard_Loss
