import tensorflow as tf

SMOOTH = 1.

def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数：
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), 
        if ``None`` prediction prediction will not be round
    返回：
        IoU/Jaccard score in range [0, 1]
    '''
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]
        
    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    intersection = tf.sum(gt * pr, axis=axes)
    union = tf.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = tf.mean(iou, axis=0)

    # weighted mean per class
    iou = tf.mean(iou * class_weights)

    return iou

jaccard_score = iou_score

def _Jaccard_Loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    """Jaccard loss function for imbalanced datasets:
    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
    Returns:
        Jaccard loss in range [0, 1]
    """
    return 1. - Jaccard_Loss(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)

def JaccardLoss():
    return _Jaccard_Loss