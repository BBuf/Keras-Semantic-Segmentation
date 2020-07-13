import os
import cv2
import tensorflow as tf
import keras.backend as K
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)

def mk_if_not_exits(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def cv2_letterbox_image(image, dst_size):
    ih, iw = image.shape[0:2]
    ew, eh = dst_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT)
    return new_img

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
 
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
 
    return flops.total_float_ops  # Prints the "flops" of the model.


# 使用callback，要稍微改一下Checkpoint()的使用方法
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_iou_score',
                 save_best_only=True, save_weights_only=True,
                 mode='max'):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, save_best_only, save_weights_only, mode)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)