#coding=utf-8
import argparse
import glob
import itertools
import random

import cv2
import numpy as np
from keras.models import load_model

import data
import Models
from Models import build_model
from metrics import metrics
from losses.B_Focal_loss import focal_loss_binary
from losses.C_Focal_loss import focal_loss_multiclasses
from losses.Dice_loss import Dice_Loss
from losses.BCE_Dice_loss import BCE_Dice_Loss
from losses.CE_Dice_loss import CE_Dice_loss
from losses.Tversky_loss import Tversky_Loss
from losses.Focal_Tversky_loss import Focal_Tversky_Loss
from losses.Weighted_Categorical_loss import Weighted_Categorical_CrossEntropy_Loss
from losses.Generalized_Dice_loss import Generalized_Dice_Loss
from losses.Jaccard_loss import Jaccard_Loss
from losses.BCE_Jaccard_Loss import BCE_Jaccard_Loss
from losses.CE_Jaccard_Loss import Jaccard_Loss


EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--test_images", type=str, default="data/test/")
parser.add_argument("--output_path", type=str, default="data/output/")
parser.add_argument("--weights_path",
                    type=str,
                    default="weights/unet/unet.50-0.742981.hdf5")
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--input_height", type=int, default=320)
parser.add_argument("--input_width", type=int, default=640)
parser.add_argument("--resize_op", type=int, default=2)
parser.add_argument("--classes", type=int, default=11)
# streetscape(12)(320x640), helen_small(11)(512x512), bbufdataset
parser.add_argument("--mIOU", type=bool, default=True)
parser.add_argument("--val_images",
                    type=str,
                    default="data/helen_small/test_image/")
parser.add_argument("--val_annotations",
                    type=str,
                    default="data/helen_small/test_label/")
parser.add_argument("--image_init", type=str, default="sub_mean")

args = parser.parse_args()

images_path = args.test_images
output_path = args.output_path
save_weights_path = args.weights_path
model_name = args.model_name
input_height = args.input_height
input_width = args.input_width
resize_op = args.resize_op
n_class = args.classes
iou = args.mIOU
image_init = args.image_init

# color
random.seed(0)
colors = [(random.randint(0, 255), random.randint(0,
                                                  255), random.randint(0, 255))
          for _ in range(5000)]

# model
model = build_model(model_name,
                    n_class,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(save_weights_path)
output_height = model.outputHeight
output_width = model.outputWidth
print(output_height)
print(output_width)

# look up test images
images = glob.glob(images_path + "*.jpg") + glob.glob(
    images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images.sort()

cnt = 0

for imgName in images:
    outName = output_path + str("%d.jpg" % cnt)
    origin_img = cv2.imread(imgName, 1)
    origin_h = origin_img.shape[0]
    origin_w = origin_img.shape[1]
    X = data.getImage(imgName, input_width, input_height, image_init,
                      resize_op)
    pr = model.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_class)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_class):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(outName, seg_img)
    cnt += 1

print("Test Success!")

# mIOU
if iou:
    tp = np.zeros(n_class)
    fp = np.zeros(n_class)
    fn = np.zeros(n_class)
    n_pixels = np.zeros(n_class)
    images_path = args.val_images
    segs_path = args.val_annotations
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    images = glob.glob(images_path + "*.jpg") + glob.glob(
        images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(
        segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()
    assert len(images) == len(segmentations)
    zipped = itertools.cycle(zip(images, segmentations))
    for _ in range(len(images)):
        img_path, seg_path = next(zipped)
        # get origin h, w
        img = data.getImage(img_path, input_width, input_height, image_init,
                            resize_op)
        gt = data.getLable(seg_path, n_class, output_width, output_height,
                           resize_op)
        pr = model.predict(np.array([img]))[0]
        gt = gt.argmax(axis=-1)
        pr = pr.argmax(axis=-1)
        gt = gt.flatten()
        pr = pr.flatten()

        for c in range(n_class):
            tp[c] += np.sum((pr == c) * (gt == c))
            fp[c] += np.sum((pr == c) * (gt != c))
            fn[c] += np.sum((pr != c) * (gt == c))
            n_pixels[c] += np.sum(gt == c)
    print(tp)
    cl_wise_score = tp / (tp + fp + fn + EPS)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IOU = np.mean(cl_wise_score)
    print("frequency_weighted_IU: ", frequency_weighted_IU)
    print("mean IOU: ", mean_IOU)
    print("class_wise_IOU:", cl_wise_score)
