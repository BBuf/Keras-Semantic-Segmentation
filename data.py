#coding=utf-8
import numpy as np
import cv2
import glob
import itertools

def getImage(path, width, height, imgNorm="sub_mean"):
    img = cv2.imread(path, 1)
    # resize到网络输入大小
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    if imgNorm == "sub_and_divide":
        img = img / 127.5 - 1
    elif imgNorm == "sub_mean":
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    else:
        img = img / 255.0
    
    return img

def getLable(path, n_classes, width, height):
    seg_labels = np.zeros((height, width, n_classes))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]
    for c in range(n_classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    return seg_labels

def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes,
                               input_height, input_width, output_height,
                               output_width, image_init):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()
    assert len(images) == len(segmentations)
    zipped = itertools.cycle(zip(images, segmentations))
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImage(im, input_width, input_height, image_init))
            Y.append(getLable(seg, n_classes, output_width, output_height))
        yield np.array(X), np.array(Y)


