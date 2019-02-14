#coding=utf-8
import numpy as np
import cv2
import glob #通配符
import itertools

def getImageArr(path, width, height):
    try:
        img = cv2.imread(path, 1)
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        return img

def getSegmentationArr(path, n_classes, width, height):
    seg_labels = np.zeros((height, width, n_classes))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]
        for c in range(n_classes):
            seg_labels[:, :, c] = (img == c).astype(int)
    except Exception as e:
        print(e)
    seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    return seg_labels

def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes,
                               input_height, input_width, output_height,
                               output_width):
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
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))
        yield np.array(X), np.array(Y)



