#coding=utf-8
# https://github.com/mdbloice/Augmentor
import Augmentor
import glob
import os
import random
import argparse

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="F:/Keras-Semantic-Segmentation/dataset1/images_prepped_train")
parser.add_argument("--mask_path", type=str, default='F:/Keras-Semantic-Segmentation/dataset1/annotations_prepped_train')
parser.add_argument("--augtrain_path", type=str, default='F:/Keras-Semantic-Segmentation/dataset1/new_img')
parser.add_argument("--augmask_path", type=str, default='F:/Keras-Semantic-Segmentation/dataset1/new_mask')
parser.add_argument("--img_type", type=str, default="png")

args = parser.parse_args()

train_path = args.train_path
mask_path = args.mask_path
augtrain_path = args.augtrain_path
augmask_path = args.augmask_path
img_type = args.img_type

def Init(train_path, mask_path):
    train_img = glob.glob(train_path + '/*.' + img_type)
    masks = glob.glob(mask_path + '/*.' + img_type)

    if len(train_img) != len(masks):
        print('train images can not match with train masks')
        return 0
    cnt = 0
    for i in range(len(train_img)):
        train_img_tmp_path = train_path + '/' + str(i)
        if not os.path.lexists(train_img_tmp_path):
            os.mkdir(train_img_tmp_path)
        img = load_img(train_path + '/' + str(i) + '.' + img_type)
        x_t = img_to_array(img)
        img_tmp = array_to_img(x_t)
        img_tmp.save(train_img_tmp_path + '/' + str(i) + '.' + img_type)

        mask_img_tmp_path = mask_path + '/' + str(i)
        if not os.path.lexists(mask_img_tmp_path):
            os.mkdir(mask_img_tmp_path)
        mask = load_img(mask_path + '/' + str(i) + '.' + img_type)
        x_l = img_to_array(mask)
        mask_tmp = array_to_img(x_l)
        mask_tmp.save(mask_img_tmp_path + '/' + str(i) + '.' + img_type)
        print("%s folder has been created!" % str(i))
        cnt += 1
    return cnt

def doAugment(num):
    sum = 0
    for i in range(num):
        p = Augmentor.Pipeline(augtrain_path + '/' + str(i))
        p.ground_truth(augmask_path + '/' + str(i))
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)  # 随机旋转
        p.flip_left_right(probability=0.5)  # 随机按概率左右翻转
        p.zoom_random(probability=0.5, percentage_area=0.8)  # 随机将一定比例面积的图形放大至全图
        #p.flip_top_bottom(probability=0.6)  # 随机按概率随即上下翻转
        p.random_distortion(probability=0.8, grid_width=10, grid_height=10, magnitude=20)  # 随机小块变形
        #p.resize(probability=1.0, width=480, height=360)
        print("\nNo.%s data is being augmented and %s data will be created" % (i, cnt))
        sum = sum + 5
        p.sample(5)
        print("Done")
    print("%s pairs of data has been created totally" % sum)

cnt = Init(train_path, mask_path)
# 默认开启了多线程
doAugment(cnt)

