# coding=utf-8
# https://github.com/mdbloice/Augmentor
import Augmentor
import glob
import os
import random
import argparse
import itertools

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str,
                    default="./data/images_prepped_train")
parser.add_argument("--mask_path", type=str,
                    default='./data/annotations_prepped_train')
parser.add_argument("--augtrain_path", type=str,
                    default='./data/new_img')
parser.add_argument("--augmask_path", type=str,
                    default='./data/new_mask')

args = parser.parse_args()

train_path = args.train_path
mask_path = args.mask_path
augtrain_path = args.augtrain_path
augmask_path = args.augmask_path
sum = 0

def Init(train_path, mask_path):
    train_img = glob.glob(train_path + '/*.jpg') + glob.glob(train_path + '/*.png') + glob.glob(train_path + '/*.bmp')
    masks = glob.glob(mask_path + '/*.jpg') + glob.glob(mask_path + '/*.png') + glob.glob(mask_path + '/*.bmp')

    if len(train_img) != len(masks):
        print('train images can not match with train masks')
        return 0
    if not os.path.lexists(augtrain_path):
        os.mkdir(augtrain_path)
    if not os.path.lexists(augmask_path):
        os.mkdir(augmask_path)

    train_img.sort()
    masks.sort()
    assert len(train_img) == len(masks)
    zipped = itertools.cycle(zip(train_img, masks))
    cnt = 0
    for i in range(len(train_img)):
        path1, path2 = next(zipped)

        train_img_tmp_path = augtrain_path + '/' + str(i)
        if not os.path.lexists(train_img_tmp_path):
            os.mkdir(train_img_tmp_path)
        img = load_img(path1)
        x_t = img_to_array(img)
        img_tmp = array_to_img(x_t)
        img_tmp.save(train_img_tmp_path + '/' + str(i) + '.jpg')

        mask_img_tmp_path = augmask_path + '/' + str(i)
        if not os.path.lexists(mask_img_tmp_path):
            os.mkdir(mask_img_tmp_path)
        mask = load_img(path2)
        x_l = img_to_array(mask)
        mask_tmp = array_to_img(x_l)
        mask_tmp.save(mask_img_tmp_path + '/' + str(i) + '.jpg')
        print("%s folder has been created!" % str(i))
        cnt += 1
    return cnt


def doAugment(num):
    sum = 0
    for i in range(num):
        p = Augmentor.Pipeline(augtrain_path + '/' + str(i))
        p.ground_truth(augmask_path + '/' + str(i))
        p.rotate(probability=0.5, max_left_rotation=5,
                 max_right_rotation=5)  # 随机旋转
        p.flip_left_right(probability=0.5)  # 随机按概率左右翻转
        # 随机将一定比例面积的图形放大至全图
        p.zoom_random(probability=0.5, percentage_area=0.8)
        # p.flip_top_bottom(probability=0.6)  # 随机按概率随即上下翻转
        # p.random_distortion(probability=0.8, grid_width=10, grid_height=10, magnitude=20)  # 随机小块变形
        # p.resize(probability=1.0, width=480, height=360)
        print("\nNo.%s data is being augmented and %s data will be created" % (i, cnt))
        sum = sum + 5
        p.sample(5)
        print("Done")
    print("%s pairs of data has been created totally" % sum)

def Merge(num):

    if not os.path.lexists(augtrain_path + '/final_img'):
        os.mkdir(augtrain_path + '/final_img')
    if not os.path.lexists(augmask_path + '/final_mask'):
        os.mkdir(augmask_path + '/final_mask')
    final_img_path = augtrain_path + '/final_img'
    final_mask_path = augmask_path + '/final_mask'
    cnt = 0
    for i in range(num):
        path = augtrain_path + '/' + str(i) + '/output/'
        print(path)
        pairs = glob.glob(path + '*.jpg')
        pairs.sort()
        half = len(pairs) // 2
        print(half)
        for i in range(half):
            path1 = pairs[i]
            path2 = pairs[i + half]
            img = load_img(path1)
            mask = load_img(path2)
            img.save(final_img_path + '/' + str(cnt) + '.jpg')
            mask.save(final_mask_path + '/' + str(cnt) + '.jpg')
            cnt += 1
    print('%d pairs of data has been created totally' % cnt)


cnt = Init(train_path, mask_path)
# 默认开启了多线程
doAugment(cnt)
Merge(cnt)
