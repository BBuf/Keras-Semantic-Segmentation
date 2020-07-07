#coding=utf-8
import glob
import cv2

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,    # 随机水平翻转
    VerticalFlip,      # 随机垂直翻转
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,    # 随机90度旋转
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,   # 随机尺寸裁剪并缩放回原始大小
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)

def data_num(train_path, mask_path):
    train_img = glob.glob(train_path)
    masks = glob.glob(mask_path)
    return train_img, masks

def mask_aug():
    
    aug = Compose([VerticalFlip(p=0.5),
                       RandomRotate90(p=0.5),
                       HorizontalFlip(p=0.5),
                       RandomSizedCrop(min_max_height=(128, 512), height=384, width=384, p=0.5)])

    return aug

def main():

    train_path = (r"./data/data-2/train_image/*.jpg")  # 输入 img 地址
    mask_path = (r"./data/data-2/train_label/*.png")  # 输入 mask 地址
    augtrain_path = (r"./data/data-2/new_image")  # 输入增强img存放地址
    augmask_path = (r"./data/data-2/new_label")  # 输入增强mask存放地址
    num = 3  # 输入增强图像增强的张数。
    aug = mask_aug()

    train_img, masks = data_num(train_path, mask_path)

    for data in range(len(train_img)):
        for i in range(num):
            image = cv2.imread(train_img[data])
            mask = cv2.imread(masks[data])
            augmented = aug(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            cv2.imwrite("./data/data-2/new_image/aug{}_{}.jpg".format(data, i), aug_image)
            cv2.imwrite("./data/data-2/new_label/aug{}_{}.png".format(data, i), aug_mask)
            print(data)
# cv2.imshow("x",aug_image)
# cv2.imshow("y",aug_mask)
# cv2.waitKey(0)
if __name__ == "__main__":
    main()

