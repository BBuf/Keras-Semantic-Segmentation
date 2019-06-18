#coding=utf-8
#coding=utf-8
#定义参数
import glob
import cv2
import numpy as np
import random
import datetime
import Models, LoadBatches
from Models import FCN8
from Models import FCN32
from Models import Segnet
from Models import MiniUnet
from Models import Unet
from Models import PSPNet
from Models import ICNet
from Models import VGGSegnet
from Models import VGGUnet
from Models import VGGFCN8
from Models import VGGFCN32
from Models import ResNet50_Segnet
from Models import ResNet50_Unet
from Models import ResNet50_FCN8
from Models import ResNet50_FCN32
from Models import MobileNetUnet
from Models import MobileNetFCN8
from Models import MobileNetFCN32
from Models import MobileNetSegnet
from keras.models import load_model

EPS = 1e-12
n_classes = 2
model_name = "vgg_unet"
images_path = "data/test/"
output_path = "data/output/"
save_weights_path = "weights/ex1"
input_height = 512
input_width = 512
epoch_num = 0
modelFns = {'fcn8':Models.FCN8.FCN8,
			'fcn32':Models.FCN32.FCN32,
			'unet':Models.Unet.Unet,
			'mini_unet':Models.MiniUnet.MiniUnet,
			'segnet':Models.Segnet.Segnet,
			'pspnet':Models.PSPNet.PSPNet,
			'icnet':Models.ICNet.ICNet,
			'icnet_bn':Models.ICNet.ICNet_BN,
			'vgg_segnet':Models.VGGSegnet.VGGSegnet,
			'vgg_unet':Models.VGGUnet.VGGUnet,
			'vgg_fcn8':Models.VGGFCN8.VGG_FCN8,
			'vgg_fcn32':Models.VGGFCN32.VGG_FCN32,
			'resnet50_segnet':Models.ResNet50_Segnet.Resnet_Segnet,
			'resnet50_unet':Models.ResNet50_Unet.Resnet_Unet,
			'resnet50_fcn8':Models.ResNet50_FCN8.Resnet_FCN8,
			'resnet50_fcn32':Models.ResNet50_FCN32.Resnet_FCN32,
			'mobilenet_unet':Models.MobileNetUnet.MobileNetUnet,
			'mobilenet_fcn8':Models.MobileNetFCN8.MobileNetFCN8,
			'mobilenet_fcn32':Models.MobileNetFCN32.MobileNetFCN32,
			'mobilenet_segnet':Models.MobileNetSegnet.MobileNetSegnet
			}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(save_weights_path + ".0")
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images.sort()

colors = [(0, 0, 0), (255, 255, 255)]
cnt = 0

def get_IOU(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl) * (pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection) / (union + EPS)
        class_wise[cl] = iou
    return class_wise

# 开始统计程序运行时间
start_time = datetime.datetime.now()

for imgName in images:
    outName = output_path + str("%d.png"%cnt)
    #print(imgName)
    X = LoadBatches.getImageArr(imgName, input_width, input_height)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis = 2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
    cnt += 1

end_time = datetime.datetime.now()

print("Total %d images, Average Time: " % cnt, end='')
print((end_time - start_time).seconds * 1.0 / cnt)