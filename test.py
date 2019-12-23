#coding=utf-8
import glob
import cv2
import numpy as np
import random
import data
import argparse
import datetime
import Models
from Models import ENet
from Models import FCN8
from Models import Segnet
from Models import Unet
from Models import PSPNet
from Models import ICNet
from Models import MobileNetUnet
from Models import MobileNetFCN8
from keras.models import load_model

EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--test_images", type = str, default="data/test/")
parser.add_argument("--output_path", type=str, default="data/output/")
parser.add_argument("--weights_path", type=str, default="weights/unet.18-0.856895.hdf5")
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--classes", type=int, default=2)

args = parser.parse_args()

images_path = args.test_images
output_path = args.output_path
save_weights_path = args.weights_path
model_name = args.model_name
input_height = args.input_height
input_width = args.input_width
n_class = args.classes

modelFns = {'fcn8':Models.FCN8.FCN8,
			'unet':Models.Unet.Unet,
			'enet':Models.ENet.ENet,
			'segnet':Models.Segnet.Segnet,
			'pspnet':Models.PSPNet.PSPNet,
			'icnet':Models.ICNet.ICNet,
			'mobilenet_unet':Models.MobileNetUnet.MobileNetUnet,
			'mobilenet_fcn8':Models.MobileNetFCN8.MobileNetFCN8
			}

modelFN = modelFns[model_name]

model = modelFN(n_class, input_height=input_height, input_width=input_width)
model.load_weights(save_weights_path)
output_height = model.outputHeight
output_width = model.outputWidth
print(output_height)
print(output_width)

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images.sort()


colors = [(0, 0, 0), (255, 0, 255), (255, 215, 0), (0, 255, 255), (255, 125, 0)]
cnt = 0

# 开始统计程序运行时间
start_time = datetime.datetime.now()

for imgName in images:
    outName = output_path + str("%d.jpg" % cnt)
    X = data.getImage(imgName, input_width, input_height)
    pr = model.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_class)).argmax(axis = 2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_class):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
    cnt += 1

end_time = datetime.datetime.now()
print("Total %d images, Average Time: " % cnt, end='')
print((end_time - start_time).seconds * 1.0 / cnt)
