#coding=utf-8
import glob
import cv2
import numpy as np
import random
import datetime
import LoadBatches, ENet
from keras.models import load_model

n_classes = 2

images_path = "data/test/"
output_path = "data/output/"
save_weights_path = "weights/ex1"
input_height = 512
input_width = 512

modelFN = ENet.ENet

m = modelFN(input_height = input_height, input_width = input_width, n_classes = n_classes)
m.load_weights(save_weights_path + ".0")
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") \
         + glob.glob(images_path + "*.jpeg")
images.sort()

colors = [(0, 0, 0), (255, 255, 255)]

cnt = 0

# 开始统计程序运行时间
start_time = datetime.datetime.now()

for imgName in images:
    outName = output_path + str("%d.png"%cnt)
    #print(imgName)
    X = LoadBatches.getImageArr(imgName, input_width, input_height)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
    cnt += 1

end_time = datetime.datetime.now()
print("Total %d images, Average Time: " % cnt)
print((end_time - start_time).seconds * 1.0 / cnt)
