# 将Keras模型转换成Caffe模型

## 依赖
```
python3 
Pycaffe 需要自己编译，CPU/GPU版均可
```

## 转换命令

```
python convert2caffe.py xxx
```

- `--model_name` 字符串类型，代表测试时使用哪个模型，支持`enet`,`unet`,`segnet`,`fcn8`等多种模型，默认为`unet`。
- `--n_classes` 整型，代表分割图像中有几种类别的像素，默认为`2`。
- `--input_height`整型，代表要分割的图像需要`resize`的长，默认为`224`。
- `--input_width` 整型，代表要分割的图像需要`resize`的宽，默认为`224`。
- `--input_model` 字符串类型，代表模型的输入路径，如`../weights/unet.05.xxx.hdf5`。
- `--output_model` 字符串类型，代表转换后的caffe模型输出的文件夹路径，默认为`./unet.prototxt`，即当前目录。
- `--output_weights` 字符串类型，代表转换后的caffe模型名字，默认为`./unet.weights`。


## 已支持OP

- InputLayer
- Conv2D/Convolution2D
- Conv2DTranspose
- DepthwiseConv2D
- SeparableConv2D
- BatchNormalization
- Dense
- ReLU
- ReLU6
- LeakyReLU
- SoftMax
- SigMoid
- Cropping2D
- Concatenate
- Merge
- Add
- Flatten
- Reshape
- MaxPooling2D
- AveragePooling2D
- Dropout
- GlobalAveragePooling2D
- UpSampling2D
- ...

## 已支持网络
- VGG16
- SqueezeNet
- InceptionV3
- InceptionV4
- Xception V1
- UNet
- ...
