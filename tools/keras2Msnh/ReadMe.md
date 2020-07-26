## Keras to Msnhnet
---
**Msnhnet Link**
- [https://github.com/msnh2012/Msnhnet](https://github.com/msnh2012/Msnhnet)
  
**Requirements**
- Keras 2 and tensorflow 1.x
  
**How to use.**
```
keras2Msnh(model,"resnet50.msnhnet", "resnet50.msnhbin")
```
**Supported Layers**

- InputLayer
- Conv2D/Convolution2D
- DepthwiseConv2D
- MaxPooling2D
- AveragePooling2D
- BatchNormalization
- LeakyReLU
- Activation(relu, relu6, leakyReLU, sigmoid, linear)
- UpSampling2D
- Concatenate/Merge
- Add
- ZeroPadding2D
- GlobalAveragePooling2D
- softmax
- Dense

**Pred Model**
- 链接：https://pan.baidu.com/s/1K3jyKesFJuj4IgLOgpmLxw 提取码：sbbn

**TEST**
- 1.Build and install Msnhnet with cmake.</br>
**Detial**: https://github.com/msnh2012/Msnhnet/blob/master/ReadMe.md
- 2.Build "project" with cmake.
- 3.Download pred model and extract it.(7zip is required) **eg**. D:/
- 4.Open terminal. ```unet /your/extract/pred model/path``` **eg**.```unet D:/models/```
- 5.Also you can use msnhnet viewer to view msnhnet.
![img](https://github.com/msnh2012/Msnhnet/blob/master/readme_imgs/msnhnetviewer.png)
- 6.Enjoy it! :D
