# 使用Keras实现ENet网络
- ENet.py 使用Keras搭建Enet的网络结构，上采样直接使用了UpSampling
- LoadBatches.py 加载1个batch的原始图片和分割标签图片，返回一个generator
- train.py 使用ENet训练并保存模型
- test.py 测试模型在新图片上得到的预测结果和单张图片前向推理需要的时间

# 环境配置
- python 3.6
- keras 2.0

# 数据集制作
使用labelImg将mask画出来生成json文件之后，再把json文件通过labelme转为png图片，可以参考下这个博客：https://blog.csdn.net/u010103202/article/details/81635436
# 训练
使用命令：
```python
python  train.py --save_weights_path=weights/ex1 
--train_images="data/dataset1/images_prepped_train/" 
--train_annotations="data/dataset1/annotations_prepped_train/" 
--val_images="data/dataset1/images_prepped_test/" 
--val_annotations="data/dataset1/annotations_prepped_test/" 
--n_classes=2 --input_height=512 --input_width=512
```
训练完成后模型会保存在weights文件夹下面，我们可以利用保存的模型来进行测试工作。

# 测试
```python
python test.py
```

# benchmark
在我的i7 cpu上单张图片前向推理的速度为0.55s/张，并且在我自己的数据集上得到的分割效果和SegNet一致。

# 注意的点
- 在test.py中一些参数按照我的数据写死了，训练你自己的数据需要重新对应修改一下。
- 如果需要引入Segnet中通过保留编码网络中最大池化过程中最大值的索引，可以自行修改，我如果修改完成也会更新。
- 论文讲解博客：https://blog.csdn.net/just_sort/article/details/87176238
- 开源不易，觉得有用请点个Star，有问题请Issue留言。
