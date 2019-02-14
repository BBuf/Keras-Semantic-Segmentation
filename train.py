#coding=utf-8
import argparse #argparse是python标准库里面用来处理命令行参数的库
import LoadBatches
import ENet
# 训练时的可选参数
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 5)
parser.add_argument("--save_weights_path", type = str)
parser.add_argument("--train_images", type = str)
parser.add_argument("--train_annotations", type = str)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default = 512)
parser.add_argument("--input_width", type=int, default = 512)
parser.add_argument('--validate', action='store_false')
parser.add_argument("--val_images", type = str, default = "")
parser.add_argument("--val_annotations", type = str, default = "")
parser.add_argument("--model_name", type = str, default = "")
parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--val_batch_size", type = int, default = 2)
parser.add_argument("--optimizer_name", type = str, default = "adadelta")

args = parser.parse_args()

# 将输入的可选参数进行赋值
train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
model_name = args.model_name
optimizer_name = args.optimizer_name
validate = args.validate
epochs = args.epochs

if validate:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations
    val_batch_size = args.val_batch_size

modelFN = ENet.ENet
m = modelFN(input_height = input_height, input_width = input_width, n_classes=n_classes)
m.compile(loss = 'categorical_crossentropy', optimizer = optimizer_name, metrics=['accuracy'])

print ("Model output shape : ", m.output_shape)
output_height = m.outputHeight
output_width = m.outputWidth
G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)
if validate:
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)
# 是否需要验证集
if not validate:
    for ep in range(epochs):
        m.fit_generator(G, 100, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".mode." + str(ep))
else:
    for ep in range(epochs):
        m.fit_generator(G, 100, validation_data=G2, validation_steps=20, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".mode." + str(ep))
