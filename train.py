#coding=utf-8
import keras
import argparse
import data
import glob
import Models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from Models import ENet
from Models import FCN8
from Models import Segnet
from Models import Unet
from Models import PSPNet
from Models import ICNet
from Models import MobileNetUnet
from Models import MobileNetFCN8

parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str, default="data/train_image/")
parser.add_argument("--train_annotations", type = str, default="data/train_label/")
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--input_height", type=int, default = 224)
parser.add_argument("--input_width", type=int, default = 224)
parser.add_argument('--validate', type=bool, default=True)
parser.add_argument("--val_images", type = str, default = "data/val_image/")
parser.add_argument("--val_annotations", type = str, default = "data/val_label/")
parser.add_argument("--epochs", type = int, default = 50)
parser.add_argument("--train_batch_size", type = int, default = 4)
parser.add_argument("--val_batch_size", type = int, default = 4)
parser.add_argument("--train_save_path", type = str, default = "weights/unet")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--model_name", type = str, default = "unet")
parser.add_argument("--optimizer_name", type=str, default="sgd")

args = parser.parse_args()

# 再定义一些keras回调函数需要的参数
# patience：没有提升的轮次，即训练过程中最多容忍多少次没有提升
patience = 50
# log_file_path：日志保存的路径
log_file_path = 'weights/log.csv'

train_images = args.train_images
train_segs = args.train_annotations
train_batch_size = args.train_batch_size

validate = args.validate
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width

train_save_path = args.train_save_path
epochs = args.epochs
load_weights = args.resume

model_name = args.model_name
optimizer_name = args.optimizer_name

if validate:
	val_images = args.val_images
	val_segs = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = {'enet':Models.ENet.ENet,
			'fcn8':Models.FCN8.FCN8,
			'unet':Models.Unet.Unet,
			'segnet':Models.Segnet.Segnet,
			'pspnet':Models.PSPNet.PSPNet,
			'icnet':Models.ICNet.ICNet,
			'mobilenet_unet':Models.MobileNetUnet.MobileNetUnet,
			'mobilenet_fcn8':Models.MobileNetFCN8.MobileNetFCN8
			}
modelFN = modelFns[model_name]
model = modelFN(n_classes, input_height=input_height, input_width=input_width)

# 统计一下训练集/验证集样本数，确定每一个epoch需要训练的iter
images = glob.glob(train_images + "*.jpg") + glob.glob(train_images + "*.png") + glob.glob(train_images + "*.jpeg")
num_train = len(images)

images = glob.glob(val_images + "*.jpg") + glob.glob(val_images + "*.png") + glob.glob(val_images + "*.jpeg")
num_val = len(images)

# 模型回调函数
early_stop = EarlyStopping('loss', min_delta=0.1, patience=patience, verbose=1)
reduce_lr = ReduceLROnPlateau('loss', factor=0.01, patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = train_save_path + '.{epoch:02d}-{acc:2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='loss', save_best_only=True, save_weights_only=False)
call_backs = [model_checkpoint, csv_logger, early_stop, reduce_lr]

model.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])


if len(load_weights) > 0:
	model.load_weights(load_weights)
print("Model output shape : ",  model.output_shape)

output_height = model.outputHeight
output_width = model.outputWidth

train_ge = data.imageSegmentationGenerator(train_images, train_segs,  train_batch_size,
									 n_classes, input_height, input_width, output_height, output_width)

if validate:
	val_ge = data.imageSegmentationGenerator(val_images, val_segs,  val_batch_size,
											 n_classes, input_height, input_width, output_height, output_width)

if not validate:
	model.fit_generator(train_ge, epochs=epochs, callbacks=call_backs, steps_per_epoch=int(num_train / train_batch_size), verbose=1, shuffle=True)
else:

	model.fit_generator(train_ge, validation_data=val_ge,epochs=epochs, callbacks=call_backs, verbose=1, steps_per_epoch=int(num_train / train_batch_size),shuffle=True, validation_steps=int(num_val / val_batch_size))
