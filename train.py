#coding=utf-8
import argparse
from Models import ENet
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
import Models , LoadBatches

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str)
parser.add_argument("--train_images", type = str)
parser.add_argument("--train_annotations", type = str)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default = 512)
parser.add_argument("--input_width", type=int, default = 512)
parser.add_argument('--validate', action='store_false')
parser.add_argument("--val_images", type = str, default = "")
parser.add_argument("--val_annotations", type = str, default = "")
parser.add_argument("--epochs", type = int, default = 5)
parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--val_batch_size", type = int, default = 2)
parser.add_argument("--load_weights", type = str, default = "")
parser.add_argument("--model_name", type = str, default = "")
parser.add_argument("--optimizer_name", type = str, default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = {'enet':Models.ENet.ENet,
			'fcn8':Models.FCN8.FCN8,
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
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name,
      metrics=['accuracy'])


if len(load_weights) > 0:
	m.load_weights(load_weights)


print("Model output shape : ",  m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path,  train_batch_size,  n_classes, input_height, input_width, output_height, output_width)


if validate:
	G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path,  val_batch_size,  n_classes, input_height, input_width, output_height, output_width)

if not validate:
	for ep in range(epochs):
		m.fit_generator(G, 100, epochs=1)
		m.save_weights(save_weights_path + "." + str(ep))
		m.save(save_weights_path + ".model." + str(ep))
else:
	for ep in range(epochs):
		m.fit_generator(G, 100, validation_data=G2, validation_steps=20,  epochs=1)
		m.save_weights(save_weights_path + "." + str(ep))
		m.save(save_weights_path + ".model." + str(ep))