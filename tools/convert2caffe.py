import keras2caffe
import keras
from keras.models import load_model
import tensorflow as tf
import os 
import os.path as osp
from keras import backend as K
import argparse
import os, sys
from keras.models import load_model
from tensorflow.python.framework import graph_util,graph_io
from tensorflow.python.framework.graph_util import convert_variables_to_constants

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import Models
from Models import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--input_model", type=str, default="../weights/xxx.hd55")
parser.add_argument("--output_model", type=str, default="./unet.prototxt")
parser.add_argument("--output_weight", type=str, default="./unet.caffemodel")
args = parser.parse_args()

n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
model_name = args.model_name
input_model_path = args.input_model
output_model = args.output_weight
output_weight = args.output_model

model = build_model(model_name,
                    n_classes,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(input_model_path)  

keras2caffe.convert(model, output_weight, output_model)
