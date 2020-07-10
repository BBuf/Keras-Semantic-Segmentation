import keras
import keras2onnx
import argparse
import onnx
import os, sys
from keras.models import load_model
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
parser.add_argument("--output_model", type=str, default="./xxx.onnx")
args = parser.parse_args()

n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
model_name = args.model_name
input_model_path = args.input_model
output_model_path = args.output_model

model = build_model(model_name,
                    n_classes,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(input_model_path)  
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, output_model_path)
