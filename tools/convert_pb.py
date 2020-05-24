#coding=utf-8
import sys
from keras.models import load_model
import os
import os.path as osp
import cv2
from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import argparse
import tensorflow as tf

import Models
from Models import build_model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.models import load_model

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    # 将会话状态冻结为已删除的计算图,创建一个新的计算图,其中变量节点由在会话中获取其当前值的常量替换.
    # session要冻结的TensorFlow会话,keep_var_names不应冻结的变量名列表,或者无冻结图中的所有变量
    # output_names相关图输出的名称,clear_devices从图中删除设备以获得更好的可移植性
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        # 从图中删除设备以获得更好的可移植性
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        # 用相同值的常量替换图中的所有变量
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--model_input_path", type=str, default="./output/model/unet40.hdf5")
parser.add_argument("--model_output_path", type=str, default="./output/model/")
parser.add_argument("--n_classes", type=int, default=11)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--input_height", type=int, default=512)
parser.add_argument("--input_width", type=int, default=512)

args = parser.parse_args()

# Start Convert
model_name = args.model_name
input_path = args.model_input_path
output_path = args.model_output_path
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width

if not os.path.isdir(output_path):
    os.mkdir(output_path)

K.set_learning_phase(0)

model = build_model(model_name, n_classes, input_height, input_width)
model.load_weights(input_path)

# 获取当前图
sess = K.get_session
# 冻结图
frozen_graph = freeze_session(sess, output_names=[model.output.op.name])

from tensorflow.python.framework import graph_io
graph_io.write_graph(frozen_graph, output_path, 'unet.pb', as_text=False)

print('Convert Pb Success!')


