#coding=utf-8
import keras
from keras.models import load_model
import tensorflow as tf
import os 
import os.path as osp
from keras import backend as K
import argparse
import onnx
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
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--graph_name", type=str, default="unet.pb")
args = parser.parse_args()

n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
model_name = args.model_name
input_model_path = args.input_model
output_dir = args.output_dir
graph_name = args.graph_name

model = build_model(model_name,
                    n_classes,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(input_model_path)  


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


# 获得当前图
sess = K.get_session()
# 冻结图
frozen_graph = freeze_session(sess, output_names=[model.output.op.name])

graph_io.write_graph(frozen_graph, output_dir, graph_name, as_text=False)

print('Convert Pb Success!')