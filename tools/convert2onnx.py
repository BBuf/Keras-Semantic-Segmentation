import keras
import keras2onnx
import onnx
from keras.models import load_model
model = load_model('/root/notebook/model/river_model5.h5')  
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = '/root/notebook/model/model.onnx'
onnx.save_model(onnx_model, temp_model_file)