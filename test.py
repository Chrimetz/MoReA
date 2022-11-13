import pickle
import numpy as np

model = pickle.load(open('Models/RandomForestRegression', 'rb'))

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 20]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print(model.predict([np.array([1,2,3,4,5,5,5,5,5,5,6,5,5,5,5,5,5,5,5,8], dtype=float)]))

inputs = {}
for inp in onx.graph.input:
    shape = str(inp.type.tensor_type.shape.dim)
    inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]

print(inputs)

# Compute the prediction with ONNX Runtime
import onnxruntime as rt
sess = rt.InferenceSession("rf_iris.onnx")
input_name = sess.get_inputs()[0].name
print(input_name)
label_name = sess.get_outputs()[0].name
print(label_name)
pred_onx = sess.run([label_name], {input_name: [np.array([1,2,3,4,5,5,5,5,5,5,6,5,5,5,5,5,5,5,5,8], dtype=float)]})[0]

print(pred_onx)