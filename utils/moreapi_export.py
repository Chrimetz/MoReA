from cnn_factory import cnn_factory
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as rt
import json

MODEL_NAMES = ["resnet50", "resnet101", "resnet152", "nasnetmobile", "nasnetlarge", "mobilenet", 
        "inceptionv3", "densenet201", "densenet169", "densenet121", "efficientnetb0", "efficientnetb1>
        "efficientnetb2", "efficientnetb3", "efficientnetb4", "efficientnetb5", "efficientnetb6",
        "efficientnetb7", "vgg16", "vgg19", "resnet50v2", "resnet101v2", "resnet152v2", "Xception",
        "InceptionResNetV2", "MobileNetV2", "alexnet", "alexNetModify1"]

factory = cnn_factory()

def export_model(name, path):
	if os.path.isfile(path + name + ".onnx"):
		print(name + " already exists")
		return False

	print("Export " + name)

	model = factory.get(name)

	input_shape = model.layers[0].input_shape
	if isinstance(input_shape, list):
		input_shape = input_shape[0]

	description = {}
	description["name"] = name
	description["details"] = "Details about " + name
	description["outputs"] = "Output description of " + name
	description["type"] = "onnx"
	description["input_features"] = [
		{
			"name": "input",
			"shape": [i if i != None else 1 for i in list(input_shape)],
			"type": "float32"
		}
	]

	with open(path + name + ".json", "w") as fp:
		json.dump(description, fp)

	spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
	output_path = path + name + ".onnx"

	tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
	

for n in MODEL_NAMES:
	export_model(n, "models/")
