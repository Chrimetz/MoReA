"""
This module can load and validate the models. Moreover it provides the
different model types to do predictions.

Classes:

	MLModel
	ONNXModel
	PickleModel
	MLModelFactory
"""

import json
import os
import pickle
from pathlib import Path

import onnxruntime as rt


class MLModel:
	"""
	A class to represent a base machine learning model.

	Attributes:
					description
									JSON-file description of the model
					model_file_name
									The path of the machine learning model

	Functions:
					load_model
					predict
	"""

	def __init__(self, description, model_file_name):
		self.description = description
		self.model_file_name = model_file_name

	def load_model(self):
		"""
		Loads the model based on its type.
		"""
		raise NotImplementedError

	def predict(self):
		"""
		Runs the prediction of the model.
		"""
		raise NotImplementedError


class ONNXModel(MLModel):
	sess = None

	def __init__(self, description, model_file_name):
		super().__init__(description, model_file_name)

	def load_model(self):
		sess = rt.InferenceSession(self.model_file_name)

		return super().load_model()

	def predict(self):
		return super().predict()


class PickleModel(MLModel):
	model = None

	def __init__(self, description, model_file_name):
		super().__init__(description, model_file_name)

	def load_model(self):
		model = pickle.load(open(self.model_file_name, 'rb'))

		return super().load_model()

	def predict(self):
		return super().predict()


class MLModelFactory:

	SUPPORTED_TYPES = ["onnx", "pickle"]

	def get(self) -> MLModel:
		return ONNXModel('Test', 'models/mnist.onnx')

	def validate_feature(self, feature, index):
		if not "name" in feature:
			raise KeyError("Name not found in input feature " + str(index))
		elif not "type" in feature:
			raise KeyError("Type not found in input feature " + str(index))

		if feature["type"] != "list" and "shape" not in feature:
			raise KeyError("Shape not found in input feature " + str(index))
		elif feature["type"] != "list" and "shape" in feature:
			if isinstance(feature["shape"], list):
				if not len(feature["shape"]) > 0:
					raise (ValueError(
						"Shape needs to have at least one dimension in input feature " + str(index)))
			else:
				raise (ValueError(
					"Shape is not a list in input feature " + str(index)))

		if feature["type"] == "list":
			if not "features" in feature:
				raise KeyError(
					"Features not found in input feature " + str(index))

			j = -1
			for f in feature["features"]:
				j = j + 1
				try:
					self.validate_feature(f, j)
				except Exception as e:
					raise e

	def validate_description(self, description):
		for s in ["name", "details", "outputs", "input_features", "type"]:
			if not s in description:
				raise KeyError(s + " not found")

		if not description["type"].lower() in self.SUPPORTED_TYPES:
			raise AssertionError("Model type is not supported")

		i = -1
		for feature in description['input_features']:
			i = i + 1
			try:
				self.validate_feature(feature, i)
			except Exception as e:
				raise e

	def load_from_directory(self, path, logger):
		logger.info("Loading ml models from " + path)

		if not os.path.exists(path):
			logger.error("Path " + path + "for ml models not found")

			return []

		loaded_models = []

		json_files = []
		other_files = []
		for (dirpath, dirnames, filenames) in os.walk(path):
			for file in filenames:
				if file.endswith(".json"):
					if file not in json_files:
						json_files.append(file)
				else:
					if file not in other_files:
						other_files.append(file)

		for file in json_files:
			model_name = file.replace(".json", "")
			model_filename = ""
			for model_file in other_files:
				p = Path(model_file)
				extensions = "".join(p.suffixes)
				filename_wo_ext = str(p).replace(extensions, "")

				if model_name == filename_wo_ext:
					model_filename = model_file

			if model_filename == "":
				logger.error("No model for description " + file + " found")
			else:
				description_file_path = os.path.join(path, file)
				model_file_path = os.path.join(path, model_filename)

				try:
					parsed_description = json.load(
						open(description_file_path, "rb"))

					try:
						self.validate_description(parsed_description)

						if parsed_description["type"].lower() == "onnx":
							loaded_models.append(
								ONNXModel(parsed_description, model_file_path)
							)
						if parsed_description["type"].lower() == "pickle":
							loaded_models.append(
								PickleModel(parsed_description,
											model_file_path)
							)
					except Exception as e:
						logger.error("Error while parsing " +
									 file + ": " + str(e))
						return False
				except:
					logger.error(file + " is not a valid json file")

		for model in loaded_models:
			logger.info(
				"Loaded " + model.description['name'] + " (Type: " 
				+ model.description['type'] + ")")

		return loaded_models

	def get_model(self, id: int):
		f = []
		for (dirpath, dirnames, filenames) in os.walk("./Models/"):
			for file in filenames:
				print(file)
				if "json" not in str(file):
					f.append(file)
			break

		modelname = f[id-1]
		print("Models/"+modelname+".json")
		model_details = json.load(open("Models/"+modelname+".json", 'rb'))

		loaded_model = pickle.load(open("Models/"+modelname, 'rb'))
		return loaded_model, model_details
