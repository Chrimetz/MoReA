"""ML Models

This module provides the basic interface of a machine learning model. 
It also includes implementations of the interface for Pickle and ONNX models.
Moreover a MLModelFactory loads models based on the description file,
validates them and returns a model type specific object based on IMLModel.

Classes:

	IMLModel
	ONNXModel
	PickleModel
	MLModelFactory

"""

import json
import os
import pickle
from pathlib import Path

import onnxruntime as rt

from abc import ABCMeta, abstractmethod

import numpy as np

class IMLModel(metaclass=ABCMeta):
	"""An interface to represent a basic machine learning model.

	Attributes:
		description (dict): JSON-parsed description of the model
		model_file_name (str): The path of the machine learning model

	Functions:
		load_model: Loads the given file and parses it as the required model type
		predict: Used to do inferencing of the model
	"""

	def __init__(self, description, model_file_name):
		"""Constructs all the necessary attributs for the object

		Args:
			description (dict): JSON-parsed description of the model
			model_file_name (str): The path of the machine learning model
		"""

		self.name = description["name"]
		self.description = description
		self.model_file_name = model_file_name

		self.load_model()

	@abstractmethod
	def load_model(self):
		"""Loads the model based on its type
		"""
		raise NotImplementedError

	@abstractmethod
	def predict(self):
		"""Runs the inferencing of the model
		"""
		raise NotImplementedError

	@abstractmethod
	def get_type(self):
		"""Returns the model type
		"""
		raise NotImplementedError

	def extract_features_flat(self, features):
		"""Extracts the input features of the model and returns a flattened list

		Args:
		    features (list): A list of the input features (possibly recursive
		    structure)
		
		Returns:
			list: A list of all input features (recursive structured flattened)

		"""
		result = []

		for i in features:
			if i["type"] != "list":
				result.append(i)
			else:
				result = result + self.extract_features_flat(i["features"])

		return result

	def get_input_features(self, flat = False):
		"""Extracts the input features of the model

		Args:
			flat (bool): A boolean to determine if the list shall be flattened.
		
		Returns:
			list: A list of all input features

		"""
		if flat:
			return self.extract_features_flat(self.description["input_features"])
		else:
			return self.description["input_features"]


class ONNXModel(IMLModel):
	"""An implementation of the IMLModel interface for ONNX models.

	Attributes:
		description (dict): JSON-parsed description of the model
		model_file_name (str): The path of the machine learning model

	Functions:
		load_model: Loads the given file and parses it as an onnx model
		predict: Used to do inferencing of the model
	"""

	def __init__(self, description, model_file_name):
		"""Calls the super constructor

		Args:
			description (dict): JSON-parsed description of the model
			model_file_name (str): The path of the machine learning model
		"""

		super().__init__(description, model_file_name)

	def load_model(self):
		"""Loads the model based on its type
		"""

		try:
			self.sess = rt.InferenceSession(self.model_file_name)
		except:
			raise SyntaxError("ONNX file is corrupt")

		return True

	def predict(self, processed_features):
		"""Runs the inferencing of the model
		"""

		pred_onx = self.sess.run(output_names=[o.name for o in self.sess.get_outputs()], input_feed=processed_features)

		result = {}
		for i in range(len(pred_onx)):
			result[self.sess.get_outputs()[i].name] = pred_onx[i].tolist()

		return {'result': result}

	def get_type(self):
		"""Returns the model type
		"""

		return "ONNX"


class PickleModel(IMLModel):
	"""An implementation of the IMLModel interface for Pickle models.

	Attributes:
		description (dict): JSON-parsed description of the model
		model_file_name (str): The path of the machine learning model

	Functions:
		load_model: Loads the given file and parses it as an pickle model
		predict: Used to do inferencing of the model
	"""

	def __init__(self, description, model_file_name):
		"""Calls the super constructor

		Args:
			description (dict): JSON-parsed description of the model
			model_file_name (str): The path of the machine learning model
		"""

		super().__init__(description, model_file_name)

	def load_model(self):
		"""Loads the model based on its type
		"""

		try:
			self.model = pickle.load(open(self.model_file_name, 'rb'))
		except Exception as e:
			raise SyntaxError("Pickle file is corrupt - " + str(e))

		return True

	def predict(self, processed_features):
		"""Runs the inferencing of the model
		"""

		model_input = np.asarray([list(x.values())[0][0] for x in processed_features["input"]])
		model_input = np.expand_dims(model_input, axis=0)
		
		prediction = self.model.predict(model_input.tolist())[0]
		
		return {'result': {'output': [x for x in prediction]}}

	def get_type(self):
		"""Returns the model type
		"""

		return "Pickle"

class MLModelFactory:
	"""A factory to load ml models, validate them and create model type
	specific model objects

	Attributes:
		SUPPORTED_TYPES (list): List of currently supported model types

	Functions:
	    validate_features: Validates if an input feature satisfies the model schema
	    validate_description: Validates if the model description satisfies the
	    	model schema
		camelCase: Convert text to camel case
	    load_from_directory: Loads all models from a directory and validates
	  		their description files
	"""

	SUPPORTED_TYPES = ["onnx", "pickle"]

	def validate_feature(self, feature, index):
		"""Validates if an input feature satisfies the model schema

		Args:
			feature (dict): JSON-parsed input feature of the model description
			index (int): Index of the input feature

		Returns:
			KeyError: If required attribute is not included
			ValueError: If a type of the input features is not correct
		"""

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

			if type(feature["features"]) != list:
				raise KeyError(
					"Features in input feature " + str(index) + " is not a list")

			j = -1
			for f in feature["features"]:
				j = j + 1
				try:
					self.validate_feature(f, j)
				except Exception as e:
					raise e

	def validate_description(self, description):
		"""Validates if a model description satisfies the model schema

		Args:
			description (dict): JSON-parsed dict of the model description

		Returns:
			KeyError: If required attribute is not included
			AssertionError: If the model can not be parsed by the 
				type specific framework
			ValueError: If a type of the input features is not correct
		"""

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

	def camelCase(self, st):
		"""Converts a string to camel case

		Args:
			st (str): String to be converted to camel case

		Returns:
			str: Camel cased input
		"""

		output = ''.join(x for x in st.title() if x.isalnum())
		return output[0:]

	def load_from_directory(self, path, logger):
		"""Loads all models from a directory and validates their description files

		Args:
			path (str): Path to locate the models
			logger (Logger): A logger to log the loading process

		Returns:
			list: List of all parsed models (based on IMLModel interface)
		"""

		logger.info("Loading ml models from " + path)

		if not os.path.exists(path):
			logger.error("Path " + path + "for ml models not found")

			return {}

		loaded_models = {}

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

						model_endpoint_name = parsed_description["name"]
						model_endpoint_name = model_endpoint_name.strip()
						model_endpoint_name = self.camelCase(model_endpoint_name)

						if not model_endpoint_name in loaded_models:
							if parsed_description["type"].lower() == "onnx":
								loaded_models[model_endpoint_name] = ONNXModel(parsed_description,
																		model_file_path)
							if parsed_description["type"].lower() == "pickle":
								loaded_models[model_endpoint_name] = PickleModel(parsed_description,
																			model_file_path)
						else:
							logger.error("Error while parsing " +
									 file + ": " + "name already exists " + 
									 model_endpoint_name)
							continue
						
					except Exception as e:
						logger.error("Error while parsing " +
									 file + ": " + str(e))
						continue
				except:
					logger.error(file + " is not a valid json file")

		for name, model in loaded_models.items():
			logger.info(
				"Loaded " + name + " (Type: "
				+ model.description['type'] + ")")

		return loaded_models