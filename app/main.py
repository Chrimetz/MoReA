"""Main

This is the main class of MoreaPI. It loads all models, starts the REST API
and provides the endpoints.

Attributes:
	app (FastAPI): REST API
	logger (Logger): Logger of FastAPI
	dir_path (str): Current working directory
	models (list): List of all available models in dir_path/models

Functions:
	get_all_models: Returns a list of all available models
	get_model: Returns details of a specific model based on its name
	request_model: Runs the inferencing of a model based on its name and input features

"""

from typing import Union, Dict, Optional
from fastapi import FastAPI, Request, HTTPException
from .ml_models import MLModelFactory, IMLModel
import json
from os import walk
from os import path
import os
import numpy as np
import logging

import sys

from pydantic import BaseModel

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
models = MLModelFactory().load_from_directory(os.path.join(dir_path, "models"), logger)

class ModelInput(BaseModel):
    features: Dict[str, Union[str, list, float, int]]

@app.get("/models/")
def get_all_models(request: Request):
	"""Get a list of all available models with their endpoint url

	Args:
	    request (Request): HTTP-GET request
		
	Returns:
		list: A list of all available models with name, endpoint and description

	"""
	result = []

	for name, model in models.items():
		model: IMLModel = model
		result.append({"name": model.name,
					   "endpoint": str(request.url)+name,
					   "description": model.description["details"]})

	return result


@app.get("/models/{model_name}")
def get_model(model_name: str):
	"""Get details of a specific model

	Args:
	    model_name (str): A unique name to identify the requested model
		
	Returns:
		dict: A dict of the model details and the input features

	"""
	if model_name not in models:
		raise HTTPException(status_code=404, detail="Model not found") 

	model = models[model_name]

	result = {
		"name": model.name,
		"details": model.description["details"],
		"outputs": model.description["outputs"],
		"type": model.description["type"],
		"input_features": model.get_input_features()
	}

	return result

@app.post("/models/{model_name}")
def request_model(model_name: str, input: ModelInput):
	"""Run the inferencing of a model based on input features

	Args:
	    model_name (str): A unique name to identify the requested model
		input (ModelInput): The input features for the inferencing

	Returns:
		dict: A dict of the model inferencing results

	"""
	if model_name not in models:
		raise HTTPException(status_code=404, detail="Model not found") 

	model = models[model_name]
	features = input.features

	processed_features = {}

	for f in model.get_input_features(True):
		if not f["name"] in features.keys():
			raise HTTPException(status_code=400, detail="Feature '" + f["name"] + "' not found")
		else:
			pf = features[f["name"]]
			if f["shape"] == [1]:
				try:
					if f["type"].lower() == "int":
						value = np.array([int(pf)])
					elif f["type"].lower() == "float":
						value = np.array([float(pf)])
					elif f["type"].lower() == "float32":
						value = np.array([float(pf)], dtype=np.float32)
					elif f["type"].lower() == "string" or f["type"].lower() == "str":
						value = np.array([str(pf)])
				except ValueError:
					raise HTTPException(status_code=400, detail="Feature '" + f["name"] + "' not parseable as type '" + f["type"] + "'") 
				except TypeError:
					raise HTTPException(status_code=400, detail="Feature '" + f["name"] + "' not parseable as type '" + f["type"] + "'") 
			else:
				try:
					if f["type"].lower() == "int":
						value = np.array(pf, dtype=int)
					elif f["type"].lower() == "float":
						value = np.array(pf, dtype=float)
					elif f["type"].lower() == "float32":
						value = np.array(pf, dtype=np.float32)
					elif f["type"].lower() == "string" or f["type"].lower() == "str":
						value = np.array(pf, dtype=str)
				except ValueError as e:
					raise HTTPException(status_code=400, detail="Feature '" + f["name"] + "': " + str(e))
		
			shape = []
			for i in range(0, len(value.shape)):
				shape.append(value.shape[i])
			
			if shape != f["shape"]:
				raise HTTPException(status_code=400, detail="Feature '" + f["name"] + "': Shape [" + ','.join(str(e) for e in shape) + "] not matching [" + ','.join(str(e) for e in f["shape"]) + "]")

			processed_features[f["name"]] = value

	final_features = {}
	for f in model.get_input_features(False):
		if f["type"].lower() != "list":
			final_features[f["name"]] = processed_features[f["name"]]
		else:
			l = []
			for x in [y["name"] for y in f["features"]]:
				l.append({x: processed_features[x]})
			final_features[f["name"]] = l
			
	try:
		prediction = model.predict(final_features)

		return prediction
	except Exception as e:
		raise HTTPException(status_code=400, detail="Prediction failed: " + str(e))
