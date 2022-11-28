from typing import Union
from fastapi import FastAPI, Request, HTTPException
from ml_models import MLModelFactory, IMLModel
import json
from os import walk
from os import path
import numpy as np
import logging

import sys

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

models = MLModelFactory().load_from_directory("./models", logger)


@app.get("/")
def read_root():
	return {"Hello": "World"}


@app.get("/models/")
def get_all_models(request: Request):
	result = []

	for name, model in models.items():
		model: IMLModel = model
		result.append({"name": model.name,
					   "endpoint": str(request.url)+name,
					   "description": model.description["details"]})

	return result


@app.get("/models/{model_name}")
def get_model(model_name: str):
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
