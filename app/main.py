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

from typing import Union, Dict
from fastapi import FastAPI
from os import path
import logging
import uvicorn
import argparse
from app.routes.v1 import model, inference
from app.ml_models import MLModelFactory

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


models = None


def main():
	print("Starting MoReA")
    
	parser = argparse.ArgumentParser(description="MoReA API")
	parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
	parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
	parser.add_argument("--log-level", type=str, default="info", help="Log level to use")
	parser.add_argument("--models-dir", type=str, help="Directory to load models from")
	args = parser.parse_args()

	if args.models_dir:
		if not path.exists(args.models_dir):
			raise ValueError("Models directory does not exist")
		else:            
			global models
			models = MLModelFactory().load_from_directory(args.models_dir, logger)

	model.set_models(models)
	inference.set_models(models)

	print(f"Starting MoReA API on {args.host}:{args.port} with log level {args.log_level}")
	# Explicitly reference the app instance
	uvicorn.run("app.main:app", host=args.host, port=args.port, log_level=args.log_level)

# Include the model metadata routes
app.include_router(model.router, prefix="/v1")
app.include_router(inference.router, prefix="/v1")

