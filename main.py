from typing import Union
from fastapi import FastAPI
from ml_models import MLModelFactory
import json
from os import walk
from os import path
import numpy as np
import logging

import sys

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MLModelFactory().load_from_directory("./models", logger)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/models/")
def read_models():
    files = []    
    for (dirpath, dirnames, filenames) in walk("./Models/"):
        for file in filenames:
            if "json" not in str(file):
                files.extend([file])        
        break

    models = []
    i = 1
    for file in files:
        models.append([file, "http://localhost:8000/model/"+str(i)])
        i = i + 1
    return models

@app.get("/model/{model_id}")
def read_model(model_id: int, data: Union[str, None] = None):
    facotry = model_factory()
    model, model_detail = facotry.get(model_id)
    json_input = json.loads(data)

    model_inputs = model_detail["Input_features"]

    model_input = []
    for (key, value) in json_input.items():     
        for (input_key, input_value) in model_inputs.items(): 
            if key == input_key and input_value == "True":  
                model_input.append(value)

    predition = model.predict([np.array(model_input, dtype=float)])
    
    return {"prediction": str(predition)}