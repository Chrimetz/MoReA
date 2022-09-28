from typing import Union
from fastapi import FastAPI
from model_factory import model_factory

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/model/{model_id}")
def read_model(model_id: int, data: Union[str, None] = None):
    facotry = model_factory()
    model = facotry.get(model_id)
    
    
    
    return {"model_id": model, "data": data}