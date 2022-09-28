from typing import Union
from fastapi import FastAPI
from model_factory import model_factory
import json
from os import walk

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/models/")
def read_models():
    f = []    
    for (dirpath, dirnames, filenames) in walk("./Models/"):
        f.extend([filenames])        
        break

    models = []
    i = 1
    for file in filenames:
        models.append([file, "http://localhost:8000/model/"+str(i)])
        i = i + 1
    return models

@app.get("/model/{model_id}")
def read_model(model_id: int, data: Union[str, None] = None):
    facotry = model_factory()
    model = facotry.get(model_id)
    json_input = json.loads(data)

    #@Todo: Dynamisch nur die Attribute an die Modelle weitereichen die benötigt werden
    #Beispielsweise durch abgleich in einer JSON-Datei mit der Beschreibung der einzelnen
    #Modelle. Dadurch müsste zukünftig nur noch das Modell plus Beschreibungs JSON hinzugefügt werden
    #anschließend kann das neue Modell direkt genutzt werde.
    model_input = []
    for (key, value) in json_input.items():        
        model_input.append(value)

    predition = model.predict([model_input])
    
    return {"prediction": str(predition)}