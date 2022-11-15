import pickle
from os import walk
import json

import onnxruntime as rt

class MLModel:
    def __init__(self, description, model_file_name):
        self.description = description
        self.model_file_name = model_file_name

    def load_model(self):
        raise NotImplementedError

    def predict(self):
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
    def __init__(self, description, model_file_name):
        super().__init__(description, model_file_name)

    def load_model(self):
        return super().load_model()

    def predict(self):
        return super().predict()

class MLModelFactory:

    def get(self):
        return ONNXModel('Test', 'models/mnist.onnx')

    def get_model(self, id: int):
        f = []    
        for (dirpath, dirnames, filenames) in walk("./Models/"):
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