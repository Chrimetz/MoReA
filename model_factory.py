import pickle
from os import walk
import json

class model_factory:

    def get(self, id: int):
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