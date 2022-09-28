import pickle
from os import walk

class model_factory:

    def get(self, id: int):
        f = []    
        for (dirpath, dirnames, filenames) in walk("./Models/"):
            f.extend(filenames)        
            break        
        filename = f[id-1]
        
        loaded_model = pickle.load(open("Models/"+filename, 'rb'))
        return loaded_model        