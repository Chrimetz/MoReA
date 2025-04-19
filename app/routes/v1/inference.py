from fastapi import APIRouter, HTTPException
from app.schemas.ModelInput import ModelInput
from app.ml_models import IMLModel
from typing import Dict

router = APIRouter()

models: Dict[str, IMLModel] = {}

def set_models(models_dict: Dict[str, IMLModel]):
    """
    Sets the models dictionary to be used by the routes.

    Args:
        models_dict (Dict[str, IMLModel]): A dictionary of model names to model instances.
    """
    global models
    models = models_dict

@router.post("/models/{model_name}")
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
