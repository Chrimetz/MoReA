from fastapi import APIRouter, HTTPException, Request
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


@router.get("/models", summary="Get all available models", tags=["Model Metadata"])
def get_all_models(request: Request):
    """
    Returns a list of all available models with their endpoint URL.

    Args:
        request (Request): HTTP request object.

    Returns:
        list: A list of all available models with name, endpoint, and description.
    """
    result = []

    for name, model in models.items():
        model: IMLModel = model
        result.append({
            "name": model.name,
            "endpoint": str(request.url) + name,
            "description": model.description["details"]
        })

    return result


@router.get("/models/{model_name}", summary="Get model details", tags=["Model Metadata"])
def get_model(model_name: str):
    """
    Returns details of a specific model.

    Args:
        model_name (str): A unique name to identify the requested model.

    Returns:
        dict: A dictionary of the model details and the input features.
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