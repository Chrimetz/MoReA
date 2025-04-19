import requests

API_URL = "http://127.0.0.1"
API_PORT = 8000
API_ROOT_PATH = "v1/models"

def get_all_models():
    r = requests.get(url = API_URL + ":" + str(API_PORT) + "/" + API_ROOT_PATH)

    return r.json()

def get_model_details(model_name):
    r = requests.get(url = API_URL + ":" + str(API_PORT) + "/" + API_ROOT_PATH + "/" + model_name)

    return r.json()

def request_model_inferencing(model_name, features):
    data = {"features": features}

    r = requests.post(url = API_URL + ":" + str(API_PORT) + "/" + API_ROOT_PATH + "/" + model_name, json=data)

    return r.json()