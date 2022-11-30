# Model Request API for EvoAI connection

In this project, a RESTFull API is developed to connect to the EvoAl framework. This project is currently work-in-progress state.

Additional ML-based Models can easily extend the API by adding the Models to the directory Models/ and a Model-Description file. This simple JSON template gives the API all the necessary details of your model. The models are automatically discovered and are available with their URLs. 

As mentioned, the project is in a work-in-progress state; thus, only sklearn models saved with pickels and .onnx models are supported currently. The model and the description file must have identical names, except for the ending. 

## Starting the API:
uvicorn main:app --reload

## API Endpoints

# All Models

The first endpoint provides a list of all available ML-based models along with its description and an individual URL to access each model

GET: localhost:8000/models

Example result:

`[
  {
    "name": "MNIST",
    "endpoint": "http://localhost:8000/models/Mnist",
    "description": "Predictive Model for recognizing handwritten digits, based on the MNIST dataset."
  },
  {
    "name": "XG Boost Regressor",
    "endpoint": "http://localhost:8000/models/XgBoostRegressor",
    "description": "Predictive Model for power estimation of CNNs on GPGPUs"
  }
]`

# Model details

Each model provides an individual endpoint for the details of the model including the possible parameters.

Example:

GET: http://localhost:8000/models/Mnist

Result:

`{
  "name": "MNIST",
  "details": "Predictive Model for recognizing handwritten digits, based on the MNIST dataset.",
  "outputs": "Probability-distribution of 10-classes for the digits 0-9.",
  "type": "onnx",
  "input_features": [
    {
      "name": "image",
      "shape": [
        28,
        28,
        1
      ],
      "type": "float"
    }
  ]
}`
