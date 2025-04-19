# MoReA - Model Request API: Deploy Machine Learning Models fast

<div align="center">
  <img src="./morealogo.png" alt="MoReA Logo" width="25%">
</div>

In this project, a RESTful API is developed to deploy machine learning models quick and in a no-code environment for experimental and simple production environments. This project is currently in a work-in-progress state.

Additional ML-based models can easily extend the API by adding the models to the directory that is specified as parameter during startup of the application. However, currently a model-description file is required. This simple JSON template gives the API all the necessary details of your model. The models are automatically discovered and are available with their URLs.

As mentioned, the project is in a work-in-progress state; thus, only sklearn models saved with pickles and `.onnx` models are supported currently. The model and the description file must have identical names, except for the file extension.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/morea.git
   cd morea
   ```
2. Install MoReA:
   ```bash
   pip install .
   ```

---

## Starting the API

### From Command Line
After installation, you can start the application directly from the command line using the `MoReA` command:

```bash
morea --host 127.0.0.1 --port 8000 --log-level debug --models-dir ./models
```

### Available Parameters
- `--host`: Specify the host address to bind the server to (default: `0.0.0.0`).
- `--port`: Specify the port number to bind the server to (default: `8000`).
- `--log-level`: Set the log level for the server. Available options are:
  - `critical`
  - `error`
  - `warning`
  - `info` (default)
  - `debug`
- `--models-dir`: Specify the directory where all models and model descriptions are located. This parameter must be specified at start and does not have a default value.

For example, to run the server on a custom host and port with debug logging:
```bash
morea --host 192.168.1.100 --port 8080 --log-level debug
```

---
## API Endpoints

### All Models

The first endpoint provides a list of all available ML-based models along with its description and an individual URL to access each model.

#### GET: `localhost:8000/models`

##### Example Result:
```json
[
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
]
```

### Model Details

Each model provides an individual endpoint for the details of the model, including the possible parameters.

#### Example:

##### GET: `http://localhost:8000/models/Mnist`

##### Result:
```json
{
  "name": "MNIST",
  "details": "Predictive Model for recognizing handwritten digits, based on the MNIST dataset.",
  "outputs": "Probability-distribution of 10-classes for the digits 0-9.",
  "type": "onnx",
  "input_features": [
    {
      "name": "image",
      "shape": [
        1,
        28,
        28,
        1
      ],
      "type": "float32"
    }
  ]
}
```

### Model Inferencing

Each model provides an endpoint for inferencing. The input data is sent via a POST request.

#### Example 1:

##### POST: `http://localhost:8000/models/Mnist`
###### Body:
```json
{
  "features": {
    "image": [[[...]]]  // Example input data
  }
}
```

##### Result:
```json
{
  "result": {
    "dense": [
      [
        7.09201655735292e-11,
        1.359023028661699e-13,
        1.7421603049072587e-9,
        3.211816022030689e-7,
        0.0000636394033790566,
        4.23660395654224e-8,
        1.9490131554565464e-14,
        0.00002632792529766448,
        0.000017869479052023962,
        0.9998917579650879
      ]
    ]
  }
}
```

#### Example 2:

##### POST: `http://localhost:8000/models/BostonHousePrices`
###### Body:
```json
{
  "features": {
    "input_1": [[-0.39592582, 0.85374287, -1.30031899, -0.29541676, -0.68858289, 0.82781921, 0.07100048, -0.31650001, -0.30917602, -1.08348329, 0.42300873, -0.71757293]]
  }
}
```

##### Result:
```json
{
  "result": {
    "price_output": [
      [
        31.021991729736328
      ]
    ],
    "ptratio_output": [
      [
        15.65169620513916
      ]
    ]
  }
}
```
