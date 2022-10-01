# Model Request API for EvoAI connection

In this project, a RESTFull API is developed to connect to the EvoAI framework. This project is currently work-in-progress state.

Additional ML-based Models can easily extend the API by adding the Models to the directory Models/ and a Model-Description file. This simple JSON template gives the API all the necessary details of your model. The models are automatically discovered and are available with their URLs. 

As mentioned, the project is in a work-in-progress state; thus, only sklearn models saved with pickels are supported. The model and the description file must have identical names, except for the ending. 

## Starting the API:
uvicorn main:app --reload

## Requesting the API 

Following is an exemplary URL:
localhost:8000/model/5?data={"INS": "1","Frequenc": "2","CUDA Cores": "3","RAM": "4","Base Frequency": "5","Boost Frequency": "5","Max Power": "5","Storage speed": "5","max Temp": "5","SMs": "5","GFLOPS": "6","Texture Units": "5","Memory Clock": "5","Memory Bandwidth": "5","ROPs": "5","L2 Cache": "5","Transistors": "5","Manufacturing Process": "5","Architecture": "5","TrainParams": "5", "NotSupporterFeature":"8"} 