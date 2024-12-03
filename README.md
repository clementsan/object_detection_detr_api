# Object detection via FastAPI

Aim: AI-driven object detection via FastAPI (on COCO image dataset)

## Direct execution

### 1. Run FastAPI interface

Command line in development mode:
> fastapi dev app/main.py

Manual command line in production:
> fastapi run app/main.py

<b>Notes:</b>
 - Serving at: http://127.0.0.1:8000 
 - API docs: http://127.0.0.1:8000/docs



### 2. Run API query via curl command

Command lines:
 - Endpoint "/":
> curl -X 'GET' -H 'accept: application/json' 'http://127.0.0.1:8000/'

- Endpoint "/info":
> curl -X 'GET' -H 'accept: application/json' 'http://127.0.0.1:8000/info'
  

 - Endpoint "/api/v1/detect":
>  curl -X POST -F "image=@./tests/data/savanna.jpg" http://127.0.0.1:8000/api/v1/detect

 - Endpoint "/api/v1/detect" with optional model type (e.g. detr-resnet-50 or detr-resnet-101):
>  curl -X POST -F "image=@./tests/data/savanna.jpg" http://127.0.0.1:8000/api/v1/detect?model=detr-resnet-50


### 3. Run API query via python script

Command line:
> python app/inference_api.py -u "http://127.0.0.1:8000/api/v1/detect" -f tests/data/savanna.jpg

### 4. Tests via pytest library

Command lines:
> pytest tests/ -v

> pytest tests/ -s -o log_cli=true -o log_level=DEBUG


## Execution via docker container

### 1. Create docker container

Command lines:
> sudo docker build -t object-detection-detr-api .

> sudo docker run --name object-detection-detr-api-cont -p 8000:8000 object-detection-detr-api

### 2. Run query via API

Command lines:
 - Endpoint "/":
> curl -X 'GET' 'http://0.0.0.0:8000/' -H 'accept: application/json'

 - Endpoint "/info":
> curl -X 'GET' -H 'accept: application/json' 'http://0.0.0.0:8000/info'

 - Endpoint "/api/v1/detect":
>  curl -X 'POST' -F "image=@./tests/data/savanna.jpg" http://0.0.0.0:8000/api/v1/detect 


