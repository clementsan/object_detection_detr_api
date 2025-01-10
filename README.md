# Object detection via FastAPI

[![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker Pulls](https://img.shields.io/docker/pulls/cvachet/object-detection-detr-api)](https://hub.docker.com/repository/docker/cvachet/object-detection-detr-api)

![example workflow](https://github.com/clementsan/object_detection_detr_api/actions/workflows/ci_python.yml/badge.svg)
![example workflow](https://github.com/clementsan/object_detection_detr_api/actions/workflows/publish_docker_image.yaml/badge.svg)

**Aim:** AI-driven object detection via FastAPI (on COCO image dataset)

**Machine learning models:**
 - facebook/detr-resnet-50
 - facebook/detr-resnet-101

------
### Table of contents:
 - [Direct execution](#1-direct-execution)
   - [Run FastAPI interface](#11-run-fastapi-interface)
   - [Run API query via curl command](#12-run-api-query-via-curl-command)
   - [Run API query via python script](#13-run-api-query-via-python-script)
   - [Tests via pytest library](#14-tests-via-pytest-library)
 - [Execution via docker container](#2-execution-via-docker-container)
   - [Build image and run docker container](#21-build-image-and-run-docker-container)
   - [Run query via API](#22-run-query-via-api)
 - [Deployment on Docker hub](#3-deployment-on-docker-hub)
 - [MLOps pipeline via GitHub actions](#4-mlops-pipeline-via-github-actions)

----

## 1. Direct execution

### 1.1. Run FastAPI interface

Command line in development mode:
> fastapi dev app/main.py

Manual command line in production:
> fastapi run app/main.py

<b>Notes:</b>
 - Serving at: http://127.0.0.1:8000 
 - API docs: http://127.0.0.1:8000/docs



### 1.2. Run API query via curl command

Command lines:
 - Endpoint "/":
> curl -X 'GET' -H 'accept: application/json' 'http://127.0.0.1:8000/'

- Endpoint "/info":
> curl -X 'GET' -H 'accept: application/json' 'http://127.0.0.1:8000/info'
  

 - Endpoint "/api/v1/detect":
>  curl -X POST -F "image=@./tests/data/savanna.jpg" http://127.0.0.1:8000/api/v1/detect

 - Endpoint "/api/v1/detect" with optional model type (e.g. detr-resnet-50 or detr-resnet-101):
>  curl -X POST -F "image=@./tests/data/savanna.jpg" http://127.0.0.1:8000/api/v1/detect?model=detr-resnet-50


### 1.3. Run API query via python script

Command line:
> python app/inference_api.py -u "http://127.0.0.1:8000/api/v1/detect" -f tests/data/savanna.jpg

### 1.4. Tests via pytest library

Command lines:
> pytest tests/ -v

> pytest tests/ -s -o log_cli=true -o log_level=DEBUG


## 2. Execution via docker container

### 2.1. Build image and run docker container

Command lines:
> sudo docker build -t object-detection-detr-api .

> sudo docker run --name object-detection-detr-api-cont -p 8000:8000 object-detection-detr-api

### 2.2. Run query via API

Command lines:
 - Endpoint "/":
> curl -X 'GET' 'http://0.0.0.0:8000/' -H 'accept: application/json'

 - Endpoint "/info":
> curl -X 'GET' -H 'accept: application/json' 'http://0.0.0.0:8000/info'

 - Endpoint "/api/v1/detect":
>  curl -X 'POST' -F "image=@./tests/data/savanna.jpg" http://0.0.0.0:8000/api/v1/detect 

 - Endpoint "/api/v1/detect" with optional model type (e.g. detr-resnet-50 or detr-resnet-101):
>  curl -X POST -F "image=@./tests/data/savanna.jpg" http://0.0.0.0:8000/api/v1/detect?model=detr-resnet-50


## 3. Deployment on Docker hub

This FastAPI application is available as a Docker container on Docker hub

URL: https://hub.docker.com/r/cvachet/object-detection-detr-api


## 4. MLOps pipeline via GitHub actions

Github actions were created to enable Continuous Integration (CI) and Continuous Deployment (CD) for this FastAPI app. 

YAML files:
 - Python testing suite: [ci_python.yml](.github/workflows/ci_python.yml)
 - Pushing to docker: [publish_docker_image.yaml](.github/workflows/publish_docker_image.yaml)
