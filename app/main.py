from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from .detection import ml_detection, utils
from contextlib import asynccontextmanager
from typing import Optional


def detection(detr_processor, detr_model, image_bytes):
    # Object detection
    results = ml_detection.object_detection(detr_processor, detr_model, image_bytes)

    # Convert dictionary of tensors to JSON
    result_json = utils.convert_tensor_dict_to_json(results)

    return result_json


# Example with global variable as dict
# ml_detr = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # ml_detr = ml_detection.load_model()
    app.processor50, app.model50 = ml_detection.load_model("facebook/detr-resnet-50")
    app.processor101, app.model101 = ml_detection.load_model("facebook/detr-resnet-101")

    yield
    # Clean up the ML model and release the resources
    del app.processor50, app.model50
    del app.processor101, app.model101
    # ml_detr.clear()


app = FastAPI(
    lifespan=lifespan,
    title="Object detection",
    description="Object detection on COCO dataset",
    version="1.0",
)


@app.get("/")
def home():
    return {"message": "Welcome to the object detection API"}

@app.get("/status")
def info():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"name": "object-detection", "description": "object detection on COCO dataset"}


# Detection with optional model type
@app.post("/api/v1/detect")
async def detect(image: UploadFile = File(...), model: Optional[str] = Query(None)):
    # Read the image file
    image_bytes = await image.read()

    # ML detection
    if (model is None) or (model == "detr-resnet-50"):
        output_json = detection(app.processor50, app.model50, image_bytes)
    elif model == "detr-resnet-101":
        output_json = detection(app.processor101, app.model101, image_bytes)
    else:
        raise HTTPException(status_code=400, detail="Incorrect model type")
    return output_json
