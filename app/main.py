from fastapi import FastAPI, File, UploadFile
from .detection import utils
from contextlib import asynccontextmanager


# Example with global variable as dict
# ml_detr = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # ml_detr = utils.load_model()
    app.processor, app.model = utils.load_model()
    yield
    # Clean up the ML model and release the resources
    del app.processor
    del app.model
    # ml_detr.clear()


app = FastAPI(
    lifespan=lifespan,
    title="Object detection",
    description="Object detection on COCO dataset",
    version="0.1",
)


@app.get("/")
def home():
    return {"message": "Welcome to the object detection API"}


@app.get("/api/v1/info")
def info():
    return {"name": "object-detection", "description": "object detection on COCO dataset"}


@app.post("/api/v1/detect")
async def detect(image: UploadFile = File(...)):
    # Read the image file
    image_bytes = await image.read()

    # Object detection
    results = utils.object_detection(app.processor, app.model, image_bytes)

    # Convert dictionary of tensors to JSON
    result_json = utils.convert_tensor_dict_to_json(results)

    return result_json
