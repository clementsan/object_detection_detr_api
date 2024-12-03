import logging
from fastapi.testclient import TestClient
from app.main import app
import os
import json


# Test API - / endpoint
def test_home():
    client = TestClient(app)
    logging.info("Testing home endpoint")
    response = client.get("/")
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the object detection API"}


# Test API - /info endpoint
def test_info():
    client = TestClient(app)
    logging.info("Testing /info endpoint")
    response = client.get("/info")
    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")
    response_gt = {"name": "object-detection", "description": "object detection on COCO dataset"}
    assert response.status_code == 200
    assert response.json() == response_gt


# Test API - /detect endpoint
def test_detect():
    logging.info("Testing /api/v1/detect endpoint")

    with TestClient(app) as client:
        # Get the directory of the current test file
        test_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the image path relative to the test directory
        image_path = os.path.join(test_dir, 'data', 'savanna.jpg')

        with open(image_path, "rb") as f:
            response = client.post("/api/v1/detect", files={"image": ("test_image.png", f, "image/png")})

    logging.debug(f"Response status: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")

    # Convert JSON string to Python dictionary
    response_data = json.loads(response.json())
    response_keys = list(response_data.keys())
    gt_keys = ['scores', 'labels', 'boxes']
    # print('response_data', response_data)
    # print('response_keys', response_keys)

    assert response.status_code == 200
    assert set(response_keys) == set(gt_keys), "Response keys do not match ground truth"
    assert len(response_data['scores']) == 5
    assert len(response_data['labels']) == 5

