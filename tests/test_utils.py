from app.detection import utils
import os
import torch
import json


# Test model loading
def test_load_model():
    processor, model = utils.load_model()
    assert processor is not None
    assert model is not None


# Test image detection
def test_object_detection():
    processor, model = utils.load_model()

    # Get the directory of the current test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the image path relative to the test directory
    image_path = os.path.join(test_dir, 'data', 'savanna.jpg')

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    results = utils.object_detection(processor, model, image_bytes)
    assert results is not None
    assert isinstance(results, dict)


# Test dictionary conversion
def test_convert_tensor_dict_to_json():
    my_dict = {'scores': torch.tensor([1, 2, 3])}
    my_list_gt = json.dumps({'scores': [1, 2, 3]})
    my_list = utils.convert_tensor_dict_to_json(my_dict)
    assert my_list == my_list_gt
