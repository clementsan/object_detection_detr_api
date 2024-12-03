from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import io


def load_model(model_uri: str):
    """Load DETR Detection-Transformer model"""
    """Doc: https://huggingface.co/docs/transformers/en/model_doc/detr"""

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(model_uri, revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(model_uri, revision="no_timm")
    return processor, model


def object_detection(processor, model, image_bytes):
    """Perform object detection task"""

    print('Object detection...')
    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)

    img = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=img, return_tensors="pt")
    # print('inputs', inputs)
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

