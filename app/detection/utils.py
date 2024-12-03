import torch
import json


def convert_tensor_dict_to_json(tensor_dict):
    """Convert a dictionary of tensors to a JSON-serializable dictionary."""

    # Convert tensors to numpy arrays
    numpy_dict = {key: value.detach().cpu().numpy().tolist() if isinstance(value, torch.Tensor) else value
                  for key, value in tensor_dict.items()}

    # Convert to JSON string
    json_str = json.dumps(numpy_dict)
    # print(json_str)
    return json_str


