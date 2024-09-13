import base64
import io
import json
from typing import Any

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from pieye.labels import imagenet_labels

app = Flask(__name__)

# Load classification model.
model = models.quantization.mobilenet_v3_large(
    weights=models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT,
    quantize=True,
)
model = torch.jit.script(model)
model = model.eval()

preprocess= transforms.Compose([
    transforms.Resize((224, 224)),  # classification model's resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_image(image_bytes: Any) -> torch.Tensor:
    image = Image.open(image_bytes)
    return preprocess(image).unsqueeze(0)  # to batch


def get_result(outputs: torch.Tensor) -> str:
    values, indices = outputs.softmax(dim=1).max(1)
    return {
        "score": values.item(),
        "label": imagenet_labels[indices.item()], 
    }


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        input_batch = transform_image(file.stream)
        with torch.no_grad():
            outputs = model(input_batch)
        
        return jsonify(get_result(outputs))


if __name__ == '__main__':
    app.run()
