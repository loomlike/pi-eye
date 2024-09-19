import argparse
import base64
import io
import json
from typing import Any

import numpy as np
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from pieye.labels import bag_detection_labels as classes
from pieye.models import BagDetectionMobileNetV3


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-path", type=str)

args = parser.parse_args()

app = Flask(__name__)

model = BagDetectionMobileNetV3(num_classes=len(classes), state_dict_path=args.model_path)
preprocess = model.preprocess


def transform_image(image_bytes: Any) -> torch.Tensor:
    image = Image.open(image_bytes)
    return preprocess(image).unsqueeze(0)  # to batch


def get_result(preds: np.ndarray) -> dict:
    return {
        "score": preds[0].tolist(),
        "label": np.array(classes)[preds[0] > 0.5].tolist(),
    }


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = Image.open(io.BytesIO(request.data))
        preds = model.predict(image)

        return jsonify(get_result(preds))


if __name__ == '__main__':
    app.run()
