"""
Refs:
    https://pytorch.org/vision/0.10/auto_examples/plot_visualization_utils.html
"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes


coco_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

np.random.seed(42)
# this will help us create a different color for each class
colors = np.random.randint(0, 256, size=(len(coco_labels), 3))

plt.rcParams["savefig.bbox"] = 'tight'


def show_bboxes(image: Tensor, outputs: Tensor, threshold: float = 0.5):
    pred_scores = outputs[0]['scores'].detach().cpu()
    pred_bboxes = outputs[0]['boxes'].detach().cpu()
    pred_bboxes = pred_bboxes[pred_scores >= threshold]
    pred_idx = outputs[0]['labels'][:len(pred_bboxes)].detach().cpu()
    pred_labels = [coco_labels[i] for i in pred_idx]
    # return pred_labels, pred_idx
    show(
        draw_bounding_boxes(
            image=image,
            boxes=pred_bboxes,
            labels=pred_labels,
            colors=colors[pred_idx].tolist(),
        )
    )


def show(images: Tensor | List[Tensor]):
    if not isinstance(images, list):
        images = [images]

    _, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
