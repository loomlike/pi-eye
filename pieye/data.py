"""

Refs: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import os
from typing import Callable, List

import pandas as pd
import torch
from torch.nn import functional as F
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes


def get_filepaths(dir: str, ext: List[str] | str = ["png", "jpg"]) -> List[str]:
    if isinstance(ext, str):
        ext = [ext]
    return sorted([os.path.join(dir, f) for f in os.listdir(dir) if os.path.splitext(f)[1][1:] in ext])


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_filepath: str,
        images_dir: str,
        transforms: Callable,
        target_transforms: Callable = None,
    ):
        """Image classification dataset

        Args:
            annotations_filepath: label csv file with the following format:
                ```
                202409130001.jpg, 0
                202409130001.jpg, 3
                ...
                ```
            images_dir: directory containing images
            transforms: torch.transforms
            target_transforms: label transform function
        """
        self.labels = pd.read_csv(annotations_filepath)
        self.images = get_filepaths(images_dir)
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(self.images[idx])
        label = self.labels.iloc[idx, 1]  # columns are [filename, label]
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        return image, label


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transforms: Callable):
        self.transforms = transforms
        self.images = get_filepaths(images_dir)
        self.masks = get_filepaths(masks_dir)


    def __getitem__(self, idx):
        # load images and masks
        image = read_image(self.images[idx])
        mask = read_image(self.masks[idx])
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        image = tv_tensors.Image(image)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)
