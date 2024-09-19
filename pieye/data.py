"""

Refs: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import os
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.nn import functional as F
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes


def get_filepaths(dir: str, ext: List[str] | str = ["png", "jpg"]) -> List[str]:
    if isinstance(ext, str):
        ext = [ext]
    return sorted([os.path.join(dir, f) for f in os.listdir(dir) if os.path.splitext(f)[1][1:] in ext])


def one_hot_encode_azureml_data_labels(
    annotations_filepath: str,
    classes: List[str] = None,
    returns_full_path: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Convert AzureML dataset labels to one-hot encoding

    Args:
        annotations_filepath: path to the label csv file generated from AzureML data labeler
        classes: list of classes to encode. If None, will be inferred from the label file
        returns_full_path: return full path of the image file or just the filename

    Returns:
        List of image filenames, List of one-hot encoded labels
    """
    labels_df = pd.read_csv(annotations_filepath)

    # Get filename
    def _parse_filepath(url):
        return url.split("/")[-1]
    
    def _parse_fullpath(url):
        # ['AmlDatastore:', '', 'data-labeling-project-name', ...]
        return "/".join(url.split("/")[3:])
    
    filenames = labels_df["Url"].apply(_parse_fullpath if returns_full_path else _parse_filepath)

    # Get labels
    one_hot_encoder = MultiLabelBinarizer(classes=classes)
    one_hot_encodings = one_hot_encoder.fit_transform(
        labels_df["Label"].apply(lambda x: x.split(",")).values
    )
    
    return filenames, one_hot_encodings


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_filepath: str = None,
        image_filepaths: List[str] = None,
        labels: np.ndarray = None,
        classes: List[str] = None,
        label_parser: Callable = one_hot_encode_azureml_data_labels,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ):
        """Image classification dataset

        Args:
            images_dir: directory containing images
            annotations_filepath: label csv file with the following format:
                ```
                202409130001.jpg, 0
                202409130001.jpg, 3
                ...
                ```
            image_filepaths: list of image filepaths
            labels: list of labels
            transforms: torch.transforms
            target_transforms: label transform function
        """
        if annotations_filepath is not None and label_parser is not None and classes is not None:
            image_filepaths, labels = label_parser(
                annotations_filepath=annotations_filepath,
                classes=classes,
            )
            
        self.labels = labels
        self.images = [os.path.join(images_dir, f) for f in image_filepaths]
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]  # columns are [filename, label]
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        label = torch.from_numpy(label)
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
