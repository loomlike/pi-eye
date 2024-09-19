import argparse

import mlflow
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pieye.data import ClassificationDataset, one_hot_encode_azureml_data_labels
from pieye.models import BagDetectionMobileNetV3
from pieye.labels import bag_detection_labels as classes


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, help="Path to save the trained model file.")
parser.add_argument("-i", "--image-path", type=str, help="Path to the folder containing images.")
parser.add_argument("-l", "--label-path", type=str, help="Path to the label csv file.")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("-r", "--learning-rate", type=float, default=0.005, help="Learning rate.")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BagDetectionMobileNetV3(classes=classes)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(250),
        transforms.RandomRotation(30),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': model.preprocess,
}

filenames, labels = one_hot_encode_azureml_data_labels(
    args.label_path,
    classes=classes,
)

X_train, X_val, y_train, y_val = train_test_split(
    filenames, labels, test_size=0.3, random_state=42,
)

train_dataset = ClassificationDataset(
    images_dir=args.image_path,
    image_filepaths=X_train,  # passing filenames
    labels=y_train,
    transforms=data_transforms["train"],
)

val_dataset = ClassificationDataset(
    images_dir=args.image_path,
    image_filepaths=X_val,  # passing filenames
    labels=y_val,
    transforms=data_transforms["val"],
)

dataloaders = {
    "train": DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8),
    "val": DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8),
}

with mlflow.start_run():
    mlflow.log_param("lr", args.learning_rate)
    mlflow.log_param("epochs", args.epochs)

    # BCEWithLogitsLoss for Multi-label classification.
    # It combines a Sigmoid layer and the BCELoss in one single class.
    criterion = nn.BCEWithLogitsLoss()

    # Note that we are only training the head first.
    optimizer = torch.optim.SGD(
        model.get_last_layer_parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    )

    print("Train the head first for 5 epochs w/ lr=0.01.")
    model.train(
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        logger=mlflow.log_metric,
    )   

    print(f"Train the whole model for {args.epochs} epochs w/ lr={args.learning_rate}.")
    optimizer = torch.optim.SGD(
        model.get_parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train(
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=args.epochs,
        logger=mlflow.log_metric,
    )    

    print(f"Saving the best model to {args.model_path}.")
    model.save(args.model_path)
