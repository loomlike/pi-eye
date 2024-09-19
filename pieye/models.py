import os
from tempfile import TemporaryDirectory
import time
from typing import Callable, List

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class BagDetectionMobileNetV3():
    def __init__(
        self,
        classes: List[str],
        state_dict_path: str = None,
    ):
        num_classes = len(classes)
        self.classes = classes
        if state_dict_path:
            self.model = mobilenet_v3_large()
            num_features = self.model.classifier[-1].in_features
            # Replace the last layer
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
            # Load weight
            self.model.load_state_dict(torch.load(state_dict_path, weights_only=True))
        else:
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            num_features = self.model.classifier[-1].in_features
            # Replace the last layer
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)

        self.preprocess = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),  # Maybe need this based on the camera angle
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"model initialized:\n{self.model}")

        # default device
        self.device = torch.device("cpu")
    
    def set_device(self, device):
        self.device = device
        # Note, currently the quantized models can only be run on CPU.
        # However, it is possible to send the non-quantized parts of the model to a GPU.
        # Unlike floating point models, you don’t need to set requires_grad=False for the
        # quantized model, as it has no trainable parameters.
        self.model = self.model.to(device)

    def train(self, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10, logger: Callable = None):
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

            torch.save(self.model.state_dict(), best_model_params_path)
            best_weighted_f1 = {"train": 0.0, "val": 0.0}

            num_samples = { phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items() }
            print(f"Number of samples: {num_samples}")

            for epoch in range(num_epochs):
                print(f"Epoch {epoch}/{num_epochs - 1} ==========")

                # Each epoch has a training and validation phase
                for phase in ["train", "val"]:
                    if phase not in dataloaders:
                        continue

                    if phase == "train":
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_preds = []
                    running_trues = []
                    epoch_loss = 0.0
                    epoch_f1 = []
                    epoch_weighted_f1 = 0.0

                    print(f"[{phase} phase]")
                    # Iterate over batches.
                    for inputs, labels in dataloaders[phase]:
                        running_trues.append(labels.numpy())

                        inputs = inputs.to(self.device)
                        labels = labels.float().to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)  # TODO: verify this
                        running_preds.append(outputs.detach().sigmoid().cpu().numpy().round().astype(int))

                    epoch_loss = running_loss / num_samples[phase]
                    if running_preds:
                        running_preds = np.vstack(running_preds)
                        running_trues = np.vstack(running_trues)
                        epoch_f1 = np.around(f1_score(y_true=running_trues, y_pred=running_preds, average=None, zero_division=np.nan), 4).tolist()
                        print(f"{phase} F1 scores: {epoch_f1}")
                        epoch_weighted_f1 = f1_score(y_true=running_trues, y_pred=running_preds, average="weighted", zero_division=np.nan)

                    if phase == "train" and scheduler is not None:
                        scheduler.step()

                    print(f"{phase} Loss: {epoch_loss:.4f} (Weighted f1: {epoch_weighted_f1:.4f})")
                    if logger:
                        for i, f1 in enumerate(epoch_f1):
                            logger(f"{phase}_{self.classes[i]}_f1", f1, step=epoch)
                        logger(f"{phase}_loss", epoch_loss, step=epoch)
                        logger(f"{phase}_f1", epoch_weighted_f1, step=epoch)

                    # Save the best model. If we don't have validation set, use training accuracy
                    if epoch_weighted_f1 > best_weighted_f1[phase]:
                        best_weighted_f1[phase] = epoch_weighted_f1.round(4).item()
                        if ("val" in dataloaders and phase == "val") or ("val" not in dataloaders):
                            torch.save(self.model.state_dict(), best_model_params_path)
                            print(f"Best model saved.")

            time_elapsed = time.time() - since
            print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            print(f"Best weighted f1: {best_weighted_f1}")

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
    def save(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)

    def predict(self, image: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.preprocess(image)
            image_tensor = image_tensor.to(self.device)
            pred = self.model(image_tensor.unsqueeze(0))
        return pred.sigmoid().cpu().numpy()

    def get_last_layer_parameters(self):
        return self.model.classifier[-1].parameters()

    def get_parameters(self):
        return self.model.parameters()
