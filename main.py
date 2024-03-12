import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = Path("data/")
train_path = data_path / "train"
test_path = data_path / "test"

image_path_list = list(data_path.glob("**/*.png"))

# this block is needed to fix an error with the weights download
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights).to(device)

data_transform = weights.transforms()

train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=data_transform)

class_names = train_dataset.classes
class_dict = train_dataset.class_to_idx

BATCH_SIZE = 32
NUM_WORKERS = 0 #os.cpu_count()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=len(class_names)).to(device)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 3

for epoch in tqdm(range(NUM_EPOCHS)):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    model.eval()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total

    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss/train_total:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss/test_total:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print()

torch.save(model.state_dict(), "models/efficientnet_b0.pth")

# function to predict the class of an image
def predict_class(image_path, model, class_dict, data_transform):
    model.eval()
    with torch.inference_mode():
        image = Image.open(image_path)
        image = data_transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        class_idx = predicted.item()
        class_name = list(class_dict.keys())[list(class_dict.values()).index(class_idx)]
        return print(f"Predicted class: {class_name}")
    
# test the function
predict_class("custom_images/IMG_5FAB788BE67E-1.png", model, class_dict, data_transform)