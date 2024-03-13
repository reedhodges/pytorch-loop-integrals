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

for param in model.features.parameters():
    param.requires_grad = False


def create_data_loaders(data_path, batch_size=32, num_workers=0, train_transform=None, test_transform=None):
    train_dataset = datasets.ImageFolder(root=data_path / 'train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root=data_path / 'test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    class_dict = train_dataset.class_to_idx

    return train_loader, test_loader, class_names, class_dict

BATCH_SIZE = 32
NUM_WORKERS = 0 #os.cpu_count()

train_loader, test_loader, class_names, class_dict = create_data_loaders(data_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, train_transform=data_transform, test_transform=data_transform)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=len(class_names)).to(device)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_step(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)

def test_step(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)

NUM_EPOCHS = 3

def train(model, train_loader, test_loader, loss_fn, optimizer, epochs=10):
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_accuracy = test_step(model, test_loader, loss_fn)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')
        
train(model, train_loader, test_loader, loss_fn, optimizer, epochs=NUM_EPOCHS)

#torch.save(model.state_dict(), "models/efficientnet_b0.pth")

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
#predict_class("custom_images/IMG_5FAB788BE67E-1.png", model, class_dict, data_transform)