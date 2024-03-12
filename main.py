import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim

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

def convert_to_rgb(x):
    return x.convert('RGB')

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(convert_to_rgb),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=data_transform)

class_names = train_dataset.classes
class_dict = train_dataset.class_to_idx

BATCH_SIZE = 32
NUM_WORKERS = 0 #os.cpu_count()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

num_classes = len(class_names)
model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 5

for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.inference_mode():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on thme test images: {100 * correct / total} %')