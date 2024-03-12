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

from utils import plot_random_image

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

plot_random_image(image_path_list, data_transform)