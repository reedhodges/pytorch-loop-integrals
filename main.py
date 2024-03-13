import torch
from torch import nn, optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path

from engine import create_data_loaders, train
from utils import fix_error_with_weights_download

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = Path("data/")
train_path = data_path / "train"
test_path = data_path / "test"

image_path_list = list(data_path.glob("**/*.png"))

fix_error_with_weights_download()

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights).to(device)

data_transform = weights.transforms()

for param in model.features.parameters():
    param.requires_grad = False

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

NUM_EPOCHS = 3
 
train(device, model, train_loader, test_loader, loss_fn, optimizer, epochs=NUM_EPOCHS)