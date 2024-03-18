import random
import torch
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot_random_image(image_path_list, data_transform):
    """
    Plot a random image from a list of image paths.

    Args:
        image_path_list (list): List of paths to images.
        data_transform (callable): A function/transform to preprocess the image.

    Returns:
        None
    """
    random_image_path = random.choice(image_path_list)
    im = Image.open(random_image_path)
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    im = data_transform(im)
    plt.imshow(im.permute(1, 2, 0))
    plt.title(random_image_path.parent.name)
    plt.show()

def predict_class(device, image_path, model, class_dict, data_transform):
    """
    Predict the class of an image using a trained model.

    Args:
        device (torch.device): Device to perform computations on (CPU or GPU).
        image_path (str): Path to the image.
        model (torch.nn.Module): The neural network model.
        class_dict (dict): Dictionary mapping class indices to class names.
        data_transform (callable): A function/transform to preprocess the image.

    Returns:
        None
    """
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
    
# this block is needed to fix an error with the weights download
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

def fix_error_with_weights_download():
    WeightsEnum.get_state_dict = get_state_dict

def set_seeds(seed: int=42):
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    