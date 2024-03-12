import random
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def plot_random_image(image_path_list, data_transform):
    random_image_path = random.choice(image_path_list)
    im = Image.open(random_image_path)
    im = data_transform(im)
    plt.imshow(im.permute(1, 2, 0))
    plt.title(random_image_path.parent.name)
    plt.show()