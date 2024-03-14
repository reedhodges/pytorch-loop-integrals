import random
import shutil

masses = ["m_" + str(i) for i in range(1, 5)]

def generate_massless_propagator():
    """
    Generate a massless propagator.

    Returns:
        tuple: A tuple containing the massless propagator expression and its power.
    """
    pwr = random.randint(0, 3)
    if pwr == 0:
        return "", pwr
    elif pwr == 1:
        return "p^2", pwr
    else:
        return f"(p^2)^{pwr}", pwr

def generate_massive_propagator():
    """
    Generate a massive propagator.

    Returns:
        tuple: A tuple containing the massive propagator expression and its power.
    """
    pwr = random.randint(1, 3)
    mass_idx = random.randint(0, len(masses)-1)
    if pwr == 1:
        return f"(p^2 - {masses[mass_idx]}^2)", pwr
    else:
        return f"(p^2 - {masses[mass_idx]}^2)^{pwr}", pwr

def generate_numerator():
    """
    Generate a numerator.

    Returns:
        tuple: A tuple containing the numerator expression and its power.
    """
    pwr = random.randint(0, 3)
    if pwr == 0:
        return "1", pwr
    elif pwr == 1:
        return "p^2", pwr
    else:
        return f"(p^2)^{pwr}", pwr

def construct_integrand():
    """
    Construct an integrand.

    Returns:
        tuple: A tuple containing the integrand expression, dimension, numerator power,
               massless propagator power, and sum of massive propagator powers.
    """
    dimension = random.randint(2, 8)
    numerator, num_pwr = generate_numerator()

    if num_pwr != 0:
        massless_prop, massless_pwr = "", 0
    else:
        massless_prop, massless_pwr = generate_massless_propagator()

    props = []
    sum_massive_pwr = 0
    num_massive_props = random.randint(0, len(masses)-1)
    for _ in range(num_massive_props):
        prop, pwr = generate_massive_propagator()
        props.append(prop)
        sum_massive_pwr += pwr
        
    if massless_pwr == 0 and sum_massive_pwr == 0 and num_pwr == 0:
        integrand = f"$\int d^{dimension}p$"
    elif massless_pwr == 0 and sum_massive_pwr == 0 and num_pwr != 0:
        integrand = f"$\int d^{dimension}p " + f"{numerator}$"
    else:
        props.insert(random.randint(0, len(props)), massless_prop)
        integrand = f"$\int d^{dimension}p " + "\\frac{" + f"{numerator}" + "}" + "{" + "".join(props) + "}$"
    
    return integrand, dimension, num_pwr, massless_pwr, sum_massive_pwr

def generate_integrand_and_check_divergence():
    """
    Generate an integrand and check its divergence.

    Returns:
        tuple: A tuple containing the integrand expression and its divergence classification.
    """
    integrand, dimension, num_pwr, massless_pwr, sum_massive_pwr = construct_integrand()
    IR_divergent = False
    UV_divergent = False
    total_num_pwr = 2*num_pwr + dimension - 1
    if massless_pwr > total_num_pwr:
        IR_divergent = True
    total_den_pwr = 2*massless_pwr + 2*sum_massive_pwr
    if total_den_pwr < total_num_pwr:
        UV_divergent = True

    if not IR_divergent and not UV_divergent:
        classification = 0
    elif IR_divergent and not UV_divergent:
        classification = 1
    else:
        classification = 2
    
    return integrand, classification

import csv
with open('integrand_data.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Integrand", "Divergence Classification"])
    for _ in range(5000):
        integrand, classification = generate_integrand_and_check_divergence()
        writer.writerow([integrand, classification])

import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_and_save_images(df, data_type):
    """
    Create and save images for integrands.

    Args:
        df (DataFrame): Pandas DataFrame containing integrands and their classifications.
        data_type (str): Type of data (e.g., 'train', 'test').
    """
    print(f"Processing {data_type} data...")
    for i, row in tqdm(df.iterrows()):

        # randomly change the font
        matplotlib.rcParams['text.usetex'] = bool(random.getrandbits(1))

        latex_string = row["Integrand"]
        class_idx = row["Divergence Classification"]
        class_dict = {0: "safe", 1: "ir_div", 2: "uv_div"}
        directory_path = f"data/{data_type}/" + class_dict[class_idx]
        os.makedirs(directory_path, exist_ok=True) 

        font_sizes = [16, 20, 24, 28, 32]
        
        plt.figure()
        plt.text(0.5, 0.5, latex_string, 
                 fontsize=random.choice(font_sizes), 
                 va='center', 
                 ha='center')
        plt.axis('off')
        plt.savefig(f"{directory_path}/integrand_{i}.png")
        plt.close()  

def split_data_and_save_images(data_filepath, test_size=0.2, random_state=42):
    """
    Split data into train and test sets and save images for integrands.

    Args:
        data_filepath (str): Filepath of the CSV file containing integrands.
        test_size (float): Size of the test set as a fraction of the whole dataset.
        random_state (int): Random seed for reproducibility.
    """
    print("Deleting old data...")
    shutil.rmtree("data")
    data = pd.read_csv(data_filepath)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    create_and_save_images(train_data, "train")
    create_and_save_images(test_data, "test")

split_data_and_save_images("integrand_data.csv")
