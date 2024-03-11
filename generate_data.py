import random

masses = ["m_" + str(i) for i in range(1, 5)]

def generate_massless_propagator():
    pwr = random.randint(0, 3)
    if pwr == 0:
        return "", pwr
    elif pwr == 1:
        return "p^2", pwr
    else:
        return f"(p^2)^{pwr}", pwr

def generate_massive_propagator():
    pwr = random.randint(1, 3)
    mass_idx = random.randint(0, len(masses)-1)
    if pwr == 1:
        return f"(p^2 - {masses[mass_idx]}^2)", pwr
    else:
        return f"(p^2 - {masses[mass_idx]}^2)^{pwr}", pwr

def generate_numerator():
    pwr = random.randint(0, 3)
    if pwr == 0:
        return "1", pwr
    elif pwr == 1:
        return "p^2", pwr
    else:
        return f"(p^2)^{pwr}", pwr

def construct_integrand():
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
    elif not IR_divergent and UV_divergent:
        classification = 2
    else:
        classification = 3
    
    return integrand, classification

import csv
with open('integrand_data.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Integrand", "Divergence Classification"])
    for _ in range(1000):
        integrand, classification = generate_integrand_and_check_divergence()
        writer.writerow([integrand, classification])

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

latex_string = r"$\int d^6p \frac{1}{(p^2 - m_3^2)(p^2 - m_3^2)^3(p^2 - m_1^2)^2}$"
fig = plt.figure()
plt.text(0.5, 0.5, latex_string, fontsize=32, va='center', ha='center')
plt.axis('off')