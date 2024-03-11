import random

masses = ["m" + str(i) for i in range(1, 5)]

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
        integrand = f"d^{dimension}p"
    elif massless_pwr == 0 and sum_massive_pwr == 0 and num_pwr != 0:
        integrand = f"d^{dimension}p " + f"{numerator}"
    else:
        props.insert(random.randint(0, len(props)), massless_prop)
        integrand = f"d^{dimension}p " + "\\frac{" + f"{numerator}" + "}" + "{" + "".join(props) + "}"
    
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
    
    return integrand, IR_divergent, UV_divergent

integrand, IR_divergent, UV_divergent = generate_integrand_and_check_divergence()
print(integrand, IR_divergent, UV_divergent)