##TEST function

import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from MLExtrem.utils import dataset_generation as dg, model_generation as mg, norm_generation as ng
from MLExtrem.supervised.tirex import tirex_transform

# Parameters for data generation
n = 10000  
Dim = 2  
split = 0.2  
alpha = 0.9  
Hill_index = 1  
angle = 0.25 

# Generate data
X = dg.gen_multilog(n, Dim, alpha, Hill_index)  
y = dg.gen_label2(X, angle=angle) 

# Parameters for the tirex_transform function
n_components = 2  #
k = int(np.sqrt(n))  # Neighborhood size, typically sqrt(n)
method = "FO"  # Transformation method
mode = "TIREX"  # Mode of transformation
get_SDR_X = False  # Whether to return SDR_X or not

# Apply the tirex_transform function
extreme_space = tirex_transform(
    X, y, n_components=n_components, k=k, method=method, mode=mode, get_SDR_X=get_SDR_X
)

# Display the extracted extreme spaces
print("Extracted extreme spaces (extreme_space):")
print(extreme_space)
