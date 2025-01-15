import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from MLExtrem.utils import dataset_generation as dg, model_generation as mg, norm_generation as ng
from MLExtrem.supervised.regression import Regressor


# Parameters for data generation
n = 100000
Dim = 2
split = 0.2
alpha = 0.9
Hill_index = 1
angle = 0.25

# Data generation
all_data = dg.gen_multilog(n, Dim + 1, alpha, Hill_index)
data = all_data[:, :Dim]  # Select feature columns
label = all_data[:, Dim]  # Select label column

# Optional visualization of the generated data
# plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
# plt.show()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split, random_state=42)

# Model generation
model = mg.Get_Model('RandomForest', problem_type='regression')

# Normalization function
norm_func = lambda x: np.linalg.norm(x, ord=2, axis=1)

# Initialization of the Regressor class
regressor = Regressor(model, norm_func)

# Training the model
threshold, X_train_unit = regressor.fit(X_train, X_test, y_train)

# Prediction on the test data
y_pred, mask_test, X_test_unit = regressor.predict(X_test, threshold)

# Evaluation of Mean Squared Error (MSE)
y_test_extrem = y_test[mask_test]  # Filter test labels based on mask
MSE = mean_squared_error(y_test_extrem, y_pred)
print(f'MSE: {MSE:.4f}')

