import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# from MLExtreme.utils import dataset_generation as dg, model_generation as mg, norm_generation as ng
# from MLExtreme.supervised.regression import Regressor
import MLExtreme as mlx


# Parameters for data generation
n = 100000
Dim = 2
split = 0.2
alpha = 0.2
angle = 0.25

# Data generation
all_data = mlx.gen_multilog(n, Dim + 1, alpha)
data = all_data[:, :Dim]  # Select feature columns
row_norms = np.linalg.norm(all_data, axis=1)

label = all_data[:, Dim]/row_norms  # label column is the last column divided by the norm of each line 

# Optional visualization of the generated data
plt.scatter(data[:, 0], data[:, 1], alpha=0.7, c = label)
plt.xlim(0,10**3)
plt.ylim(0,10**3)
plt.colorbar(label='Color scale')
plt.show()

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

## reconstruction of the third column:
## here,  label = y  = X_3/\sqrt{X_1^2 + X_2^2 + X_3^3}; all entries nonnegative.
##straightforward inversion yields:
# X_3 =  y/ sqrt{1 - y^2} *  sqrt(x_1^2 + X_2^2)

def inverse_transform_target(y,x): ## x is an array of extreme test points, y is the predicted target between 0 and 1. 
    ## this is the inverse transformation relative to the one used to create the label 
    norm_x = np.linalg.norm(x, ord=2, axis=1)
    predicted_x = y/np.sqrt(1-y**2) * norm_x
    return predicted_x

## test of the above function 
true_recons = inverse_transform_target(label, data )
np.sum((true_recons - all_data[:,2])**2)
## good

predicted_X3 = inverse_transform_target(y_pred, X_test[mask_test,:])
true_X3 = inverse_transform_target(y_test_extrem,X_test[mask_test,:])


plt.scatter(true_X3, predicted_X3 , alpha=0.5)
xmax = 10**3
plt.xlim(0,xmax)
plt.ylim(0,xmax)
plt.plot([0, xmax], [0, xmax], 'r--', label='Diagonal: y = x')
plt.show()
