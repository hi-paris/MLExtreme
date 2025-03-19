import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from MLExtrem.utils import dataset_generation as dg ##, model_generation as mg, norm_generation as ng
# from MLExtrem.supervised.classification import Classifier
from sklearn.ensemble import RandomForestClassifier

import MLExtrem as mlx



####################################################
## 1 toy example with  hard decision boundary in 2D
####################################################

# Parameters for data generation
n = 10000  
Dim = 2  
split = 0.2  
alpha = 0.9  
angle = 0.25 

# Data generation
data = mlx.gen_multilog(n, Dim, alpha)
label = mlx.gen_label2(data, angle=angle)

# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue')
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
plt.show()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split, random_state=42)

# Choice of an off-the-shelf classification algorithm
## see https://scikit-learn.org/stable/supervised_learning.html

## option 1 (to be removed): use mg suggested models
#model = mg.Get_Model('RandomForest', problem_type='classification')
## option 2: pick yourself a model in sklearn, previously imported
model = RandomForestClassifier()

# Normalization function
norm_func = lambda x: np.linalg.norm(x, ord=2, axis=1)

# Classifier class initialization
classifier = mlx.Classifier(model, norm_func)

# Model training
threshold, X_train_unit = classifier.fit(X_train, X_test, y_train)

# Prediction on the test data
y_pred, mask_test, X_test_unit = classifier.predict(X_test, threshold)

# Accuracy evaluation
y_test_extrem = y_test[mask_test]
accuracy = accuracy_score(y_test_extrem, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Display classification results
classifier.plot_classif(X_test_unit, y_test_extrem, y_pred)

###################################
## 2 example with two classes from noisy dirichlet mixture angular distribution
## based on gen_rv_dirimix 

## class 0: dirichlet parameters
Mu0 = np.array([[0.2, 0.7], [0.8, 0.3]])  # Means of the two components
lnu0 = np.log(10) * np.ones(2)  # log(10) for each component
wei0 = np.array([0.1, 0.9])  # Mixture weights

# Plot the mixture density
mlx.plot_pdf_dirimix_2D(Mu0, lnu0, wei0)

## same for class 1 but symetric
## class 1: dirichlet parameters
Mu1 = np.array([[0.7, 0.2], [0.3, 0.8]])  # Means of the two components
lnu1 = np.log(10) * np.ones(2)  # log(10) for each component
wei1 = np.array([0.1, 0.9])  # Mixture weights
# Plot the mixture density
mlx.plot_pdf_dirimix_2D(Mu1, lnu1, wei1)



help(mlx.gen_rv_dirimix)

n0 = 10000
n1 = 10000
alpha = 2


data0 = mlx.gen_rv_dirimix(n0, alpha, Mu0, wei0, lnu0)
data1 = mlx.gen_rv_dirimix(n1, alpha, Mu1, wei1, lnu1)

data = np.vstack((data0,data1))
label = np.vstack((np.zeros(n0).reshape(-1,1), np.ones(n1).reshape(-1,1))).flatten()
# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue').flatten()
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
plt.show()


# Splitting the data into training and test sets
split = 0.2  
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split, random_state=42)

## choice of a classification algorithm 
model = RandomForestClassifier()

# Normalization function
norm_func = lambda x: np.linalg.norm(x, ord=2, axis=1)

# Classifier class initialization
classifier = mlx.Classifier(model, norm_func)

# Model training
threshold, X_train_unit = classifier.fit(X_train, X_test, y_train)

# Prediction on the test data
y_pred, mask_test, X_test_unit = classifier.predict(X_test, threshold)

# Accuracy evaluation
y_test_extrem = y_test[mask_test]
accuracy = accuracy_score(y_test_extrem, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Display classification results
classifier.plot_classif(X_test_unit, y_test_extrem, y_pred)
## todo : modify this to plot not only the angles but the full test points
