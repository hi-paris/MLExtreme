import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from MLExtrem.utils import dataset_generation as dg, model_generation as mg, norm_generation as ng
from MLExtrem.supervised.classification import Classifier

# Data generation
data = dg.gen_multilog(n, Dim, alpha, Hill_index)
label = dg.gen_label2(data, angle=angle)

# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue')
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
plt.show()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split, random_state=42)

# Model generation
model = mg.Get_Model('RandomForest', problem_type='classification')

# Normalization function
norm_func = lambda x: np.linalg.norm(x, ord=2, axis=1)

# Classifier class initialization
classifier = Classifier(model, norm_func)

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
