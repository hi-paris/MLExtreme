# # Tutorial notebook for regression on extreme covariates

# import os
# os.getcwd()
# os.chdir("../")
# os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pareto
import MLExtreme as mlx

# ## 1. data generation in  a simple additive model
# the assumptions for regular variation w.r.t. the covariate
# (see Huet et al.) is satisfied.

# Parameters for data generation (logistic model)
n = 10000
Dim = 2
# alpha_dep = 0.7
# X = mlx.gen_multilog(n,Dim,alpha_dep)**(1/4)
w1 = np.random.uniform(size=n)
W = np.column_stack([w1, 1-w1])
R = pareto.rvs(b=4, size=n)
X = W*R.reshape(-1, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

def tail_reg_fun(angle):
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([10 * np.ones(p1), 0.1 * np.ones(p - p1)])
    return np.dot(angle, beta)

def bulk_reg_fun(angle):
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([0.1 * np.ones(p1), 10 * np.ones(p - p1)])
    # beta = np.ones(p)
    return np.dot(angle, beta)

def bulk_decay_fun(radius):
    return 1 /(radius)**4

y = mlx.gen_target_CovariateRV(X,tail_reg_fun,bulk_reg_fun,bulk_decay_fun)
# Vizualisation
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', alpha=0.5)
colorbar = plt.colorbar(scatter)
colorbar.set_label('Target Value (y)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot  with Heatmap Colors')
plt.show()

# ## Training and prediction, for fixed values of k for training and fixed
# prediction threshold.

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
# Choice of an off-the-shelf classification algorithm
# see https://scikit-learn.org/stable/supervised_learning.html

# Pick  a classifier model in sklearn, previously imported
model = RandomForestRegressor()

# Normalization function
def norm_func(x):
    return np.linalg.norm(x, ord=2, axis=1)


# Regressor class initialization
regressor = mlx.Regressor(model, norm_func)

# thresholds definitions
ratio_train = 0.1
ratio_test = 0.01
Norm_test = norm_func(X_test)
thresh_predict = np.quantile(Norm_test, 1-ratio_test)
k_train = ratio_train * n

# Model training
threshold, ratio, X_train_extreme = regressor.fit(X_train,  y_train,
                                                  k=k_train)
# predict above a larger threshold (extrapolation)

# Prediction on the test data
y_pred_extreme,  X_test_extreme, mask_test = regressor.predict(
                                            X_test, thresh_predict)

# mean_squared_error evaluation
y_test_extreme = y_test[mask_test]
mse = mean_squared_error(y_test_extreme, y_pred_extreme)
print(f'mse: {mse:.4f}')

# Display classification results
# show X
regressor.plot_predictions(y_test_extreme,y_pred_extreme)

# ## Threshold choice by cross-validation
# 
#  thresh_predict is kept fixed as above (downstream task) but k_train
# must now be chosen by the user

ratio_train = np.geomspace(0.005, 0.05, num=10)
k_train = (n * ratio_train).astype(int)

regressor = mlx.Regressor(model, norm_func)


kscores = []
kscores_sd = []
# !time consuming
for k in k_train:
    mean_scores, sd_mean_scores, _ = regressor.cross_validate(
        X, y, k=k, thresh_predict=thresh_predict,
        scoring=mean_squared_error,
        random_state=42 + 100*k)
    kscores.append(mean_scores)
    kscores_sd.append(sd_mean_scores)

kscores = np.array(kscores)
kscores_sd = np.array(kscores_sd)
plt.plot(k_train, kscores)
plt.fill_between(k_train, kscores + 1.64 * kscores_sd,
                 kscores - 1.64 * kscores_sd, color='blue', alpha=0.2)
plt.show()

i_opt = np.argmin(kscores)
k_opt = k_train[i_opt]
k_opt

# optimal k seems to be around 150 on this example. 


