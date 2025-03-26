# %%
# Set working directory if necessary
import os
os.getcwd()
#os.chdir("../")

# %% [markdown]
# Imports

# %% 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pareto
import MLExtreme as mlx

#import dcor
#import time 


# %%
def norm_func(x):
    return np.linalg.norm(x, ord=2, axis=1)



# %%
# sample size and dimension
n = 20000
Dim = 2

# %% [markdown]
# # 1. Regression 

# Generating covariates X
np.random.seed(1)
alpha = 2
X = mlx.gen_rv_dirimix(alpha,  Mu =np.array([ [0.5 ,  0.5]]),
                        wei=np.array([1]), lnu=np.array([2]),
                        size=n)
# w1 = np.random.uniform(size=n)
# W = np.column_stack([w1, 1-w1])
# R = pareto.rvs(b=alpha, size=n)
# X = W*R.reshape(-1, 1)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Simulated Covariates")
plt.show()

# %%
#  Generating the target 
y = mlx.gen_target_CovariateRV(X, param_decay_fun= 2*alpha)
# Vizualisation
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(X[:, 0]**(1/4), X[:, 1]**(1/4), c=y, cmap='gray', alpha=0.5)
# # NB: exponent 1/4 above is meant to help visualization only. may be removed.
# colorbar = plt.colorbar(scatter)
# colorbar.set_label('Target Value (y)')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter Plot of the regression dataset')
# plt.show()



# %% [markdown]
# # 2. Classif 

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

n = 20000
np.random.seed(2)
X, y = mlx.gen_classif_data_diriClasses(mu0=np.array([0.7, 0.3]),
                                        lnu=np.log(10)*np.ones(2),
                                        alpha=2,
                                        index_weight_noise=2, size=n)
# more details: 
# help(mlx.gen_classif_data_diriClasses)


# Visualization of the generated data
colors = np.where(y == 1, 'red', 'blue').flatten()
plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
plt.show()
#



# %% [markdown]
# ## 3. Data Analysis

# %%
# Splitting the data into training and test sets
split = 0.5
n_train = n * (1-split)
# n_test = n * split 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=split,
                                                    random_state=42)

ratios = np.linspace(10/n_train, 0.2, num=100)
pvalues, ratio_max = mlx.test_indep_radius_rest(X_train, y_train,
                                                ratios,  norm_func)
pvalues
mlx.plot_indep_radius_rest(pvalues, ratios,  ratio_max, n_train)

i_max = np.max(np.where(pvalues > 0.05)[0])
ratio_max = ratios[i_max]
ratio_max
k_max = int(ratio_max * n_train)
k_max

class Example:
    def __init__(self, x):
        self.x = x

x = 2

obj = Example(x)

obj.x

x+=1

obj.x

obj.x = obj.x + 1
obj.x

x = []

Example(x)

Example(x)

