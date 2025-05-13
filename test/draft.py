# %%
# # Set working directory if necessary
import os
os.getcwd()
os.chdir("../")
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
#from scipy.stats import pareto
import pprint as pp
import MLExtreme as mlx

# #import dcor
# import time 


# import pdb
# # %%
# # sample size and dimension
# n = 20000
# Dim = 2

def tata(x):
    x = 3
    print(x)

def titi(x,y, **kwargs):
    return x+y

def toto(z, x, **kwargs):
    return z + titi(x,**kwargs)

toto(x=1,y=2,z=3, a=2)

radius = 3
epsilon = 0.3
X_test = np.array([
    [12, 13, 14],
    [1, 1, 1],
    [12, 0.5, 14],
    [12, 5, 14],
    [1, 2, 3],
    [12, 0.5, 0.5],
    [12, 0.5, 0.5],
    [0.5, 0.5, 12]])

bin_array = mlx.binary_large_features(X_test, radius, epsilon=epsilon)
bin_array
faces, counts = mlx.damex_0(bin_array)
# faces_dict = mlx.list_to_dict_size(faces) # # order faces by their dimension: usage??
faces2, limit_mass = mlx.damex_fit(X_test, radius, epsilon=epsilon,
                                   min_counts=0,
                                   standardize=False)

print(faces)
print(faces2)
print(counts)
print(limit_mass)

# %% [Damex tutorial]

# %%
# Generate data
# Mu = np.array([
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1],
#     [1, 1, 0, 0, 0],
#     [0, 0, 0, 1, 1],
#     [1, 1, 1, 1, 1]
# ])
def norm_func(X):
    return np.max(X, axis = 1)

n = 10000
Mu = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 1],
    ]) 
k = np.shape(Mu)[0]
d = np.shape(Mu)[1]
wei = np.ones(k)/k
Mu, wei = mlx.normalize_param_dirimix(Mu, wei)
print(f'Mu matrix: \n {np.round(Mu, 3)}')
print(f'Weights: {np.round(wei,3)}' )
lnu =  np.log(10/np.max(Mu, axis=1)) 
alpha= 2
Mu_bulk  = np.ones((k,d))/d
np.random.seed(42)
X = mlx.gen_rv_dirimix(alpha, np.round(Mu,3),  wei, lnu,
                       scale_weight_noise=1, Mu_bulk=Mu_bulk,
                       index_weight_noise=alpha/2, size=n)

plt.figure(figsize=(10, 10))
X_disp = X**(1/alpha) # for easier viz
max_val = np.max(X_disp)
scatter = plt.scatter(X_disp[:, 0], X_disp[:, 1], alpha=0.5)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
# NB: exponent alpha/4 above is meant to help visualization only. may be removed.
plt.xlabel('Feature 1')
plt.ylabel('Feature d')
plt.title('Scatter Plot of the  dataset')
plt.show()



# %%
# selecting k # not great. k too large. 
# ratio_ext = np.linspace(0.01, 1, num=10)
# pval, ratio_max = mlx.test_indep_radius_rest(Xt, y=None, ratio_ext=ratio_ext,
#                                              norm_func=norm_func)
# mlx.plot_indep_radius_rest(pval, ratio_ext, ratio_max, n)

# %% [markdown]
#  ## Running DAMEX

# %%
Xt = mlx.rank_transform(X)
ratio_extremes = 0.02
norm_Xt = norm_func(Xt)
threshold = np.quantile(norm_Xt, 1 - ratio_extremes) # radial threshold
# Beware the following is not k. it is comprised between k and d * k. 
number_extremes = np.sum(norm_Xt >= threshold)
epsilon = 0.15  # damex 'distance to subface' tolerance parameter
#min min number of  samples per retained face / number of extremes
min_ratio_per_subface = 0.005 
min_counts = int(min_ratio_per_subface * number_extremes)


Xt_disp = Xt**(1/4)
thresh_disp = threshold**(1/4)
eps_thresh_disp = (epsilon * threshold)**(1/4)
# NB: exponent alpha/4 above is meant to help visualization only. may
# be removed.
max_val = np.max(Xt_disp)*1.01
scatter = plt.scatter(Xt_disp[:, 1], Xt_disp[:, 2], alpha=0.5, c='gray')
plt.xlim(0.9, max_val)
plt.ylim(0.9, max_val)
plt.plot([thresh_disp, thresh_disp], [0, thresh_disp], c='blue',
         label='radial threshold')
plt.plot([0, thresh_disp], [thresh_disp, thresh_disp], c='blue')
plt.plot([thresh_disp, max_val], [eps_thresh_disp, eps_thresh_disp], c='red',
         label='tolerance ditance-to-subface threshold')
plt.plot([eps_thresh_disp, eps_thresh_disp], [thresh_disp, max_val], c='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature d')
plt.title('Scatter Plot of the rank_transformed  dataset and thresholds')
plt.legend()
plt.show()

faces, mass = mlx.damex_fit(X, threshold, epsilon, min_counts=min_counts,
                        standardize=True)
print("\n")
print("List of subfaces found by DAMEX:")
pp.pprint(faces)
print("Associated limit mass:")
print(mass)

# %%
# true subfaces
true_bin_array = 1*(np.round(Mu, 3) > 0)
faces0, counts0 = mlx.damex_0(true_bin_array)

print("True faces:")
print(true_bin_array)
pp.pprint(faces0)

x_raw = np.array([[10, 20], [30, 40], [50, 60]])
mlx.rank_transform(x_raw)

x_train = np.array([[1, 2], [3, 4], [5, 6]])
x_test = np.array([[2, 3], [4, 5]])
mlx.rank_transform_test(x_train, x_test)
