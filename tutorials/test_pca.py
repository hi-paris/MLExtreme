# Author: Anne Sabourin
# Description: DAMEX/CLEF tutorial: comparison
# Date: May 2025

import os
os.getcwd()
os.chdir("../")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import MLExtreme as mlx



# Example RV-dirimix  data generation p = 2
n = 10000  # Number of samples
p = 3  # Ambient dimension (2D in this case)
k = 2  # Number of components in the Dirichlet mixture
alpha = 6# Shape parameter of the Pareto distribution

# Mixture means (Mu), log scale (lnu), and weights (wei)
Mu = np.array([[0.1, 0.1, 0.8], [0.45, 0.45, 0.1]])  # k* p matrix of means
lnu = np.log([50,50] )  # log(10) for both components
wei = np.array([0.5,0.5])  # weights for the mixture components
Mu, wei = mlx.normalize_param_dirimix(Mu, wei)
# # inspect the angular density
# Mu_wei = wei  @ Mu
# # Display the result
# print(Mu_wei)
# mlx.plot_pdf_dirimix_3D(Mu, wei, lnu)
np.random.seed(1)
X = mlx.gen_rv_dirimix(alpha, Mu, wei, lnu, scale_weight_noise=1,
                       index_weight_noise=alpha*1.1,
                       size=n)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='gray', alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

# %%
# Select the radial threshold via distance-covariance tests.
# See the tutorials on classification and regression for more details.
def norm_func(x):
    return np.sum(x**2, axis=1)**(1/2)


ntests_thresh = 10
ratio_ext = np.geomspace(0.05, 0.8, num=ntests_thresh)
pval, ratio_max = mlx.test_indep_radius_rest(X, y=None, ratio_ext=ratio_ext,
                                             norm_func=norm_func)


mlx.plot_indep_radius_rest(pval, ratio_ext, ratio_max, n)



norm_X = norm_func(X)
thresh_train =np.quantile(norm_X, q=1-ratio_max/1.2)
xpca = mlx.Xpca(thresh_train=thresh_train, beta=1, centered=True)

xpca.fit(X)

xpca.eigenvalues
xpca.principal_axes

xpca.screeplot()

X_test = mlx.gen_rv_dirimix(alpha, Mu, wei, lnu, scale_weight_noise=1,
                            index_weight_noise=3,
                            size=n)

X_test_recons, X_test_extreme, mask = xpca.predict(X_test, max_dim=1)

np.min(X_test_recons)


fig = plt.figure()
Z = X_test_recons
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], color='gray', alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

Scores, _, _ = xpca.scores(X_test, max_dim=3)
Scores
xpca.mean_angle


plt.figure()
plt.scatter(Scores[:, 0], Scores[:, 1], c='gray', alpha=0.5)
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()


Err = xpca.recons_error(X_test, max_dim=1, test_beta=0)
n_ext = len(Err)
cumMSE = np.cumsum(Err)/np.arange(1, n_ext+1)
plt.figure()
plt.plot(cumMSE)
plt.show()
