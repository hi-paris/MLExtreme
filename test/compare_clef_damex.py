# Author: Anne Sabourin
# Description: DAMEX/CLEF tutorial: comparison

# %% [markdown]
# # Comparison between CLEF and DAMEX.
"""This notebook demonstrates that DAMEX often identifies many 'false'
subfaces in high-dimensional settings, even when a reasonable
`min_counts` threshold is set to ignore small subfaces, and even when
considering only maximal subfaces.

This tendency for false discovery is even more pronounced in
adversarial settings (controlled by the `add_noisy_feature` Boolean
variable), where a random feature is added to each extreme sample
point.

In contrast, CLEF generally identifies fewer subfaces that are
reasonably close to the 'true' subfaces.

From a purely quantitative perspective, DAMEX generally outperforms
CLEF in terms of deviance.

In summary, while the output of DAMEX may be less interpretable and
convenient for further analysis—a primary motivation behind the
subsequent development of CLEF—DAMEX may still outperform CLEF in
terms of 'distance to truth,' which can be interpreted as predictive
performance.

"""
# %%
# Imports 
import sys
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import MLExtreme as mlx
import pdb

# Define the norm function as the infinite norm (other norms are not
# implemented in DAMEX / CLEF), also defining this norm function is
# not needed for running clef/damex, only unsed her for visualisation
# and analysis of the output.

def norm_func(x):
    return np.max(x, axis=1)


# %% [markdown]
# ## Ground truth definition:
# Define or draw a list of subfaces of the unit sphere defining the
# support of the limit measure.

# %%
Plot = False
seed = 42
dim = 20  # try 5, 20, 50, 100:
# For high dimension reduce max_size to prevent computational overload. 
num_subfaces = 10  # try 5, 20, 50
subfaces_list = mlx.gen_subfaces(dimension=dim,
                                 num_subfaces=num_subfaces,
                                 max_size=10, # try 4, 10, 20
                                 prevent_inclusions=False,
                                 seed=seed)

# # uncomment for a simpler example:
# subfaces_list = [[0, 1], [1, 2], [2, 3, 4]]

if False:   # change to True to print the list of subfaces 
    print(subfaces_list)
    pp.pprint(mlx.list_to_dict_size(subfaces_list))

subfaces_matrix = mlx.subfaces_list_to_matrix(subfaces_list,dim)

# dimension, number of mixture components, weights and Dirichlet center locations 
n = int(np.sqrt(dim) * 10**3)
k = np.shape(subfaces_matrix)[0]
dim = np.shape(subfaces_matrix)[1]
# Define admissible dirichlet mixture parameters  (for the limit angular measure
# of a Pareto - marginally - standardized  heavy-tailed vector, based on
# the matrix of subfaces. Avoids potentially imbalanced settings accross features
wei = np.ones(k)/k
Mu, wei = mlx.normalize_param_dirimix(subfaces_matrix, wei)
# just printing the subfaces and their weight and
# recording it as a list for further usage
faces_true = subfaces_list
wei_true = wei
print(f'Mu matrix: \n {np.round(Mu, 3)}')
print(f'Weights: {np.round(wei, 3)}' )

# %% [markdown]
# ### Hardness settings for the tail problem 
"""
` lnu ` parameter below is the logarithm of the concentration
parameter for the Dirichlet mixture.  it is a crucial parameter
which determines the 'hardness' of the clustering problem.  If any
exp(lnu[i)) * Mu[i,j] <1 the problem is pathologicallly hard, in the
sense that the mass on any subface concentrates ON THE BOUNDARY of
that face On the contrary if all exp(lnu[i)) * Mu[i,j] >> 1, the
problem is very easy.

Play around with 'hardness' parameter below, which should be
strictly positive. Values greater than one correspond to very hard
problems, values close to zero generate very easy problems. Interesting
results happen for hardness ~ 0.8

"""

# %%
hardness = 0.8

min_mus = np.zeros(k)
for j in range(k):
    min_mus[j] = np.min(Mu[j, Mu[j, :] > 0 ])
lnu = np.log(1/(hardness**2) * 1/min_mus )
# check (change to True the condition below to display)
if False:
    print("absolute dirichlet parameters: ")
    print(Mu * np.exp(lnu).reshape(-1, 1))


# %% [markdown]
"""
Other parameter setting governing the speed of convergence of the
conditional distribution above a radial threshold towards the limit
measure.
"""

# %%
# regular variation index of the data
alpha = 2

# centers of mass of the Dirichlet mixture in the bulk (vanishing impact above large radial thresholds):
Mu_bulk = np.ones((k, dim))/dim

# `index_weight_noise' below how fast the impact of noise decreases
# with large radiial thresholds. Namely, the noise 's weight decreases
# as :  (C/radius)**index_weight_noise
index_weight_noise = 4 

# %% [markdown]
# ### Dataset generation and visualisation

# %%
# generate data 
np.random.seed(42)
X = mlx.gen_rv_dirimix(alpha, np.round(Mu, 3),  wei, lnu,
                       scale_weight_noise=1, Mu_bulk=Mu_bulk,
                       index_weight_noise=index_weight_noise, size=n)

# %% [markdown]
# Add a noisy 'marge' feature to each sample point. Set the option to
# True for comparison.

# %%
add_noisy_feature = True
if add_noisy_feature:
    n = np.shape(X)[0]
    dim = np.shape(X)[1]
    for i in range(n):
        jmax = np.argmax(X[i,])
        j_noise = np.random.randint(dim-1)
        if j_noise == jmax:
            j_noise = dim-1
        X[i,j_noise] = X[i, jmax]
        

# define rank-transformed data
Xt = mlx.rank_transform(X)

# generate test data for unsupervised evaluation
np.random.seed(12345)
Xtest = mlx.gen_rv_dirimix(alpha, np.round(Mu, 3),  wei, lnu,
                           scale_weight_noise=1, Mu_bulk=Mu_bulk,
                           index_weight_noise=index_weight_noise, size=5*n)

# define rank-trainsformed test data 
std_Xtest = mlx.rank_transform(Xtest)

# %%
# pairwise plot of generated data for two dependent components
len_faces = np.sum(subfaces_matrix, axis=1)
r_i = np.where(len_faces > 1)[0][0]
iplot = subfaces_list[r_i][0]
jplot = subfaces_list[r_i][1]

if Plot:
    plt.figure(figsize=(10, 10))
    X_disp = X**(alpha/4)  # for easier viz only 
    max_val = np.max(X_disp)
    scatter = plt.scatter(X_disp[:, iplot], X_disp[:, jplot], alpha=0.5)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature d')
    plt.title('Scatter Plot of the raw  dataset')
    plt.show()

# %%
# rank transformed  generated data with  unit pareto margins
# and visualization
if Plot:
    plt.figure(figsize=(10, 10))
    X_disp = Xt**(1/4)  # for easier viz only
    max_val = np.max(X_disp)
    scatter = plt.scatter(X_disp[:, iplot], X_disp[:, jplot], alpha=0.5)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature d')
    plt.title('Scatter Plot of the  rank transformed dataset')
    plt.show()

# %% [markdown]
# # Data Analysis
# ## Radial threshold selection before model fitting

# %%
# selecting the radial threshold via distance-covariance tests,
#see supervised tutorials for more details.
ntests_thresh = 10
ratio_ext = np.geomspace(0.05, 0.3, num=ntests_thresh)
pval, ratio_max = mlx.test_indep_radius_rest(Xt, y=None, ratio_ext=ratio_ext,
                                             norm_func=norm_func)

if Plot:
    mlx.plot_indep_radius_rest(pval, ratio_ext, ratio_max, n)

print(f'maximum ratio of extreme samples from rule-of-thumb \
distance covariance test: {mlx.round_signif(ratio_max, 2)}')
ratio_extremes = ratio_max * 4/5
norm_Xt = norm_func(Xt)
# radial threshold selected with rule-of-thumb radius-versus-rest
# independence tests:
threshold  = np.quantile(norm_Xt, 1 - ratio_extremes)
# Beware the following is not 'k' in the notations of Goix et al's paper.
# Instead  it is comprised between k and d * k. It is the number of data points
# above the radial threshold. 
number_extremes = np.sum(norm_Xt >= threshold )

# ###############
# %% [markdown]
# ##  DAMEX  clustering 

include_singletons = False
use_max_subfaces = True
min_counts = 0  ##np.sqrt(number_extremes)*2
rate = 10

damex_clust = mlx.damex(min_counts=min_counts,  
                        thresh_train=threshold, thresh_test=threshold,
                        include_singletons=include_singletons,
                        use_max_subfaces=use_max_subfaces,
                        rate=rate)

ntests = 20
vect_eps = np.geomspace(10**(-3), 1, num=ntests)
eps_select_aic, damex_aic_opt, damex_aic_values = \
    damex_clust.select_epsilon_AIC(
        vect_eps, Xt, standardize=False, plot=True, update_epsilon=False)

eps_select_cv, damex_cv_opt, damex_cv_values = damex_clust.select_epsilon_CV(
    vect_eps, Xt, standardize=False, plot=True, update_epsilon=False)


damex_clust.fit(Xt, epsilon=eps_select_aic, standardize=False)
_, damex_aic_dev_to_true = damex_clust.deviance_to_true(faces_true, wei_true)


damex_clust.fit(Xt, epsilon=eps_select_cv, standardize=False)
_, damex_cv_dev_to_true = damex_clust.deviance_to_true(faces_true, wei_true)

print('deviance truth to damex aic:')
print(damex_aic_dev_to_true)

print('deviance truth to damex cv:')
print(damex_cv_dev_to_true)

# ############
# %% [markdown]
### CLEF

clef_clust = mlx.clef(thresh_train=threshold, thresh_test=threshold,
                      include_singletons=include_singletons,
                      rate=rate)

ntests = 20
vect_kappa = np.geomspace(10**(-3), 1, num=ntests)
kappa_select_aic, clef_aic_opt, clef_aic_values = \
    clef_clust.select_kappa_min_AIC(
        vect_kappa, Xt, standardize=False, unstable_kappam_max=0.02, 
        plot=True, update_kappa_min=False)

kappa_select_cv, clef_cv_opt, clef_cv_values = clef_clust.select_kappa_min_CV(
    vect_kappa, Xt, standardize=False,  unstable_tol_max=0.02,
    plot=True, update_kappa_min=False, random_state=1)


clef_clust.fit(Xt, kappa_min=kappa_select_aic, standardize=False)
_, clef_aic_dev_to_true = clef_clust.deviance_to_true(faces_true, wei_true)


clef_clust.fit(Xt, kappa_min=kappa_select_cv, standardize=False)
_, clef_cv_dev_to_true = clef_clust.deviance_to_true(faces_true, wei_true)


print('deviance truth to clef aic:')
print(clef_aic_dev_to_true)

print('deviance truth to clef cv:')
print(clef_cv_dev_to_true)

print("number of subfaces found by DAMEX:")
print(len(damex_clust.subfaces))
print("number of subfaces found by CLEF:")
print(len(clef_clust.subfaces))
print("approximate number of true subfaces:")
print(num_subfaces)

# %% [markdown]
# to inspect all subfaces, change to True 
# %%
if False :
    if use_max_subfaces:
        print("damex max subfaces")
        pp.pprint(mlx.list_to_dict(damex_clust.maximal_subfaces,
                                   damex_clust.maximal_masses))
    else:
        print("damex subfaces")
        pp.pprint(mlx.list_to_dict(damex_clust.subfaces,
                                   damex_clust.masses))

    print("clef subfaces")
    pp.pprint(mlx.list_to_dict(clef_clust.subfaces,
                               clef_clust.masses))

    print("true subfaces")
    pp.pprint(mlx.list_to_dict(faces_true,
                               wei_true))

