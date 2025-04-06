# %% [markdown]
# # [Damex tutorial]

# %% [markdown]
"""Implements the DAMEX algorithm described in [1] and the CLEF algorithm in [2,3].
The considered unsupervised task is to discover the groups of components of a
random vector which are comparatively likely to be simultaneously large.

The tutorial proposes new methods to choose the tuning parameter
epsilon in DAMEX and kappa in CLEF, mainly based on AIC and
cross-validation, and involving a specific pseudo-likelihood for
(random) subsets of features among {1, ..., d}. Although those methods
have not been investigated in theory, this notebook provides some
empirical evidence of the relevance of the proposed selection
criteria. Alternative stopping criteria for CLEF as propsoed in [3]
are currently not implemented.

[1] Goix, N., Sabourin, A., & Clémençon, S. (2017). Sparse
representation of multivariate extremes with applications to anomaly
detection. Journal of Multivariate Analysis, 161, 12-31.

[2] Chiapino, M., & Sabourin, A.  Feature clustering for extreme
events analysis, with application to extreme stream-flow data. In
International workshop on new frontiers in mining complex patterns
(pp. 132-147). Cham: Springer International Publishing.

[3] Chiapino, M., Sabourin, A., & Segers, J. (2019).  Identifying
groups of variables with the potential of being large
simultaneously. Extremes, 22, 193-222.

"""


# %%
# # Set working directory if necessary
# import os
# os.getcwd()
# #os.chdir("../")
# %% 
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
dim = 20  # try 5, 20, 50, 100
num_subfaces = 5  # try 5, 20, 50
subfaces_list = mlx.gen_subfaces(dimension=dim,
                                 num_subfaces=num_subfaces,
                                 max_size=6, # try 4, 10, 20
                                 prevent_inclusions=False,
                                 seed=seed)

# # uncomment for a simpler example:
# subfaces_list = [[0, 1], [1, 2], [2, 3, 4]]

if False:   # change to True to print the list of subfaces 
    print(subfaces_list)
    pp.pprint(mlx.list_to_dict_size(subfaces_list))

subfaces_matrix = mlx.subfaces_list_to_matrix(subfaces_list,dim)
#print(subfaces_matrix)
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
hardness = 0.5

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# %% [markdown]
# # CLEF ALGORITHM (Chiapino et al. , [1,2])


# %%
#create a CLEF clustering instance and fit it with default value epsilon=0.1.
include_singletons = False
# TODO explain
clusteringClef = mlx.clef(thresh_train=threshold, thresh_test=threshold,
                          include_singletons_train=include_singletons,
                          include_singletons_test=include_singletons)
clef_subfaces, clef_masses = clusteringClef.fit(X)

# %%
# Visualization of rank-transformed data and threshold (default; epsilon=0.1)
Xt_disp = Xt**(1/4)
thresh_disp = clusteringClef.thresh_train**(1/4)
eps_thresh_disp = (clusteringClef.thresh_train)**(1/4)
max_val = np.max(Xt_disp)*1.01
# NB: exponent alpha/4 above is meant to help visualization only. may
# be removed without altering  the analysis.
if Plot:
    scatter = plt.scatter(Xt_disp[:, iplot], Xt_disp[:, jplot], alpha=0.5,
                          c='gray')
    plt.xlim(0.9, max_val)
    plt.ylim(0.9, max_val)
    plt.plot([thresh_disp, thresh_disp], [0, thresh_disp], c='blue',
             label='radial threshold')
    plt.plot([0, thresh_disp], [thresh_disp, thresh_disp], c='blue')
    plt.plot([thresh_disp, max_val], [eps_thresh_disp, eps_thresh_disp], c='red',
             label='tolerance distance-to-subface threshold')
    plt.plot([eps_thresh_disp, eps_thresh_disp], [thresh_disp, max_val], c='red')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature d')
    plt.title('Scatter Plot of the rank_transformed  dataset and \
    selected thresholds')
    plt.legend()
    plt.show()


# %% [markdown]
# inspection of subfaces and masses, with the option of
# considering maximal subfaces (for inclusion) only, to facilitate
# further comparison with CLEF. Indeed in principle, the maximal
# subfaces issued by DAMEX and subfaces issued by CLEF are two
# estimates of the same object, see [2].

# %%
faces_dict_clef, mass_dict_clef = mlx.list_to_dict(clef_subfaces,
                                                   clef_masses)

faces_dict_true, mass_dict_true = mlx.list_to_dict(faces_true, wei_true)

if True:
    print("List of subfaces found by CLEF:")
    pp.pprint(faces_dict_clef)
    print("(CLEF) Associated limit mass:")
    print(mass_dict_clef)
    # print("List of maximal subfaces found by DAMEX:")
    # pp.pprint(faces_max_dict)
    # print("(Damex) Associated limit maximal mass:")
    # print(mass_max_dict)
    print("True  list of subfaces: ")
    pp.pprint(faces_dict_true)
    print("(True) Associated limit mass:")
    print(mass_dict_true)

# %% [markdown]
# ## Unsupervised scoring.
# See the DAMEX section for details.


# %%
# Deviance to true parameters: 
deviance_est_true, deviance_true_est = clusteringClef.deviance_to_true(
    faces_true, wei_true)
# same thing with maximal subfaces
deviance_estmax_true, deviance_true_estmax = clusteringClef.deviance_to_true(
    faces_true, wei_true)


print("AIC deviance estimate of expected deviance:")
print(clusteringClef.get_AIC(Xt, standardize=False))

print("CV estimate of expected deviance")
print(np.mean(clusteringClef.deviance_CV(Xt, standardize=False)))
print("test set  estimate of expected deviance")
print(clusteringClef.deviance(std_Xtest, 
                              standardize=False))
print("Deviances between faces, masses: \
a) estimated to true b) true to estimated")
print(deviance_est_true, deviance_true_est)
print("Deviances between  maximal faces, masses: \
a) estimated to true b) true to estimated")
print(deviance_estmax_true, deviance_true_estmax)


#%% [markdown]
""" For this sample size, AIC appears to be more precise than
cross-validation to estimate the expected deviance. However, this
should not be taken as a general rule until further theoretical
understanding of this pseudo-AIC criterion is achieved. Additionally,
the deviance between the true parameters (faces and masses) and the
estimated ones is, in principle, similar to the test set estimate of
the total deviance, which is also observed here.  """
# TODO CHECK: it is not clear why in practive the (subfaces-mass) deviance is even closer to the CV estimate. this may be just random. 


# %%[markdown]# 
# ## Choosing Epsilon based on AIC / CV
# here and below we propose choosing epsilon in DAMEX using the AIC criterion. A CV-based selection rule is also implemented (but here we retain the AIC slection rule in the end)

# %%
# select kappa_min with AIC and update model:
ntests = 20
vect_kappa = np.geomspace(10**(-3), 1, num=ntests)
kappa_select_AIC, aic_opt, aic_values = clusteringClef.select_kappa_min_AIC(
    vect_kappa, X, plot=True, update_kappa_min=True)


print('kappa_min parameter selection (AIC):')
print(mlx.round_signif(clusteringClef.kappa_min, 2))

# select kappa_min with CV but don't update model in view of earlier findings.
print(clusteringClef.include_singletons_train)
print(clusteringClef.include_singletons_test)
ntests = 20
vect_kappa = np.geomspace(10**(-3), 1, num=ntests)

kappa_select_CV, _, _ = clusteringClef.select_kappa_min_CV(
    vect_kappa, X, cv=5, plot=True,
    include_singletons_train = include_singletons,
    include_singletons_test = include_singletons,
    update_kappa_min=False)

print('kappa_min parameter selection (CV):') 
print(mlx.round_signif(kappa_select_CV, 2))

# check that kappa_min stored in object is still AIC's one.
clusteringClef.kappa_min



# %% [markdown]
# ## wrapping up: comparison of all metrics 

# %%
# recommended choice of kappa here: by AIC
nkappa = 20
kappa_vect = np.geomspace(0.03, 1, num=nkappa)
local_clust = mlx.clef(kappa_min=0.1, thresh_train=threshold,
                       thresh_test=threshold,
                       include_singletons_test=include_singletons,
                       include_singletons_train=include_singletons)
kappa_aic, deviance_aic, aic_vect = local_clust.select_kappa_min_AIC(
    kappa_vect, Xt, standardize=False,  plot=False)

# agreement with other metrics: 
#aic_masses = np.zeros(nkappa)
aic_vals = np.zeros(nkappa)
deviance = np.zeros(nkappa)
deviance_train = np.zeros(nkappa)
deviance_cv = np.zeros(nkappa)
deviance_est_true = np.zeros(nkappa)
deviance_true_est = np.zeros(nkappa)

for i in range(nkappa):
    clust = mlx.clef(kappa_min=kappa_vect[i], thresh_train=threshold,
                     thresh_test=threshold,
                     include_singletons_train=include_singletons,
                     include_singletons_test=include_singletons)
    ##pdb.set_trace()
    faces, masses = clust.fit(Xt, standardize=False)
 #   aic_masses[i] = clust.get_AIC_masses()  # /number_extremes
    aic_vals[i] = clust.get_AIC(Xt,  standardize=False)
    # clust.deviance(Xt, standardize=False) + \
        # 2 * len(clust.masses)/number_extremes
    deviance[i] = clust.deviance(
        std_Xtest, 
        standardize=False)
    deviance_train[i] = clust.deviance(Xt,  standardize=False)
    ##pdb.set_trace()
    deviance_cv_scores = clust.deviance_CV(
        Xt, standardize=False, 
        random_state=13+137*i, cv=5)
    deviance_cv[i] = np.mean(deviance_cv_scores)
    deviance_est_true[i], deviance_true_est[i] = clust.deviance_to_true(
        faces_true, wei_true)

if True:
    plt.scatter(kappa_vect, deviance_train, c='black',
                label='deviance on training set', alpha=0.5)
    plt.scatter(kappa_vect, aic_vals, c='orange', label='AIC', alpha=0.5)
    plt.scatter(kappa_vect, deviance, c='red',
                label='deviance on test set', alpha=0.5)
    plt.scatter(kappa_vect, deviance_cv, c='pink',
                label='deviance_cv')
    plt.scatter(kappa_vect, deviance_est_true, c='blue',
                label='deviance_est_true')
    plt.scatter(kappa_vect, deviance_true_est, c='green',
                label='deviance_true_est')
    plt.plot([kappa_aic, kappa_aic], [0, deviance_aic], c='orange')
    plt.legend()
    plt.title("CLEF metrics sensitivity to kappa_min")
    plt.show()

# %% [markdown]
"""
**Conclusion:** All goes as planned:

    - deviance on testing set and AIC are very close or even
     indistinguishable as soon as kappa_min is not too small, reflecting
     the fact that AIC is a consistent estimate of the deviance on a
     test set.

    - deviance on train set and AIC are close because the number of
      'parameters' of the mode (i.e. number of faces) is small
      compared with the sample size (here, the number of extremes)

    - deviance on train is close, but less than, deviance on test,
      reflecting slight overfitting.

    - deviance on test is a reasonable approximation of the deviance
      of the estimated parameter from the true model (not the other
      way around, think about it)

- CV deviance is different from test deviance but follows a similar pattern.

** Recommendation** to choose kappa_min, use `select_kappa_min_AIC ` and
   `select_kappa_min_CV'. The results should be similar. If they are
   not, you are in trouble because the (extreme) sample size may be too small.

"""
