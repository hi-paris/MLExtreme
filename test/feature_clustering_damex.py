# %% [markdown]
# # [Damex tutorial]

# %% [markdown]
"""
Implements the DAMEX algorithm described in [1].
The considered unsupervised task is to discover the groups of components of a random vector which are comparatively likely to be simultaneously large. 

[1] Goix, N., Sabourin, A., & Clémençon, S. (2017). Sparse representation of multivariate extremes with applications to anomaly detection. Journal of Multivariate Analysis, 161, 12-31.
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

# Define the norm function as the infinite norm.
def norm_func(x):
    return np.max(x, axis=1)

# %% [markdown]
# Define or draw a list of subfaces of the unit sphere forming the
# support of the limit measure.

# %%
Plot = True
seed = 42
dim = 10  # try 5, 20, 50, 100
num_subfaces = 5 # try 5, 20, 50
subfaces_list = mlx.gen_subfaces(dimension=dim,
                                 num_subfaces=num_subfaces,
                                 max_size=5, # try 4, 10, 20
                                 prevent_inclusions=False,
                                 seed=seed)

# # uncomment for a simpler example:
# subfaces_list = [[0, 1], [1, 2], [2, 3, 4]]

if False:   # change to True to print the list of subfaces 
    print(subfaces_list)
    pp.pprint(mlx.list_to_dict_size(subfaces_list))

subfaces_matrix = mlx.subfaces_list_to_matrix(subfaces_list)
#print(subfaces_matrix)

# %% [markdown]
# ##Generate data

# %%
# dimension, number of mixture components, weights and Dirichlet center locations 
n = int(np.sqrt(dim) * 10**3)
k = np.shape(subfaces_matrix)[0]
dim = np.shape(subfaces_matrix)[1]
# define admissible dirichlet mixture parameters
# (for the limit angular measure of a Pareto - marginally - standardized  hevay-tailed vector,
# based on the matrix of subfaces.
wei = np.ones(k)/k
Mu, wei = mlx.normalize_param_dirimix(subfaces_matrix, wei)
# just printing it and recording it as a list for further usage
faces_true = subfaces_list
wei_true = wei
print(f'Mu matrix: \n {np.round(Mu, 3)}')
print(f'Weights: {np.round(wei, 3)}' )

# %% [markdown]
# ` lnu ` parameter below is the logarithm f the concentration parameter for the Dirichlet mixture.
# it is a crucial parameter which determines the 'hardness' of the clustering problem.
# If  any exp(lnu[i)) * Mu[i,j] <1  the problem is pathologicallly hard, in the sense that the mass on any subface concentrates ON THE BOUNDARY of that face (however in low dimension CLEF yields OK results)
# On the contrary if  all exp(lnu[i)) * Mu[i,j]  >> 1, the problem is very easy.
#
# Play around with 'hardness' parameter below, which should be
# strictly positive. Values greater than one correspond to very hard
# problems, values close to zero generate very easy problems. Interesting
# results happen for lnu ~ 0.5
hardness= 0.8

min_mus = np.zeros(k)
for j in range(k):
    min_mus[j] = np.min(Mu[j, Mu[j, :] > 0 ])
lnu = np.log(1/(hardness**2) * 1/min_mus )
# check
if False: 
    print("absolute dirichlet parameters: ")
    print(Mu * np.exp(lnu).reshape(-1,1))


# %%
# other tuning parameters 
alpha = 2  # regular variation index
Mu_bulk = np.ones((k, dim))/dim  # centers of mass of the bulk distribution 
index_weight_noise = 4

# %%
# generate data 
np.random.seed(42)
X = mlx.gen_rv_dirimix(alpha, np.round(Mu, 3),  wei, lnu,
                       scale_weight_noise=1, Mu_bulk=Mu_bulk,
                       index_weight_noise=index_weight_noise, size=n)

# rank transformed data
Xt = mlx.rank_transform(X)

# test data for unsupervised evaluation
Xtest = mlx.gen_rv_dirimix(alpha, np.round(Mu, 3),  wei, lnu,
                           scale_weight_noise=1, Mu_bulk=Mu_bulk,
                           index_weight_noise=index_weight_noise, size=n)

std_Xtest = mlx.rank_transform_test(X, Xtest)

# %% pairwise plot of generated data for two dependent components
if Plot:
    len_faces = np.sum(subfaces_matrix, axis=1)
    r_i = np.where(len_faces > 1)[0][0]
    iplot = subfaces_list[r_i][0]
    jplot = subfaces_list[r_i][1]
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

# %%  rank transformation of the generated data wiht unit pareto margins
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
#  ## Radial threshold selection 

# %%
# selecting the radial threshold
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
# Beware the following is not 'k' from . it is comprised between k and d * k.
number_extremes = np.sum(norm_Xt >= threshold )

# %%[markdown]
# ## Choosing Epsilon: Two automatic rules.
# This is an  original, unpublished method for selecting the parameter epsilon
# in DAMEX.  While not yet supported by formal theory, the underlying
# concept is intuitive and practical.
# 
# ### A) An AIC-Based Method.
# The goal is to avoid trivial clusterings, such as a
# uniform scattering of points on many subfaces.  Here, a
# cluster is identified with a subface.
#
# **Key Idea**: A well-chosen
# epsilon should yield an informative clustering of points, characterized
# by low entropy. This is directly related to the AIC of a categorical model where each identified subface corresponds to a category. 
# To avoid unstable solutions   for small epsilon with one large cluster (the central face) and a many small clusters (low dimensional subfaces), the algorithm searches for a maximizer of the AIC eps_max_aic (ie a bad epsilon) in the range [0, unsafe_eps_max] and then looks for a minimum [eps_max_aic, vect_eps[-1]], where
# vect_eps is the grid passed in argument for the search. Values of the latter should be sorted in increasing order. 

# %%

ntests = 20
vect_eps = np.geomspace(10**(-3), 0.6, num=ntests)
eps_select_AIC = mlx.damex_select_epsilon_AIC(vect_eps, X, threshold,
                                              plot=True, standardize=True,
                                              unstable_eps_max=0.1)

print('epsilon parameter selections (AIC): ')  # , cardinality): ')
print(eps_select_AIC)

# %%
# parameter setting for DAMEX
# damex 'distance to subface' tolerance parameter: selected via
# AIC method (eps_select_AIC) 
epsilon = eps_select_AIC # keep AIC selected epsilon for further analysis 
print(f'aggregated selected epsilon: {mlx.round_signif(epsilon, 2)}')  
# min number of  samples per retained face / number of extremes:
# #min_ratio_per_subface = 0
min_counts = 0
# # Do not impose a priori any minimum number of points  in the subfaces
# # returned by DAMEX. 
#  alternative:np.sqrt(number_extremes) / 2


# %% [markdown]
# Visualization of rank-transformed data and thresholds

#%%
Xt_disp = Xt**(1/4)
thresh_disp = threshold**(1/4)
eps_thresh_disp = (epsilon * threshold)**(1/4)
max_val = np.max(Xt_disp)*1.01
# NB: exponent alpha/4 above is meant to help visualization only. may
# be removed.
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
# ## DAMEX

# %%
single_train_damex = True  ## works well both with true and false
single_test_damex= True
faces, mass = mlx.damex(X, threshold, epsilon=epsilon, 
                        min_counts= 0,  # np.sqrt(number_extremes)/10,
                        standardize=True, include_singletons=single_train_damex)
 

faces_dict, mass_dict = mlx.list_to_dict_size(faces, mass)
faces_dict_true, mass_dict_true = mlx.list_to_dict_size(faces_true, wei_true)
if False:
    print("List of subfaces found by DAMEX:")
    pp.pprint(faces_dict)
    print("(Damex) Associated limit mass:")
    print(mass_dict)
    print("True  list of subfaces: ")
    pp.pprint(faces_dict_true)
    print("(True) Associated limit mass:")
    print(mass_dict_true)

# %% [markdown]
# Scoring (informed): here we leverage the knowledge of the 'true' subfaces

estim_to_true_error = mlx.setDistance_error(faces, mass, faces_true,
                                            reference_masses=wei_true,
                                            dimension=dim)
true_to_estim_error = mlx.setDistance_error(faces_true, wei_true, faces,
                                            reference_masses=mass, 
                                            dimension=dim)

# goodness of fit, unsupervised
# (in practice, no test set is available but cross-validation may be used instead)
#
#bin_Xtest = mlx.binary_large_features(std_Xtest, threshold)


total_mass = threshold * number_extremes / n
damex_unsupervised_error = mlx.setDistance_subfaces_data(
    faces,  threshold, std_Xtest, include_singletons=single_test_damex,
    epsilon=epsilon,masses=mass)


print(f'DAMEX set distance estimated_to_true_error: {estim_to_true_error}')
print(f'DAMEX set distance true_to_estimated_error: {true_to_estim_error}')
print(f'DAMEX set distance data_to_estimated_error: \
{damex_unsupervised_error}')


# # AIC values
# print(f'DAMEX,  AIC: \
# {mlx.AIC_clustering(mass, number_extremes, total_mass)}')
# print(f'MAX-DAMEX,   AIC: \
# {mlx.AIC_clustering(max_mass, number_extremes, total_mass)}')
# print(f'truncated-DAMEX,  AIC: \
# {mlx.AIC_clustering(truncated_mass, number_extremes, total_mass)}')
# print(f'MAX-truncated-DAMEX,  AIC: \
# {mlx.AIC_clustering(max_truncated_mass, number_extremes, total_mass)}')




# %%
print("\n Now, with a different  choice of epsilon  (ie. previous one /10\n")

# %%
faces_wrong, mass_wrong = mlx.damex(X, threshold, epsilon=epsilon/10, 
                                    min_counts= 0, 
                                    standardize=True,
                                    include_singletons=single_train_damex)
# print("\n")
# print("List of subfaces found by badly tuned DAMEX:")
# pp.pprint(faces_wrong)
# print("Associated limit mass:")
# print(mass)

# %% [markdown]
# Scoring (informed): here we leverage the knowledge of the 'true' subfaces

wrong_estim_to_true_error = mlx.setDistance_error(faces_wrong, mass_wrong,
                                                  faces_true, wei_true,
                                                  dimension=dim)
wrong_true_to_estim_error = mlx.setDistance_error(faces_true, wei_true,
                                                  faces_wrong, mass_wrong,
                                                  dimension=dim)


wrong_damex_unsupervised_error = mlx.setDistance_subfaces_data(
    faces_wrong,  threshold, std_Xtest, include_singletons=single_test_damex,
    epsilon=epsilon,masses=mass_wrong)


# print(f'DAMEX set distance estimated_to_true_error: {estim_to_true_error}')
# print(f'DAMEX set distance true_to_estimated_error: {true_to_estim_error}')
print(f'DAMEX wrong epsilon set distance estimated_to_true_error: \
{wrong_estim_to_true_error}')
print(f'DAMEX wrong epsilon set distance true_to_estimated_error: \
{wrong_true_to_estim_error}')
print(f'DAMEX wrong epsilon  set distance data_to_estimated_error: \
{wrong_damex_unsupervised_error}')



# %% [markdown]
# ## AIC scoring
# Here the likelihood term for the likelihood is that of  the  categorical
# models associated with the clusterings and there mectors of empirical masses.
# Because not all observed faces are recorder  in DAMEX in case of truncation with 'min_counts' >0, or when only maximal faces only are retained, the  likelihood includes an additional category including all samples observed in none of the clusters

# %%

# # maximal faces of damex and associated msases: 
# max_faces = mlx.find_maximal_faces(faces_dict)
# max_mass = mlx.damex_estim_subfaces_mass(max_faces, X, threshold, epsilon, 
#                                         standardize=True)
# # truncated damex 
# truncated_faces, truncated_mass = mlx.damex(
#     X, threshold, epsilon, min_counts=np.sqrt(number_extremes),
#     include_singletons=singletons)
# truncated_faces_dict, _ = mlx.list_to_dict_size(truncated_faces)
# # maximal faces of truncated damex
# max_truncated_faces = mlx.find_maximal_faces(truncated_faces_dict)
# max_truncated_mass = mlx.damex_estim_subfaces_mass(max_truncated_faces,
#                                                    X, threshold, epsilon,
#                                                   standardize=True)

# # missing mass in damex clustering:
# print(f'Total tail mass: {total_mass}')
# print(f'Missing mass DAMEX: {total_mass - np.sum(mass)}')
# print(f'Missing mass MAX-DAMEX: {total_mass - np.sum(max_mass)}')
# print(f'Missing mass truncated-DAMEX: {total_mass - np.sum(truncated_mass)}')
# print(f'Missing mass MAX-truncated-DAMEX: \
# {total_mass - np.sum(max_truncated_mass)}')


# %% [markdown]
# ## Metrics adequacy

# %%

# normalize = True  ## always works with  true. to be removed. 
min_counts =  0 #np.sqrt(number_extremes)/10
# faces_true, counts_true = mlx.damex_0(true_bin_array,
#                                       include_singletons=single_test)
# wei_true = np.ones(len(faces_true))/len(faces_true)

neps = 20
eps_vect = np.geomspace(0.001, 1, num=neps)
damex_unsup_error_vect = np.zeros(neps)
damex_superv_error_vect = np.zeros(neps)
for i in range(neps):
    damex_faces_i, mass_i  = mlx.damex(X, threshold, eps_vect[i],
                                       min_counts=min_counts,
                                       include_singletons=single_train_damex)
    damex_unsup_error_vect[i] = mlx.setDistance_subfaces_data(
        damex_faces_i,  threshold, std_Xtest,
        include_singletons=single_test_damex, 
        epsilon=eps_vect[i], masses=mass_i) 
    damex_superv_1 = mlx.setDistance_error(
        damex_faces_i, mass_i, faces_true, wei_true, dimension=dim)
    damex_superv_2 = mlx.setDistance_error(
        faces_true, wei_true, damex_faces_i, mass_i, dimension=dim)
    damex_superv_error_vect[i]  = max(damex_superv_1, damex_superv_2)

    
# print('damex error for different epsilons:')
# print('unsupervised: ')
# print(damex_unsup_error_vect)
# print('supervised: ')
# print(damex_superv_error_vect)

i_maxerr = np.argmax(damex_unsup_error_vect[eps_vect < 1/10])
eps_maxerr = eps_vect[i_maxerr]
i_mask = eps_vect <= eps_maxerr
i_unsup = np.argmin(damex_unsup_error_vect +
                    (1e+23) * i_mask)
                    #(1e+23) * (damex_unsup_error_vect == 0))
eps_unsup = eps_vect[i_unsup]
error_eps_unsup = damex_superv_error_vect[i_unsup]

i_sup = np.argmin(damex_superv_error_vect +
                    (1e+23) * i_mask)
eps_sup = eps_vect[i_sup]#np.argmin(damex_superv_error_vect)]
error_eps_sup = np.min(damex_superv_error_vect)
print(f'\n unsupervised error-based best epsilon, supervised error: {eps_unsup:4f}, \
{error_eps_unsup:4f}')
print(f'supervised error-based best epsilon, supervised error: {eps_sup:4f}, \
{error_eps_sup:4f}')
if Plot:
    plt.scatter(eps_vect, damex_unsup_error_vect, c='blue', label='unsupervised')
    plt.scatter(eps_vect, damex_superv_error_vect, c= 'orange', label='supervised')
    plt.plot([eps_unsup,eps_unsup],[0, error_eps_unsup],c='blue')
    plt.plot([eps_sup,eps_sup],[0, error_eps_sup],c='orange')
    plt.legend()
    plt.title("DAMEX: sensitivity to epsilon")
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# %% [markdown]
# # CLEF

# %% [markdown]
# Choosing kappa_min: AIC and number of faces, see DAMEX tutorial for details

# %
kappa_num = 35
kappa_vect = np.linspace(0.001, 0.4, num=kappa_num)
kappa_aic = mlx.clef_select_kappa_AIC(
    kappa_vect, X, threshold, plot=True)
# NB include_singletons=False  in AIC  had unstable behaviour: removed. 

print('CLEF: AIC- kappa parameter selection: ')
print(kappa_aic)
kappa_min = kappa_aic  # keep AIC selected kappa for further analysis

# %% [markdown]
# ## CLEF with selected threshold and kappa_min

# %
single_train_clef = True # should CLEF take into account singleton features?
single_test_clef = True
clef_faces = mlx.clef(X, threshold, kappa_min, standardize=True,
                      include_singletons=single_train_clef)

clef_mass = mlx.clef_estim_subfaces_mass(clef_faces, X, threshold)

clef_to_true_error = mlx.setDistance_error(clef_faces, clef_mass, 
                                           faces_true, wei_true, dimension=dim)
true_to_clef_error = mlx.setDistance_error(faces_true, wei_true, 
                                           clef_faces, clef_mass, dimension=dim)
clef_unsupervised_error = mlx.setDistance_subfaces_data(
    clef_faces, threshold, std_Xtest, single_test_clef, epsilon=None,
    masses=clef_mass)
print(f'CLEF set distance estimated_to_true_error: {clef_to_true_error}')
print(f'CLEF set distance true_to_estimated_error: {true_to_clef_error}')
print(f'CLEF set distance data_to_estimated_error: {clef_unsupervised_error}')


# %[markdown] ##Comparison in terms of AIC Here the likelihood term
# for the likelihood is that of the categorical models associated with
# the clusterings and there vectors of empirical masses.  ####Because not
# all observed faces are recorded in CLEF (nor in DAMEX in case of
# truncation with 'min_counts' >0, the likelihood includes an
# additional category including all samples observed in none of the
# clusters

#%%
total_mass =  threshold * number_extremes / n

print(f'CLEF - AIC:  \
{mlx.AIC_clustering(clef_mass, number_extremes, total_mass)}')


# %% [markdown]
# ## Sensitivity to kappa_min, metrics adequacy
# single_test= True gives unstable (or advrsarial) behaviour. remove the option?)
nkappa = 40
kappa_vect = np.geomspace(0.001, 0.5, num=nkappa)
clef_unsup_error_vect = np.zeros(nkappa)
clef_superv_error_vect = np.zeros(nkappa)
for i in range(nkappa):
    clef_faces_i  = mlx.clef(X, threshold, kappa_vect[i], standardize=True,
                             include_singletons=single_train_clef)
    clef_mass_i = mlx.clef_estim_subfaces_mass(clef_faces_i, X, threshold)
    clef_unsup_error_vect[i] = mlx.setDistance_subfaces_data(
        clef_faces_i, threshold, std_Xtest, include_singletons=single_test_clef,
        epsilon=None,  masses=clef_mass_i)
    clef_superv_1 = mlx.setDistance_error(
        clef_faces_i, clef_mass_i, faces_true, wei_true, dimension=dim)
    clef_superv_2 = mlx.setDistance_error(
        faces_true, wei_true, clef_faces_i, clef_mass_i, dimension=dim)
    clef_superv_error_vect[i]  = max(clef_superv_1, clef_superv_2)

# print('clef error for different kappas:')
# print(clef_unsup_error_vect)
# print(clef_superv_error_vect)
i_unsup = np.argmin(clef_unsup_error_vect)
i_superv = np.argmin(clef_superv_error_vect)
kappa_superv = kappa_vect[i_superv]
kappa_unsup = kappa_vect[i_unsup]
error_kappa_sup = clef_superv_error_vect[i_superv]
error_kappa_unsup = clef_superv_error_vect[i_unsup]
print(f'unsupervised error-based best kappa, supervised error:\
{kappa_unsup:4f}, {error_kappa_unsup:4f}')
print(f'supervised error-based best epsilon, supervised error: {kappa_superv:4f}, \
{error_kappa_sup:4f}')

if Plot: 
    plt.scatter(kappa_vect, clef_unsup_error_vect, label='unsupervised', c='blue')
    plt.scatter(kappa_vect, clef_superv_error_vect, label='supervised', c='orange')
    plt.plot([kappa_unsup, kappa_unsup], [0, error_kappa_unsup], c='blue')
    plt.plot([kappa_superv, kappa_superv], [0, error_kappa_sup], c='orange')
    plt.legend()
    plt.title("CLEF: sensitivity to kappa_min")
    plt.show()


# %% [markdown]
# ## CLEF with selected threshold and kappa_min

# %%
clef_faces_unsup = mlx.clef(X, threshold, kappa_unsup, standardize=True,
                            include_singletons=single_train_clef)

clef_mass_unsup = mlx.clef_estim_subfaces_mass(clef_faces_unsup, X, threshold)

clef_unsup_to_true_error = mlx.setDistance_error(
    clef_faces_unsup, clef_mass_unsup, faces_true, wei_true, dimension=dim)
true_to_clef_unsup_error = mlx.setDistance_error(
    faces_true, wei_true, clef_faces_unsup, clef_mass_unsup, dimension=dim)
clef_unsup_unsupervised_error = mlx.setDistance_subfaces_data(
    clef_faces_unsup, threshold, std_Xtest, single_test_clef, epsilon=None,
    masses=clef_mass_unsup)

print(f'tuned CLEF set distance estimated_to_true_error: \
{clef_unsup_to_true_error}')
print(f'tuned CLEF set distance true_to_estimated_error: \
{true_to_clef_unsup_error}')
print(f'tuned CLEF set distance data_to_estimated_error: \
{clef_unsup_unsupervised_error}')

