
# #######################################
# ## From data_generation: begin
# #######################################

def gen_classif_data_diriClasses(Mu0, wei0, lnu0=None, alpha=4,
                                 size0=1, size1=1):
    wei1 = wei0[::-1]
    if lnu0 is None:
        lnu0 = np.log(2 / np.min(Mu0, axis=1))
    data0 = gen_rv_dirimix(alpha, Mu0, wei0, lnu0, scale_weight_noise=0.5,
                           index_weight_noise=0.5, size=size0)
    data1 = gen_rv_dirimix(alpha, Mu0, wei1, lnu0, scale_weight_noise=0.5,
                           index_weight_noise=0.5, size=size1)
    y = np.vstack((np.zeros(size0).reshape(-1, 1),
                  np.ones(size1).reshape(-1, 1))).flatten()
    X =  np.vstack((data0, data1))
    permut = np.random.permutation(size0 + size1)
    X = X[permut, :]
    y=y[permut]
    return X,  y



# ## target generation for classification models
def gen_label(Matrix, angle=0.2):
    """
    Generate labels for a 2D classification toy example with and
    explicit decision boundary and no difference between tail and bulk
    behaviour

    Parameters:
    -----------
    Matrix : ndarray, shape (n, 2)
        2D array where each row represents a point in 2D space.
    angle : float, optional
        Angular range parameter. Points within this range are labeled as 1.
        Default is 0.2.

    Returns:
    --------
    ndarray, shape (n,)
        Array of labels (0 or 1) for each point in the input Matrix.
    """
    Vect_angle = np.arctan2(Matrix[:, 1], Matrix[:, 0])
    lower_bound = angle * np.pi / 2
    upper_bound = (1 - angle) * np.pi / 2
    label = [
        0 if (point_angle < lower_bound or point_angle > upper_bound) else 1
        for point_angle in Vect_angle
    ]

    return np.array(label)


def gen_label2(Matrix, angle=0.2):
    """
    Generate labels for a 2D toy example with an explicit decision boundary and
    explicit decision boundary and no difference between tail and bulk
    behaviour

    Parameters:
    -----------
    Matrix : ndarray, shape (n, 2)
        2D array where each row represents a point in 2D space.
    angle : float, optional
        Angular range parameter. Points within this range are labeled as 0.
    Default is 0.2.

    Returns:
    --------
    ndarray, shape (n,)
        Array of labels (0 or 1) for each point in the input Matrix.
    """
    Vect_angle = np.arctan2(Matrix[:, 1], Matrix[:, 0])
    label = []
    for point_angle in Vect_angle:
        first_condition = 0 <= point_angle < angle * np.pi / 4
        second_condition = (1 - angle) * np.pi / 4 <= point_angle < np.pi / 4
        third_condition = (
            np.pi / 4 + angle * np.pi / 4 <= point_angle <
            np.pi / 4 + (1 - angle) * np.pi / 4
        )

        if first_condition or second_condition or third_condition:
            label.append(0)
        else:
            label.append(1)
    return np.array(label)


# #######################################
# ## From data_generation: end
# #######################################


# #########################
# from supervised_classif.py : begin
# #########################

# ## 1. toy example with  hard decision boundary in 2D
# all data follow the tail structure (angular decision boundary)
# no bias variance compromise in the choice of k


# Parameters for data generation
n = 1000
Dim = 2
split = 0.2
alpha_dep = 0.9
angle = 0.25

# Data generation
data = mlx.gen_multilog(Dim, alpha_dep, size=n)**(1/4)
label = mlx.gen_label2(data, angle=angle)

# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue')
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
plt.show()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                    test_size=split,
                                                    random_state=42)

# Choice of an off-the-shelf classification algorithm
# see https://scikit-learn.org/stable/supervised_learning.html

# Pick  a classifier model in sklearn, previously imported
model = RandomForestClassifier()

# Normalization function
def norm_func(x):
    return np.linalg.norm(x, ord=2, axis=1)


# Classifier class initialization
classifier = mlx.Classifier(model, norm_func)

# Model training
threshold, ratio, X_train_extreme = classifier.fit(X_train,  y_train, k=200)

# predict above a larger threshold (extrapolation)
ratio_ext = ratio / 2
Norm_test = norm_func(X_test)
thresh_predict = np.quantile(Norm_test, 1- ratio_ext)

# Prediction on the test data
y_pred_extreme,  X_test_extreme, mask_test = classifier.predict(
                                            X_test, thresh_predict)

# Accuracy evaluation
y_test_extreme = y_test[mask_test]
accuracy = accuracy_score(y_test_extreme, y_pred_extreme)
print(f'Accuracy: {accuracy:.4f}')
hamming = hamming_loss(y_test_extreme, y_pred_extreme)
print(f'0-1 loss: {hamming:.4f}')

# Display classification results
# show X
classifier.plot_classif(X_test_extreme, y_test_extreme, y_pred_extreme)

# show Theta
X_test_extrem_unit = X_test_extreme / norm_func(X_test_extreme)[:, np.newaxis]
classifier.plot_classif(X_test_extrem_unit, y_test_extreme, y_pred_extreme)


# 2. generating dirichlet mixtures ##
Mu0 = np.array([[0.2, 0.8], [0.8, 0.2]])  # Means of the two components
lnu0 = np.log(10) * np.ones(2)  # log(10) for each component
wei0 = np.array([0.1, 0.9])  # Mixture weights

# # class 0: dirichlet parameters
# Mu0 = np.array([[0.2, 0.8], [0.8, 0.2]])  # Means of the two components
# wei0 = np.array([0.9, 0.1])  # Mixture weights
# lnu0 = np.log(10 / np.min(Mu0, axis=1)) # log concentration for each component
# Plot the mixture density
mlx.plot_pdf_dirimix_2D(Mu0, wei0, lnu0)
n0=5000
n1=n0

data, label = mlx.gen_classif_data_diriClasses(Mu0, wei0, lnu0,
                                               size0=n0, size1=n1)
# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue').flatten()
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
plt.show()



# #########################
# from supervised_classif.py - end
# #########################

# ############################
# from prediction_missing_component : begin
# ##########################

id_extreme_pred = (norm_X >= thresh_predict)
X_extreme = X[id_extreme_pred, :]
y1_extreme = y1[id_extreme_pred]
y2_extreme = y2[id_extreme_pred]

r_ext = norm_X[id_extreme_pred]
plt.scatter(r_ext, y1_extreme, c='blue', alpha=0.5, label='y1_extreme')
plt.scatter(r_ext, y2_extreme, c='red', alpha=0.5, label='y2_extreme')
plt.scatter(r_ext, Z[id_extreme_pred], c='black', alpha=0.5, label='Z_extreme')
# Add title and legend
plt.title('Independence of y1/y2/Z (for ||x|| large) versus Radius ||x||')
plt.legend()


test1 = spearmanr(r_ext, y1_extreme)
test2 = spearmanr(r_ext, y2_extreme)
print(f'spearman correl test, radius versus y1:\n \
correl: {test1.statistic} , p-value : {test1.pvalue}')
print(f'spearman correl  test, radius versus y2: \n \
correl: {test2.statistic} , p-value : {test2.pvalue}')
test_naive = spearmanr(r_ext, Z[id_extreme_pred])
print(f'correl test, radius versus z: \n \
correl: {test_naive.statistic} , p-value : {test_naive.pvalue}')

def statistic(x, y):
    return spearmanr(x, y).statistic
test1_perm = permutation_test(data=[r_ext, y1_extreme], statistic=statistic,
                              permutation_type='pairings')
bin_y1 = (y1_extreme > 0.28).astype(int)
dummy_test1 = permutation_test(data=[r_ext, bin_y1], statistic= statistic,
                               permutation_type='pairings')

                        
test2_perm = permutation_test(r_ext, y2_extreme)
print(f'permutation correl test, radius versus y1:\n \
correl: {test1.statistic} , p-value : {test1.pvalue}')
print(f'permutation correl  test, radius versus y2: \n \
correl: {test2.statistic} , p-value : {test2.pvalue}')
test_naive_perm = permutation_test(r_ext, Z[id_extreme_pred])
print(f'permutation correl test, radius versus z: \n \
correl: {test_naive_perm.statistic} , p-value : {test_naive_perm.pvalue}')

plt.show()

# Conclusion: above the prediction test, pearson correlation test
# does not detect significant dependence between radius and transformed target.
# However of course it does so with the original target

# ###############################
# independence test k/target:

#    pvalue : str, optional
        # 'asymptotic' or 'permutation'. Default is 'asymptotic', mainly
        # because 'permutation' is computationally intensive. 'permutation'
        # is recommended only for n*ratio_ext < 50.


# if (n*np.max(ratio_ext) > 500)  and pvalue == 'permutation':
#     # Print a warning at the beginning of execution
#     warnings.warn(
#         "The permutation test is computationally intensive and may take a \
#         long time to complete. Consider using  default `pvalue='asymptotic'",
#         UserWarning
#     )

# if (n*np.min(ratio_ext) < 500)  and pvalue == 'asymptotic':
#     # Print a warning at the beginning of execution
#     warnings.warn(
#         "The asymptotic p-value may be unreliable for 'ratio_ext * n < 500'.\
#         Consider using `pvalue='permutation'` ",
#         UserWarning
#     )
# , pvalue='asymptotic'):
def test_indep_target_radius(X, y, ratio_ext, norm_func):
    """Performs a Spearman correlation test between y_i's such that ||X_i||
    exceeds its empirical quantile (1-ratio_ext).

    This test helps determine whether the regular variation assumptions for
    the prediction model are satisfied, for ratio_ext sufficiently small.
    When training a prediction model on extreme covariates (classification
    or regression), it is recommended to perform this independence test for
    ratio_ext=k_train, where k_train is the number of extremes used to train
    the prediction model.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Covariates.
    y : 1D array, shape (n,)
        Targets.
    norm_func : callable
        The norm functional to measure extremality of samples in X.
    ratio_ext : 1D array, size m
        The ratio of (extreme) observations considered for the test.
    
    Returns
    -------
    A dictionary with entries: 
        statistics : ndarray
            The test statistics for each ratio in ratio_ext.
        pvalues : ndarray
            The p-values for each ratio in ratio_ext.
        upper_accept, lower_accept: ndarrays:
            upper and lower bounds delimitating the  'accept' region
            (at confidence level 0.95) around zero for each ratio in ratio_ext.

    Details
    _______________
    The upper and low boundaries of the non-rejection regions are computed
    using the Fisher transform method for n*ratio_ext >= 30.
    If n*ratio_ext < 30 a permutation method is used instead, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html.

    """
    if isinstance(ratio_ext, float):
        ratio_ext = [ratio_ext]
    statistics = []
    pvalues = []
    lower_accept = []
    upper_accept = []
    norm_X = norm_func(X)

    def statistic(a, b):
        return stats.spearmanr(a, b).statistic

    for ratio in ratio_ext:
        threshold = np.quantile(norm_X, 1-ratio)
        id_extreme = (norm_X >= threshold)
        y_extreme = y[id_extreme]
        r_ext = norm_X[id_extreme]
        if len(y_extreme) < 30:   # pvalue == 'permutation':
            test = stats.permutation_test(data=[r_ext, y_extreme],
                                          statistic=statistic,
                                          permutation_type='pairings')
            upp = np.quantile(test.null_distribution, 0.975)
            low = np.quantile(test.null_distribution, 0.025)
            upper_accept.append(upp)
            lower_accept.append(low)
        else:
            test = stats.spearmanr(r_ext, y_extreme)
            halfw = 1.96 * np.sqrt(1.06 / (len(y_extreme) - 3))
            lower_accept.append(-halfw)
            upper_accept.append(halfw)

        statistics.append(round_signif(test.statistic, 2))
        pvalues.append(round_signif(test.pvalue, 2))

    print(f'spearman statistics: {statistics}')
    print(f'spearman p-values: {pvalues}')
    return {
        'ratio_ext': ratio_ext,
        'n': len(norm_X),
        'statistics': np.array(statistics),
        'pvalues': np.array(pvalues),
        'lower_accept': np.array(lower_accept),
        'upper_accept': np.array(upper_accept)
        }


def plot_indep_target_radius(test_indep):
    """
    test_indep is a dictionary returned by `test_indep_target_radius`
    """
    statistics = test_indep['statistics']
    lower = test_indep['lower_accept']
    upper = test_indep['upper_accept']
    ratios = test_indep['ratio_ext']
    n  = test_indep['n']  
    kk = (ratios * n).astype(int)
    fig, ax = plt.subplots()
    ax.fill_between(kk, lower, upper, color='green', alpha=0.5,
                    label='non-rejection region')
    #colors = ['green' if p > 0.05 else 'red' for p in pvalues1]
    ax.scatter(kk, statistics, c='black', label='Spearman statistic')
    # Add a secondary x-axis with a different scale
    def ratio_transform(x):
        return x / n

    def inverse_ratio_transform(x):
        return x * n
    
    secax = ax.secondary_xaxis('top',
                               functions=(ratio_transform,
                                          inverse_ratio_transform))
    secax.set_xlabel('Ratio (k / n)')
    ax.set_title('Spearman correlation test for extreme norm(X) Values vs Target')
    ax.set_xlabel('k')
    ax.set_ylabel('Spearman Statistic')
    ax.legend()
    plt.grid()
    plt.show()



# ######################
# from supervised_regression : cv on rank transform
# %%
# CV on rank transformed?

# %%
ratio_train_vect = np.geomspace(0.005, 0.2, num=10)
k_train_vect = (n_train * ratio_train_vect).astype(int)
thresh_train_vect = np.array([np.quantile(norm_X_train, 1 - r)
                              for r in ratio_train_vect])
regressor = mlx.Regressor(model, norm_func)
kscores = []
kscores_sd = []
count = 1
# cv-loop (time consuming)
for thresh in thresh_train_vect:
    count+=1
    mean_scores, sd_mean_scores, _ = regressor.cross_validate(
        X_train_rt, y_train, thresh_train=thresh, thresh_predict=thresh_predict,
        scoring=mean_squared_error,
        random_state=42 + 103*count)
    kscores.append(mean_scores)
    kscores_sd.append(sd_mean_scores)

kscores = np.array(kscores)
kscores_sd = np.array(kscores_sd)
plt.plot(k_train_vect, kscores)
plt.fill_between(k_train_vect, kscores + 1.64 * kscores_sd,
                 kscores - 1.64 * kscores_sd, color='blue', alpha=0.2)
plt.show()

# %%
i_opt = np.argmin(kscores)
k_opt_rt = k_train_vect[i_opt]
print(f'optimal k with cv on rank transformed: {k_opt_rt}')
print(f'recall optimal k with cv, standard case: {k_opt}')


# %%
# Retraining with k_opt_rt
# Model training
regressor = mlx.Regressor(model, norm_func)
threshold, ratio, X_train_extreme = regressor.fit(X_train_rt,  y_train,
                                                  k=k_opt_rt)
# Prediction on the test data
y_pred_extreme_cv_rt,  X_test_extreme_rt, _ = regressor.predict(
    X_test_rt, thresh_predict)

# mean_squared_error evaluation
mse_cv_rt = mean_squared_error(y_test_extreme_rt, y_pred_extreme_cv_rt)
print(f'mse rank-tranformed after cv: {mse_cv_rt:.4f}')
print(f'mse rank-transfored before cv: {mse_rt:.4f}')

# Not convincing 


# ##### from regression.py:
       
    # def evaluate(self,y_true, y_pred):
    #     """
    #     Evaluate the regression model using Mean Squared Error.

    #     Parameters
    #     ----------
    #     y_true : array-like of shape (n_samples,)
    #         The true values.

    #     y_pred : array-like of shape (n_samples,)
    #         The predicted values.

    #     Returns
    #     -------
    #     mse : float
    #         The mean squared error of the predictions.
    #     """
    #     return mean_squared_error(y_true, y_pred)

    # def cross_validate(self, X, y, cv=5, scoring='neg_mean_squared_error'):
    #     """
    #     Perform cross-validation and return the mean score.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #         The input samples.

    #     y : array-like of shape (n_samples,)
    #         The target values.

    #     cv : int, optional
    #         The number of folds for K-fold cross-validation.

    #     scoring : str, optional
    #         The scoring metric for cross-validation.

    #     Returns
    #     -------
    #     mean_score : float
    #         The mean score from cross-validation.
    #     """
    #     scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
    #     return np.mean(scores)
