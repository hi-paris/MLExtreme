import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from . import utilities as ut
from ...utils.EVT_basics import rank_transform, round_signif
#import pdb

# ################
# ## DAMEX CLASS
# ################


class damex:
    """
    An unsupervised feature clustering model to learn "which subsets of
    features of a random vector are likely to be simultaneously large, while
    the others are small".
    """

    def __init__(self, epsilon=0.1, min_counts=0, thresh_train=None,
                 thresh_test=None,
                 include_singletons=False,  use_max_subfaces=False,
                 rate=10):
        """
        Initialize the DAMEX model with specified parameters.

        Parameters:
        - epsilon (float): Tolerance level for clustering.
        - min_counts (int): Minimum number of points required to form
            a cluster.
        - thresh_train (float): Threshold for training data.
        - thresh_test (float): Threshold for test data.
        - include_singletons (bool): Whether to include singletons in
              training and testing.
        - use_max_subfaces (Bool): whether to consider only maximal faces
        for inclusion when computing deviance scores.
        - rate (float, >0): rate parameter for the dispersion model
            on subfaces.
        """
        self.epsilon = epsilon
        self.min_counts = min_counts
        self.thresh_train = thresh_train
        self.thresh_test = thresh_test
        self.include_singletons = include_singletons
        self.use_max_subfaces = use_max_subfaces
        self.rate = rate
        self.subfaces = None  # Identified subfaces
        self.masses = None  # Masses associated with subfaces
        self.maximal_subfaces = None  # Maximal subfaces
        self.maximal_masses = None  # Masses of maximal subfaces
        self.total_mass = None  # Total mass of extremes
        self.number_extremes = None  # Number of extreme points
        self.dimension = None  # Dimensionality of the data

    def fit(self, X, epsilon=None,
            standardize=True):
        """
        Fit the DAMEX model to the data.

        Parameters:
        - X (np.ndarray): Input data.
        - threshold (float): Threshold for identifying extremes.
        - epsilon (float): Tolerance level for clustering.
        - min_counts (int): Minimum number of points required to form a
          cluster.
        - standardize (bool): Whether to standardize the data.
        - include_singletons (bool): Whether to include singletons.

        Returns:
        - Subfaces (list): Identified subfaces (or maximal subfaces in case
        self.use_max_subfaces is True).
        - Masses (list): Masses associated with subfaces.
        """
        # Update instance attributes with optional arguments passed
        if epsilon is not None:
            self.epsilon = epsilon

        # Record dimension
        self.dimension = np.shape(X)[1]

        # Standardize X if needed
        Xt = rank_transform(X) if standardize else X
        norm_Xt = np.max(Xt, axis=1)

        # Set the training threshold if not provided
        if self.thresh_train is None:
            self.thresh_train = np.quantile(
                norm_Xt, 1 - 1 / np.sqrt(len(norm_Xt)))

        # Update total mass accordingly
        self.number_extremes = np.sum(norm_Xt > self.thresh_train)
        self.total_mass = self.thresh_train * \
            self.number_extremes / len(norm_Xt)

        # Fit the model
        Subfaces, Masses = ut.damex_fit(
            Xt, self.thresh_train, self.epsilon, self.min_counts,
            standardize=False, include_singletons=self.include_singletons)

        # Update instance's attributes
        self.subfaces = Subfaces
        self.masses = Masses

        # compute maximal subfaces and update model
        _, _ = self.construct_maximal_subfaces(Xt, standardize=False)

        if self.use_max_subfaces:
            return self.maximal_subfaces, self.maximal_masses
        else:
            return Subfaces, Masses
    

    def construct_maximal_subfaces(self, X, standardize=True):
        """
        Construct maximal subfaces from the fitted model.

        Parameters:
        - X (np.ndarray): Input data.
        - standardize (bool): Whether to standardize the data.

        Returns:
        - dict_max_faces (dict): Dictionary of maximal subfaces.
        - dict_max_masses (dict): Dictionary of masses associated with maximal
          subfaces.
        """
        if self.subfaces is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' "
                               "with appropriate arguments before using this "
                               "method.")

        faces_dict, _ = ut.list_to_dict(self.subfaces, self.masses)
        self.maximal_subfaces = ut.find_maximal_faces(faces_dict, lst=True)
        self.maximal_masses = ut.estim_subfaces_mass(self.maximal_subfaces,
                                                     X, self.thresh_train,
                                                     self.epsilon,
                                                     standardize=standardize)

        dict_max_faces, dict_max_masses = ut.list_to_dict(
            self.maximal_subfaces, (self.maximal_masses).tolist())
        return dict_max_faces, dict_max_masses

    def deviance(self, Xtest, thresh_test=None, standardize=False):
        """
        Calculate the deviance of the model on test data.

        Parameters:
        - Xtest (np.ndarray): Test data.
        - use_max_subfaces (bool): Whether to use maximal subfaces.
        - include_singletons_train (bool): Whether to include singletons from
          training.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        - threshold (float): Threshold for identifying extremes.
        - rate (float, >0): Rate parameter for deviance calculation.
        - standardize (bool): Whether to standardize the data.

        Returns:
        - float: Deviance value.
        """
        if self.subfaces is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' "
                               "with appropriate arguments before using this "
                               "method.")

        Xt = rank_transform(Xtest) if standardize else Xtest

        # update test threshold if passed as argument
        if thresh_test is not None:
            self.thresh_test = thresh_test
        # set thresh_test to thresh_train if not previously specified
        if self.thresh_test is None:
            self.thresh_test = self.thresh_train

        Subfaces = self.maximal_subfaces if self.use_max_subfaces \
            else self.subfaces
        Masses = self.maximal_masses if self.use_max_subfaces \
            else self.masses

        negative_pseudo_lkl = ut.setDistance_subfaces_data(
            Subfaces, self.thresh_test, Xt,
            self.include_singletons, self.epsilon, Masses,
            self.total_mass, dispersion_model=True, rate=self.rate)

        return 2 * negative_pseudo_lkl

    def get_AIC(self, Xtrain,  standardize=True):
        """
        Calculate the Akaike Information Criterion (AIC) for the model.

        Parameters:
        - Xtrain (np.ndarray): Training data.
        - use_max_subfaces (bool): Whether to use maximal subfaces.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        - rate (float, >0): Rate parameter for deviance calculation.
        - standardize (bool): Whether to standardize the data.

        Returns:g
        - float: AIC value.
        """
        if self.masses is None:
            raise RuntimeError("Fit the model before computing the AIC")
        
        Masses = self.maximal_masses if self.use_max_subfaces \
            else self.masses
        intern_deviance = self.deviance(Xtrain, self.thresh_train, standardize)
        return intern_deviance + 2 * len(Masses) / self.number_extremes

    def select_epsilon_AIC(self, grid, X, standardize=True,
                           unstable_eps_max=0.05, plot=False,
                           update_epsilon=False):
        """
        Select the optimal epsilon value based on AIC.

        Parameters:
        - grid (list): Grid of epsilon values to test.
        - X (np.ndarray): Input data.
        - standardize (bool): Whether to standardize the data.
        - unstable_eps_max (float): Maximum epsilon value for unstable
          solutions.
        - min_counts (int): Minimum number of points required to form a
          cluster.
        - use_max_subfaces (bool): Whether to use maximal subfaces.
        - thresh_train (float): Threshold for training data.
        - thresh_test (float): Threshold for test data.
        - include_singletons_train (bool): Whether to include singletons in
          training.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        - rate (float, >0): Rate parameter for deviance calculation.
        - plot (bool): Whether to plot the AIC values.
        - update_epsilon (bool): Whether to update the model's epsilon value.

        Returns:
        - eps_select_aic (float): Selected epsilon value.
        - aic_opt (float): Optimal AIC value.
        - vect_aic (np.ndarray): Vector of AIC values.
        """
        old_epsilon = deepcopy(self.epsilon)
        Xt = rank_transform(X) if standardize else X
        ntests = len(grid)
        vect_aic = np.zeros(ntests)

        for counter, eps in enumerate(grid):
            _, _ = self.fit(Xt,  eps, False)
            vect_aic[counter] = self.get_AIC(Xt, standardize=False)

        i_maxerr = np.argmax(vect_aic[grid < unstable_eps_max])
        eps_maxerr = grid[i_maxerr]
        i_mask = grid <= eps_maxerr
        i_minAIC = np.argmin(vect_aic + (1e+23) * i_mask)
        eps_select_aic = grid[i_minAIC]
        aic_opt = vect_aic[i_minAIC]

        if plot:
            print(f'DAMEX: selected epsilon with AIC criterion '
                  f'{round_signif(eps_select_aic, 2)}')
            plt.figure(figsize=(10, 5))
            plt.xlabel('epsilon')
            plt.ylabel('AIC')
            plt.title('DAMEX- AIC versus epsilon')
            plt.scatter(grid, vect_aic, c='gray', label='AIC')
            plt.plot([eps_select_aic, eps_select_aic], [0, max(vect_aic)],
                     c='red')
            plt.grid(True)
            plt.show()

        # Reset instance's epsilon value with initial or selected value
        if update_epsilon:
            self.epsilon = eps_select_aic
        else:
            self.epsilon = old_epsilon

        # Re-fit the model with initial or final epsilon
        _, _ = self.fit(Xt, self.epsilon, False)

        return eps_select_aic, aic_opt, vect_aic

    def deviance_CV(self, X, standardize=True, epsilon=None,
                    thresh_test=None, cv=5, random_state=None):
        """Calculate the deviance using cross-validation.

        Parameters:
        
        - X (np.ndarray): Input data.
        
        - standardize (bool): Whether to standardize the data. default to True.
        
        - epsilon (float): Tolerance level for clustering. Default to
          None. If None, the instance's attribute will be used.
        
        - thresh_test (float, optional): radial threshold for test
          sets. If None, the instance's attribute will be used. If the
          latter is also None, the training threshold is used instead.
        
        - cv (int): Number of cross-validation folds. Default to 5
        
        - random_state (int): Random seed for reproducibility. Default to None

        Returns:
        
        - np.ndarray: Cross-validated deviance scores.

        """
        Xt = rank_transform(X) if standardize else X

        # Update instance's attributes if passed as arguments
        if epsilon is not None:
            self.epsilon = epsilon
        if thresh_test is not None:
            self.thresh_test = thresh_test
        if self.thresh_test is None:
            self.thresh_test = self.thresh_train
        
        cv_neglkl_scores = ut.ftclust_cross_validate(
            Xt, standardize=False, algo='damex', tolerance=self.epsilon,
            min_counts=self.min_counts, use_max_subfaces=self.use_max_subfaces,
            thresh_train=self.thresh_train, thresh_test=self.thresh_test,
            include_singletons_train=self.include_singletons,
            include_singletons_test=self.include_singletons,
            rate=self.rate, cv=cv, random_state=random_state)

        return 2 * cv_neglkl_scores

    def select_epsilon_CV(self, grid, X, standardize=True,
                          update_epsilon=False,
                          unstable_tol_max=0.05, thresh_test=None,  cv=5,
                          random_state=None, plot=False):
        """
        Select the optimal epsilon value based on cross-validation.

        Parameters:
        - grid (list): Grid of epsilon values to test.
        - X (np.ndarray): Input data.
        - standardize (bool): Whether to standardize the data.
        - update_epsilon (bool): Whether to update the model's epsilon value.
        - unstable_tol_max (float): Maximum tolerance value for unstable
          solutions.
        - cv (int): Number of cross-validation folds.
        - random_state (int): Random seed for reproducibility.
        - plot (bool): Whether to plot the CV deviance values.


        Returns:
        - tol_cv (float): Selected epsilon value.
        - deviance_tol_cv (float): Deviance value for the selected epsilon.
        - cv_deviance_vect (np.ndarray): Vector of CV deviance values.
        """
        Xt = rank_transform(X) if standardize else X
        epsilon_old = deepcopy(self.epsilon)
        ntol = len(grid)
        cv_deviance_vect = np.zeros(ntol)

        for i, eps in enumerate(grid):
            cv_scores = self.deviance_CV(Xt, False, eps, thresh_test, cv,
                                         random_state)
            cv_deviance_vect[i] = np.mean(cv_scores)

        i_maxerr = np.argmax(cv_deviance_vect[grid < unstable_tol_max])
        tol_maxerr = grid[i_maxerr]
        i_mask = grid <= tol_maxerr
        i_cv = np.argmin(cv_deviance_vect + (1e+23) * i_mask)
        tol_cv = grid[i_cv]
        deviance_tol_cv = cv_deviance_vect[i_cv]

        if plot:
            plt.scatter(grid, cv_deviance_vect, c='gray', label='CV deviance')
            plt.plot([tol_cv, tol_cv], [0, deviance_tol_cv], c='red',
                     label='selected value')
            plt.grid()
            plt.legend()
            plt.title("DAMEX: clustering pseudo-deviance, K-fold CV scores")
            plt.show()

        if update_epsilon:
            self.epsilon = tol_cv
        else:
            self.epsilon = epsilon_old

        # Re-fit the model with final epsilon
        _, _ = self.fit(Xt, False)

        return tol_cv, deviance_tol_cv, cv_deviance_vect

    def deviance_to_true(self, subfaces_true, weights_true):
        """
        Calculate the deviance between the estimated subfaces and the true
        subfaces.

        Parameters:
        - subfaces_true (list): True subfaces.
        - weights_true (list): Weights of the true subfaces.
        - use_max_subfaces (bool): Whether to use maximal subfaces.
        - rate (float, >0): Rate parameter for deviance calculation.

        Returns:
        - est_to_truth (float): Deviance from estimated to true subfaces.
        - truth_to_est (float): Deviance from true to estimated subfaces.
        """
        
        if self.subfaces is None:
            raise RuntimeError("DAMEX has not been fitted yet")

        if self.use_max_subfaces:
            if self.maximal_subfaces is None:
                raise RuntimeError("Construct maximal subfaces first")
            Subfaces = self.maximal_subfaces
            Masses = self.maximal_masses
        else:
            Subfaces = self.subfaces
            Masses = self.masses

        if isinstance(Masses, list):
            Masses = np.array(Masses)
            
        if isinstance(weights_true, list):
            weights_true = np.array(weights_true)

        Subfaces_matrix = ut.subfaces_list_to_matrix(
            Subfaces, self.dimension)
        Subfaces_true_matrix = ut.subfaces_list_to_matrix(
            subfaces_true, self.dimension)

        if not self.include_singletons:
            id_keep_estim = np.where(np.sum(Subfaces_matrix, axis=1) >= 2)[0]
            id_keep_true = np.where(np.sum(Subfaces_true_matrix,
                                           axis=1) >= 2)[0]
            Subfaces_matrix = Subfaces_matrix[id_keep_estim]
            Masses = Masses[id_keep_estim]
            Subfaces_true_matrix = Subfaces_true_matrix[id_keep_true]
            weights_true = weights_true[id_keep_true]
            
        est_to_truth = 2 * ut.setDistance_error_m2m(
            Subfaces_matrix, Masses, Subfaces_true_matrix, weights_true, 1,
            True, self.rate)
        truth_to_est = 2 * ut.setDistance_error_m2m(
            Subfaces_true_matrix, weights_true, Subfaces_matrix, Masses,
            self.total_mass, True, self.rate)
        
        return est_to_truth, truth_to_est
        
        # est_to_truth = 2 * ut.setDistance_error_l2l(
        #     Subfaces, Masses, subfaces_true, weights_true, 1, self.dimension,
        #     True, rate)

        # truth_to_est = 2 * ut.setDistance_error_l2l(
        #     subfaces_true, weights_true, Subfaces, Masses, self.total_mass,
        #     self.dimension, True, rate)

        # return est_to_truth, truth_to_est
