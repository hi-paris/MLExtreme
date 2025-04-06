
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from . import utilities as ut  # #import binary_large_features
from ...utils.EVT_basics import rank_transform, round_signif

# import itertools as it
# import numpy as np
# import matplotlib.pyplot as plt
# import itertools as it
# import networkx as nx
import pdb
# from . import utilities as ut
# from ...utils.EVT_basics import rank_transform, round_signif
# # from .damex import entropy, AIC_clustering
# # from . import ftclust_analysis as fca

class clef:
    def __init__(self, kappa_min=0.1, thresh_train=None,
                 thresh_test=None, include_singletons_train=True,
                 include_singletons_test=True):
        """
        Initialize the DAMEX model with specified parameters.

        Parameters:
        - kappa_min (float): Tolerance level for clustering.
        - min_counts (int): Minimum number of points required to form a cluster.
        - thresh_train (float): Threshold for training data.
        - thresh_test (float): Threshold for test data.
        - include_singletons_train (bool): Whether to include singletons in
          training.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        """
        self.kappa_min = kappa_min
        self.thresh_train = thresh_train
        self.thresh_test = thresh_test
        self.include_singletons_train = include_singletons_train
        self.include_singletons_test = include_singletons_test
        self.subfaces = None  # Identified subfaces
        self.masses = None  # Masses associated with subfaces
        self.total_mass = None  # Total mass of extremes
        self.number_extremes = None  # Number of extreme points
        self.dimension = None  # Dimensionality of the data

    def fit(self, X, threshold=None, kappa_min=None,
            standardize=True, include_singletons=None):
        """
        Fit the CLEF model to the data.

        Parameters:
        - X (np.ndarray): Input data.
        - threshold (float): Threshold for identifying extremes.
        - kappa_min (float): Tolerance level for clustering.
        - standardize (bool): Whether to standardize the data.
        - include_singletons (bool): Whether to include singletons.

        Returns:
        - Subfaces (list): Identified subfaces.
        - Masses (list): Masses associated with subfaces.
        """
        # Update instance attributes with optional arguments passed
        if kappa_min is not None:
            self.kappa_min = kappa_min
        if threshold is not None:
            self.thresh_train = threshold
        if include_singletons is not None:
            self.include_singletons_train = include_singletons

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
        Subfaces = ut.clef_fit(
            Xt, self.thresh_train, self.kappa_min, 
            standardize=False,
            include_singletons=self.include_singletons_train)
        Masses = ut.estim_subfaces_mass(
            Subfaces, Xt, self.thresh_train, epsilon=None,
            standardize=False)
        # Update instance's attributes
        self.subfaces = Subfaces
        self.masses = Masses

        return Subfaces, Masses

    def deviance(self, Xtest, 
                 #include_singletons_train=False,
                 include_singletons_test=None,
                 threshold=None, rate=10, standardize=False):
        """
        Calculate the deviance of the model on test data.

        Parameters:
        - Xtest (np.ndarray): Test data.
        - remove_singletons_train (bool): Whether to remove singletons from
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

        # Update instance's attributes if passed as arguments
        if threshold is not None:
            self.thresh_test = threshold
        if self.thresh_test is None:
            self.thresh_test = self.thresh_train
        if include_singletons_test is not None:
            self.include_singletons_test = include_singletons_test

        Subfaces = self.subfaces
        Masses = self.masses

        negative_pseudo_lkl = ut.setDistance_subfaces_data(
            Subfaces, self.thresh_test, Xt, #remove_singletons_train,
            self.include_singletons_test, None, Masses,
            self.total_mass, dispersion_model=True, rate=rate)

        return 2 * negative_pseudo_lkl

    
    def get_AIC(self, Xtrain, 
                # remove_singletons_train=False,
                include_singletons_test=None,
                rate=10, standardize=True):
        """
        Calculate the Akaike Information Criterion (AIC) for the model.

        Parameters:
        - Xtrain (np.ndarray): Training data.
        - remove_singletons_train (bool): Whether to remove singletons from
          training.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        - rate (float, >0): Rate parameter for deviance calculation.
        - standardize (bool): Whether to standardize the data.

        Returns:
        - float: AIC value.
        """
        if self.masses is None:
            raise RuntimeError("Fit the model before computing the AIC")

        Masses =  self.masses
        intern_deviance = self.deviance(Xtrain,
                                        # remove_singletons_train,
                                        include_singletons_test, None, rate,
                                        standardize)
        return intern_deviance + 2 * len(Masses) / self.number_extremes

    def select_kappa_min_AIC(self, grid, X, standardize=True,
                             unstable_kappam_max=0.05, 
                             thresh_train=None,
                             thresh_test=None, include_singletons_train=None,
                             include_singletons_test=None, rate=10, plot=False,
                             update_kappa_min=True):
        """
        Select the optimal kappa_min value based on AIC.

        Parameters:
        - grid (list): Grid of kappa_min values to test.
        - X (np.ndarray): Input data.
        - standardize (bool): Whether to standardize the data.
        - unstable_kappam_max (float): Maximum kappa_min value for unstable
          solutions.
        - thresh_train (float): Threshold for training data.
        - thresh_test (float): Threshold for test data.
        - include_singletons_train (bool): Whether to include singletons in
          training.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        - rate (float, >0): Rate parameter for deviance calculation.
        - plot (bool): Whether to plot the AIC values.
        - update_kappa_min (bool): Whether to update the model's kappa_min value.

        Returns:
        - kappam_select_aic (float): Selected kappa_min value.
        - aic_opt (float): Optimal AIC value.
        - vect_aic (np.ndarray): Vector of AIC values.
        """
        old_kappa_min = deepcopy(self.kappa_min)
        Xt = rank_transform(X) if standardize else X
        ntests = len(grid)
        vect_aic = np.zeros(ntests)

        for counter, kapp in enumerate(grid):
            subfaces, masses = self.fit(
                Xt, thresh_train, kapp, standardize=False,
                include_singletons=include_singletons_train)
            vect_aic[counter] = self.get_AIC(
                Xt,
                # not self.include_singletons_train,
                include_singletons_test, rate,
                standardize=False)

        i_maxerr = np.argmax(vect_aic[grid < unstable_kappam_max])
        kapp_maxerr = grid[i_maxerr]
        i_mask = grid <= kapp_maxerr
        i_minAIC = np.argmin(vect_aic + (1e+23) * i_mask)
        kapp_select_aic = grid[i_minAIC]
        aic_opt = vect_aic[i_minAIC]

        if plot:
            print(f'CLEF: selected kappa_min with AIC criterion: '
                  f'{round_signif(kapp_select_aic, 2)}')
            plt.figure(figsize=(10, 5))
            plt.xlabel('kappa_min')
            plt.ylabel('AIC')
            plt.title('CLEF: AIC versus kappa_min')
            plt.scatter(grid, vect_aic, c='gray', label='AIC')
            plt.plot([kapp_select_aic, kapp_select_aic], [0, max(vect_aic)],
                     c='red')
            plt.grid(True)
            plt.show()

        # Update instance's kappa_min value
        if update_kappa_min:
            self.kappa_min = kapp_select_aic
        else:
            self.kappa_min = old_kappa_min

        # Re-fit the model with final kappa_min
        _, _ = self.fit(Xt, thresh_train, self.kappa_min,
                        False,
                        include_singletons_train)

        return kapp_select_aic, aic_opt, vect_aic

    def deviance_CV(self, X, standardize=True, kappa_min=None,
                    include_singletons_train=None, include_singletons_test=None,
                    thresh_train=None, thresh_test=None, 
                    rate=10, cv=5, random_state=None):
        """
        Calculate the deviance using cross-validation.

        Parameters:
        - X (np.ndarray): Input data.
        - standardize (bool): Whether to standardize the data.
        - kappa_min (float): Tolerance level for clustering.
        - include_singletons_train (bool): Whether to include singletons in
          training.
        - include_singletons_test (bool): Whether to include singletons in
          testing.
        - thresh_train (float): Threshold for training data.
        - thresh_test (float): Threshold for test data.
        - rate (float, >0): Rate parameter for deviance calculation.
        - cv (int): Number of cross-validation folds.
        - random_state (int): Random seed for reproducibility.

        Returns:
        - np.ndarray: Cross-validated deviance scores.
        """
        Xt = rank_transform(X) if standardize else X
        norm_Xt = np.max(Xt, axis=1)

        # Update instance's attributes if passed as arguments
        if kappa_min is not None:
            self.kappa_min = kappa_min
        if thresh_train is not None:
            self.thresh_train = thresh_train
        if self.thresh_train is None:
            self.thresh_train = np.quantile(norm_Xt,
                                            1 - 1 / np.sqrt(len(norm_Xt)))
        if thresh_test is not None:
            self.thresh_test = thresh_test
        if self.thresh_test is None:
            self.thresh_test = self.thresh_train
        if include_singletons_train is not None:
            self.include_singletons_train = include_singletons_train
        if include_singletons_test is not None:
            self.include_singletons_test = include_singletons_test

        cv_neglkl_scores = ut.ftclust_cross_validate(
            Xt, standardize=False, algo='clef', tolerance=self.kappa_min,
            min_counts=None, use_max_subfaces=None,
            thresh_train=self.thresh_train, thresh_test=self.thresh_test,
            include_singletons_train=self.include_singletons_train,
            include_singletons_test=self.include_singletons_test,
            rate=rate, cv=cv, random_state=random_state)

        return 2 * cv_neglkl_scores

    def select_kappa_min_CV(self, grid, X, standardize=True,
                            unstable_tol_max=0.05, 
                            thresh_train=None,
                            thresh_test=None, include_singletons_train=None,
                            include_singletons_test=None, rate=10, cv=5,
                            random_state=None, plot=False,
                            update_kappa_min=True):
        """
        Select the optimal kappa_min value based on cross-validation.

        Parameters:
        - grid (list): Grid of kappa_min values to test.
        - X (np.ndarray): Input data.
        - standardize (bool): Whether to standardize the data.
        - unstable_tol_max (float): Maximum tolerance value for unstable
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
        - cv (int): Number of cross-validation folds.
        - random_state (int): Random seed for reproducibility.
        - plot (bool): Whether to plot the CV deviance values.
        - update_kappa_min (bool): Whether to update the model's kappa_min value.

        Returns:
        - tol_cv (float): Selected kappa_min value.
        - deviance_tol_cv (float): Deviance value for the selected kappa_min.
        - cv_deviance_vect (np.ndarray): Vector of CV deviance values.
        """
        Xt = rank_transform(X) if standardize else X
        kappa_min_old = deepcopy(self.kappa_min)
        ntol = len(grid)
        cv_deviance_vect = np.zeros(ntol)
        #pdb.set_trace()
        for i, kapp in enumerate(grid):
            cv_scores = self.deviance_CV(Xt, False, kapp,
                                         include_singletons_train,
                                         include_singletons_test, thresh_train,
                                         thresh_test,
                                         rate,
                                         cv, random_state)
            cv_deviance_vect[i] = np.mean(cv_scores)

        i_maxerr = np.argmax(cv_deviance_vect[grid < unstable_tol_max])
        tol_maxerr = grid[i_maxerr]
        i_mask = grid <= tol_maxerr
        i_cv = np.argmin(cv_deviance_vect + (1e+23) * i_mask)
        tol_cv = grid[i_cv]
        deviance_tol_cv = cv_deviance_vect[i_cv]

        if plot:
            print(f'CLEF: selected kappa_min with CV estimate of deviance: '
                  f'{round_signif(tol_cv, 2)}')
            plt.scatter(grid, cv_deviance_vect, c='gray', label='CV deviance')
            plt.plot([tol_cv, tol_cv], [0, deviance_tol_cv], c='red',
                     label='selected value')
            plt.grid()
            plt.legend()
            plt.title("CLEF: expected  deviance estimated with  K-fold CV")
            plt.show()

        if update_kappa_min:
            self.kappa_min = tol_cv
        else:
            self.kappa_min = kappa_min_old

        # Re-fit the model with final kappa_min
        _, _ = self.fit(Xt, thresh_train, self.kappa_min, False,
                        include_singletons_train)
        return tol_cv, deviance_tol_cv, cv_deviance_vect

    def deviance_to_true(self,  subfaces_true, weights_true,
                         include_singletons=None, rate=10):
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
            raise RuntimeError("CLEF has not been fitted yet")

        if include_singletons is None:
            include_singletons = self.include_singletons_test
        
        Subfaces_matrix = ut.subfaces_list_to_matrix(
            self.subfaces, self.dimension)
        Masses = self.masses
        if isinstance(self.masses, list):
            Masses = np.array(Masses)

        Subfaces_true_matrix = ut.subfaces_list_to_matrix(
            subfaces_true, self.dimension)
        if isinstance(weights_true, list):
            weights_true = np.array(weights_true)

        if not include_singletons:
            id_keep_estim = np.where(np.sum(Subfaces_matrix, axis=1) >= 2)[0]
            id_keep_true = np.where(np.sum(Subfaces_true_matrix,
                                           axis=1) >= 2)[0]
            Subfaces_matrix = Subfaces_matrix[id_keep_estim]
            Masses = Masses[id_keep_estim]
            Subfaces_true_matrix = Subfaces_true_matrix[id_keep_true]
            weights_true = weights_true[id_keep_true]
            
        est_to_truth = 2 * ut.setDistance_error_m2m(
            Subfaces_matrix, Masses, Subfaces_true_matrix, weights_true, 1,
            True, rate)
        truth_to_est = 2 * ut.setDistance_error_m2m(
            Subfaces_true_matrix, weights_true, Subfaces_matrix,
            Masses, self.total_mass, True, rate)
        
        return est_to_truth, truth_to_est
        # Subfaces = self.subfaces
        # Masses = self.masses
                
        # est_to_truth = 2 * ut.setDistance_error_l2l(
        #     Subfaces, Masses, subfaces_true, weights_true, 1, self.dimension,
        #     True, rate)

        # truth_to_est = 2 * ut.setDistance_error_l2l(
        #     subfaces_true, weights_true, Subfaces, Masses, self.total_mass,
        #     self.dimension, True, rate)

        # return est_to_truth, truth_to_est

