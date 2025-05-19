import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from . import utilities as ut  # #import binary_large_features
from MLExtreme.utils.EVT_basics import rank_transform, round_signif


class clef:
    """
    A class for performing clustering of large extreme features (CLEF) on a dataset.

    This class provides methods to fit the CLEF model to data, calculate deviance, compute the
    Akaike Information Criterion (AIC), select optimal kappa_min values based on AIC or
    cross-validation, and calculate the deviance between estimated and true subfaces.

    Parameters:
    -----------
    kappa_min : float, default=0.1
        Tolerance level for clustering.
    thresh_train : float, optional
        Threshold for training data.
    thresh_test : float, optional
        Threshold for test data.
    rate : float, default=10
        Rate parameter for the dispersion model on subfaces.

    Attributes:
    -----------
    subfaces : list
        Identified subfaces.
    masses : array-like
        Masses associated with subfaces.
    total_mass : float
        Total mass of extremes.
    number_extremes : int
        Number of extreme points.
    dimension : int
        Dimensionality of the data.
    """

    def __init__(self, kappa_min=0.1, thresh_train=None, thresh_test=None, rate=10):
        """
        Initialize the CLEF model with specified parameters.

        Parameters:
        -----------
        kappa_min : float, default=0.1
            Tolerance level for clustering.
        thresh_train : float, optional
            Threshold for training data.
        thresh_test : float, optional
            Threshold for test data.
        rate : float, default=10
            Rate parameter for the dispersion model on subfaces.
        """
        self.kappa_min = kappa_min
        self.thresh_train = thresh_train
        self.thresh_test = thresh_test
        self.rate = rate
        self.subfaces = None  # Identified subfaces
        self.masses = None  # Masses associated with subfaces
        self.total_mass = None  # Total mass of extremes
        self.number_extremes = None  # Number of extreme points
        self.dimension = None  # Dimensionality of the data

    def fit(self, X, kappa_min=None, standardize=True):
        """
        Fit the CLEF model to the data.

        Parameters:
        -----------
        X : np.ndarray
            Input data.
        kappa_min : float, optional
            Tolerance level for clustering. If None, the instance's attribute will be used.
        standardize : bool, default=True
            Whether to standardize the data.

        Returns:
        --------
        Subfaces : list
            Identified subfaces.
        Masses : np.ndarray
            Masses associated with subfaces.
        """
        # Update instance attributes with optional arguments passed
        if kappa_min is not None:
            self.kappa_min = kappa_min

        # Record dimension
        self.dimension = np.shape(X)[1]

        # Standardize X if needed
        Xt = rank_transform(X) if standardize else X
        norm_Xt = np.max(Xt, axis=1)

        # Set the training threshold if not provided
        if self.thresh_train is None:
            self.thresh_train = np.quantile(norm_Xt, 1 - 1 / np.sqrt(len(norm_Xt)))

        # Update total mass accordingly
        self.number_extremes = np.sum(norm_Xt > self.thresh_train)
        self.total_mass = self.thresh_train * self.number_extremes / len(norm_Xt)

        # Fit the model
        Subfaces = ut.clef_fit(
            Xt,
            self.thresh_train,
            self.kappa_min,
            standardize=False,
            include_singletons=False,
        )
        Masses = ut.estim_subfaces_mass(
            Subfaces, Xt, self.thresh_train, epsilon=None, standardize=False
        )
        # Update instance's attributes
        self.subfaces = Subfaces
        self.masses = Masses

        return Subfaces, Masses

    def deviance(self, Xtest, thresh_test=None, standardize=False):
        """Calculate the deviance of the (trained) model on test data, normalized by the extreme sample size.

        First the test data is transformed into a binary matrix, say
        `testBinary`. Each row corresponds to a sample point which
        norm exceeds `thresh_test`.  The binary entries indicates
        which coordinates of a given extreme point exceed the quantity
        `epsilon * thresh_test`.

        Then the subfaces previously learnt by the model (stored in
        `self.subfaces`), together with the associated masses, are
        compared to `testBinary` in a a dispersion model.

        The basic building block of the dispersion model is a unit
        deviance d(subface_i, subface_j) defined between subfaces, as
        the ratio between the cardinalities of the symmetric
        difference and the union of the two sets of coordinates
        defining the suface, see
        :func:`unsupervised.feature_clustering.utilities.unit_deviance`. This is the same unit deviance as the one proposed in Reference [4]  below 

        The total (normalized) deviance is then defined as
        
        .. math::

            \\frac{- 2}{n_{extremes}}\\sum_{i=1}^{n_{extremes}} \
            \\log\\Big( \\sum_{j=1}^{n_{faces}} \
            w_j \\exp(- rate *d(testBinary_i, subface_j) ) \\Big), 


        where `n_{extremes}` is the number of samples which radius
        exceeds `thresh_test` in `Xtest`, rate=`self.rate`. It may be seen as

        "Twice the negative pseudo-likelihood in a dispersion model à la Jorgensen, divided by the extreme sample size"

        References:
        ------------
        [1] Cordeiro, G. M., Labouriau, R., & Botter, D. A. (2021). \
        An introduction to Bent Jørgensen’s ideas. Brazilian Journal of Probability and Statistics, 35(1), 2-20.

        [2] Jørgensen, B. (1987). Exponential dispersion models. \
        Journal of the Royal Statistical Society Series B: Statistical Methodology, \
        49(2), 127-145.

        [3] Jørgensen, B. (1997). The theory of dispersion models. CRC Press.

        [4] Chiapino, M., & Sabourin, A. (2016, September). Feature clustering for extreme events analysis, with application to extreme stream-flow data. In International workshop on new frontiers in mining complex patterns (pp. 132-147). Cham: Springer International Publishing.

        Parameters:
        -----------
        Xtest : np.ndarray
            Test data.
        thresh_test : float, optional
            Threshold for identifying extremes in the test set.
            If None, the instance's attribute will be used.
        standardize : bool, default=False
            Whether to standardize the data.

        Returns:
        --------
        float
            Deviance value,that is twice the negative pseudo-likelihood, divided by the number of extremes.

        """
        if self.subfaces is None:
            raise RuntimeError(
                "The model has not been fitted yet. Call 'fit' "
                "with appropriate arguments before using this "
                "method."
            )

        Xt = rank_transform(Xtest) if standardize else Xtest

        # Update instance's attributes if passed as arguments
        if thresh_test is not None:
            self.thresh_test = thresh_test
        if self.thresh_test is None:
            self.thresh_test = self.thresh_train

        Subfaces = self.subfaces
        Masses = self.masses

        negative_pseudo_lkl = ut.total_deviance(
            Subfaces, Masses, Xt, self.thresh_test, False, None,
            rate=self.rate
        )

        return 2 * negative_pseudo_lkl

    def get_AIC(self, Xtrain, standardize=True):
        """Calculate the Akaike Information Criterion (AIC) for the
        model incorporating the normalized deviance from
        :func:`self.deviance` and the (normalized again) number of
        components in the model, that is the number of subfaces with
        positive extreme mass. More precisely the function returns

        self.deviance  + 2 * number_of_subfaces/number_of_extremes

        where whe recall that self.deviance is:

        "twice the negative pseudo log-likelihood in a dispersion model"

        

        Parameters:
        -----------
        Xtrain : np.ndarray
            Training data.
        standardize : bool, default=True
            Whether to standardize the data.

        Returns:
        --------
        float
            AIC value.

        """
        if self.masses is None:
            raise RuntimeError("Fit the model before computing the AIC")

        Masses = self.masses
        intern_deviance = self.deviance(Xtrain, self.thresh_train, standardize)
        return intern_deviance + 2 * len(Masses) / self.number_extremes

    def select_kappa_min_AIC(
        self,
        grid,
        X,
        standardize=True,
        unstable_kappam_max=0.05,
        plot=False,
        update_kappa_min=False,
    ):
        """
        Select the optimal kappa_min value based on AIC.

        Parameters:
        -----------
        grid : list
            Grid of kappa_min values to test.
        X : np.ndarray
            Input data.
        standardize : bool, default=True
            Whether to standardize the data.
        unstable_kappam_max : float, default=0.05
            Maximum kappa_min value for unstable solutions. The selected value returned by the function will be greater than this parameter. 
        plot : bool, default=False
            Whether to plot the AIC values.
        update_kappa_min : bool, default=False
            Whether to update the model's kappa_min value.

        Returns:
        --------
        kappam_select_aic : float
            Selected kappa_min value.
        aic_opt : float
            Optimal AIC value.
        vect_aic : np.ndarray
            Vector of AIC values.
        """
        old_kappa_min = deepcopy(self.kappa_min)
        Xt = rank_transform(X) if standardize else X
        ntests = len(grid)
        vect_aic = np.zeros(ntests)

        for counter, kapp in enumerate(grid):
            _, _ = self.fit(Xt, kapp, standardize=False)
            vect_aic[counter] = self.get_AIC(Xt, standardize=False)

        i_maxerr = np.argmax(vect_aic[grid < unstable_kappam_max])
        kapp_maxerr = grid[i_maxerr]
        i_mask = grid <= kapp_maxerr
        i_minAIC = np.argmin(vect_aic + (1e23) * i_mask)
        kapp_select_aic = grid[i_minAIC]
        aic_opt = vect_aic[i_minAIC]

        if plot:
            print(
                f"CLEF: selected kappa_min with AIC criterion: "
                f"{round_signif(kapp_select_aic, 2)}"
            )
            plt.figure(figsize=(10, 5))
            plt.xlabel("kappa_min")
            plt.ylabel("AIC")
            plt.title("CLEF: AIC versus kappa_min")
            plt.scatter(grid, vect_aic, c="gray", label="AIC")
            plt.plot([kapp_select_aic, kapp_select_aic], [0, aic_opt], c="red")
            plt.grid(True)
            plt.show()

        # Update instance's kappa_min value with initial or selected value
        if update_kappa_min:
            self.kappa_min = kapp_select_aic
        else:
            self.kappa_min = old_kappa_min

        # Re-fit the model with final kappa_min
        _, _ = self.fit(Xt, self.kappa_min, False)

        return kapp_select_aic, aic_opt, vect_aic

    def deviance_CV(
        self,
        X,
        standardize=True,
        kappa_min=None,
        thresh_test=None,
        cv=5,
        random_state=None,
    ):
        """
        Estimate the expected deviance using cross-validation. See :func:`deviance`

        Parameters:
        -----------
        X : np.ndarray
            Input data.
        standardize : bool, default=True
            Whether to standardize the data.
        kappa_min : float, optional
            Tolerance level for clustering. If None, the instance's attribute will be used.
        thresh_test : float, optional
            Radial threshold for test sets. If None, the instance's attribute will be used.
        cv : int, default=5
            Number of cross-validation folds.
        random_state : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        np.ndarray
            Cross-validated deviance scores.
        """
        Xt = rank_transform(X) if standardize else X

        # Update instance's attributes if passed as arguments
        if kappa_min is not None:
            self.kappa_min = kappa_min
        if thresh_test is not None:
            self.thresh_test = thresh_test
        if self.thresh_test is None:
            self.thresh_test = self.thresh_train

        cv_neglkl_scores = ut.ftclust_cross_validate(
            Xt,
            standardize=False,
            algo="clef",
            tolerance=self.kappa_min,
            min_counts=None,
            use_max_subfaces=None,
            thresh_train=self.thresh_train,
            thresh_test=self.thresh_test,
            include_singletons_train=False,
            include_singletons_test=False,
            rate=self.rate,
            cv=cv,
            random_state=random_state,
        )

        return 2 * cv_neglkl_scores

    def select_kappa_min_CV(
        self,
        grid,
        X,
        standardize=True,
        update_kappa_min=False,
        unstable_tol_max=0.05,
        thresh_test=None,
        cv=5,
        random_state=None,
        plot=False,
    ):
        """
        Select the optimal kappa_min value based on cross-validation.

        Parameters:
        -----------
        grid : list
            Grid of kappa_min values to test.
        X : np.ndarray
            Input data.
        standardize : bool, default=True
            Whether to standardize the data.
        update_kappa_min : bool, default=False
            Whether to update the model's kappa_min value.
        unstable_tol_max : float, default=0.05
            Maximum tolerance value for unstable solutions.
        cv : int, default=5
            Number of cross-validation folds.
        random_state : int, optional
            Random seed for reproducibility.
        plot : bool, default=False
            Whether to plot the CV deviance values.

        Returns:
        --------
        tol_cv : float
            Selected kappa_min value.
        deviance_tol_cv : float
            Deviance value for the selected kappa_min.
        cv_deviance_vect : np.ndarray
            Vector of CV deviance values.
        """
        Xt = rank_transform(X) if standardize else X
        kappa_min_old = deepcopy(self.kappa_min)
        ntol = len(grid)
        cv_deviance_vect = np.zeros(ntol)

        for i, kapp in enumerate(grid):
            cv_scores = self.deviance_CV(Xt, False, kapp, thresh_test, cv, random_state)
            cv_deviance_vect[i] = np.mean(cv_scores)

        i_maxerr = np.argmax(cv_deviance_vect[grid < unstable_tol_max])
        tol_maxerr = grid[i_maxerr]
        i_mask = grid <= tol_maxerr
        i_cv = np.argmin(cv_deviance_vect + (1e23) * i_mask)
        tol_cv = grid[i_cv]
        deviance_tol_cv = cv_deviance_vect[i_cv]

        if plot:
            print(
                f"CLEF: selected kappa_min with CV estimate of deviance: "
                f"{round_signif(tol_cv, 2)}"
            )
            plt.scatter(grid, cv_deviance_vect, c="gray", label="CV deviance")
            plt.plot(
                [tol_cv, tol_cv], [0, deviance_tol_cv], c="red",
                label="selected value"
            )
            plt.grid()
            plt.legend()
            plt.title("CLEF: expected deviance estimated with K-fold CV")
            plt.show()

        if update_kappa_min:
            self.kappa_min = tol_cv
        else:
            self.kappa_min = kappa_min_old

        # Re-fit the model with final kappa_min
        _, _ = self.fit(Xt, self.kappa_min, False)

        return tol_cv, deviance_tol_cv, cv_deviance_vect

    def deviance_to_true(self, subfaces_true, weights_true):
        """Calculate the deviance between the estimated subfaces and
        the true subfaces.

        The function  returns a tuple (est_to_truth, truth_to_est) respectively
        defined as follows:

        .. math::

            \\text{est-to-truth} = -2\
        \\sum_{j}^{\\text{N-estimated}} \
        w^{estim}_j \\log \\Big( \
        \\sum_{i=1}^{\\text{N-true}}\
       w^{true}_i e^{-\\text{rate}*d(\\text{estimated-subface}_j, \
        \\text{true-subface}_i)} \
        \\Big)
        
            \\text{tuth-to-est} = -2\
        \\sum_{i=1}^{\\text{N-true}}\
         w^{true}_i \\log \\Big( \
        \\sum_{j}^{\\text{N-estimated}} \
        w^{estim}_j 
         e^{-\\text{rate}*d(\\text{estimated-subface}_j, \
        \\text{true-subface}_i)} \
        \\Big)
        
        where :math:`w^{estim}_j =` self.mass[j-1] / sum(self.masses) and
        :math:`w^{truth}_j` is the mass of the considered true subface divided
        by the total mass of extremes 
        
        Parameters:
        -----------
        subfaces_true : list
            True subfaces.
        weights_true : list
            Weights of the true subfaces.

        Returns:
        --------
        est_to_truth : float
            Deviance from estimated to true subfaces.
        truth_to_est : float
            Deviance from true to estimated subfaces.

        """
        if self.subfaces is None:
            raise RuntimeError("CLEF has not been fitted yet")

        Subfaces_matrix = ut.subfaces_list_to_matrix(self.subfaces, self.dimension)
        Masses = self.masses
        if isinstance(self.masses, list):
            Masses = np.array(Masses)

        Subfaces_true_matrix = ut.subfaces_list_to_matrix(subfaces_true, self.dimension)
        if isinstance(weights_true, list):
            weights_true = np.array(weights_true)

        # if not self.include_singletons:
        id_keep_estim = np.where(np.sum(Subfaces_matrix, axis=1) >= 2)[0]
        id_keep_true = np.where(np.sum(Subfaces_true_matrix, axis=1) >= 2)[0]
        Subfaces_matrix = Subfaces_matrix[id_keep_estim]
        Masses = Masses[id_keep_estim]
        Masses = Masses / np.sum(Masses) if np.sum(Masses) > 0 else Masses
        Subfaces_true_matrix = Subfaces_true_matrix[id_keep_true]
        weights_true = weights_true[id_keep_true]
        weights_true = (
            weights_true / np.sum(weights_true)
            if np.sum(weights_true) > 0
            else weights_true
        )

        est_to_truth = 2 * ut.total_deviance_binary_matrices(
            Subfaces_matrix, Masses, Subfaces_true_matrix, weights_true, self.rate
        )
        truth_to_est = 2 * ut.total_deviance_binary_matrices(
            Subfaces_true_matrix, weights_true, Subfaces_matrix, Masses, self.rate
        )

        return est_to_truth, truth_to_est


# class clef:
#     def __init__(self, kappa_min=0.1, thresh_train=None,
#                  thresh_test=None,
#                  rate=10):
#         """Initialize the DAMEX model with specified parameters.

#         Parameters:
#         -----------

#         - kappa_min (float): Tolerance level for clustering.

#         - min_counts (int): Minimum number of points required to
#             form a cluster.

#         - thresh_train (float): Threshold for training data.

#         - thresh_test (float): Threshold for test data.

#          """
#         self.kappa_min = kappa_min
#         self.thresh_train = thresh_train
#         self.thresh_test = thresh_test
#         self.rate = rate
#         self.subfaces = None  # Identified subfaces
#         self.masses = None  # Masses associated with subfaces
#         self.total_mass = None  # Total mass of extremes
#         self.number_extremes = None  # Number of extreme points
#         self.dimension = None  # Dimensionality of the data

#     def fit(self, X,  kappa_min=None, standardize=True):
#         """Fit the CLEF model to the data.

#         Parameters:
#         -----------

#         - X (np.ndarray): Input data.

#         - kappa_min
#         (float): Tolerance level for clustering. If None, the
#         instance's attribute will be used.

#         - standardize (bool):
#         Whether to standardize the data.


#         Returns:
#         ---------

#         - Subfaces (list): Identified subfaces.

#         - Masses (np.ndarray): Masses associated with subfaces.

#         """
#         # Update instance attributes with optional arguments passed
#         if kappa_min is not None:
#             self.kappa_min = kappa_min

#         # Record dimension
#         self.dimension = np.shape(X)[1]

#         # Standardize X if needed
#         Xt = rank_transform(X) if standardize else X
#         norm_Xt = np.max(Xt, axis=1)

#         # Set the training threshold if not provided
#         if self.thresh_train is None:
#             self.thresh_train = np.quantile(
#                 norm_Xt, 1 - 1 / np.sqrt(len(norm_Xt)))

#         # Update total mass accordingly
#         self.number_extremes = np.sum(norm_Xt > self.thresh_train)
#         self.total_mass = self.thresh_train * \
#             self.number_extremes / len(norm_Xt)

#         # Fit the model
#         Subfaces = ut.clef_fit(
#             Xt, self.thresh_train, self.kappa_min,
#             standardize=False, include_singletons=False)
#         Masses = ut.estim_subfaces_mass(
#             Subfaces, Xt, self.thresh_train, epsilon=None,
#             standardize=False)
#         # Update instance's attributes
#         self.subfaces = Subfaces
#         self.masses = Masses

#         return Subfaces, Masses

#     def deviance(self, Xtest, thresh_test=None,  standardize=False):
#         """
#         Calculate the deviance of the model on test data.

#         Parameters:
#         -----------

#         - Xtest (np.ndarray): Test data.

#         - thresh_test
#         (float): Threshold for identifying extremes in the test set.
#         If None, the instance's attribute will be used instead. If the
#         lmatter is also None, thresh_test will be set to the same
#         value as thresh_train.

#         - standardize (bool): Whether to standardize the data.

#         Returns:
#         ---------

#         - float: Deviance value.
#         """
#         if self.subfaces is None:
#             raise RuntimeError("The model has not been fitted yet. Call 'fit' "
#                                "with appropriate arguments before using this "
#                                "method.")

#         Xt = rank_transform(Xtest) if standardize else Xtest

#         # Update instance's attributes if passed as arguments
#         if thresh_test is not None:
#             self.thresh_test = thresh_test
#         if self.thresh_test is None:
#             self.thresh_test = self.thresh_train

#         Subfaces = self.subfaces
#         Masses = self.masses

#         negative_pseudo_lkl = ut.total_deviance(
#             Subfaces, Masses, Xt,  self.thresh_test,
#             False, None, rate=self.rate)

#         return 2 * negative_pseudo_lkl

#     def get_AIC(self, Xtrain, standardize=True):
#         """
#         Calculate the Akaike Information Criterion (AIC) for the model.

#         Parameters:

#         - Xtrain (np.ndarray): Training data.

#         - standardize (bool): Whether to standardize the data.

#         Returns:

#         - float: AIC value.
#         """
#         if self.masses is None:
#             raise RuntimeError("Fit the model before computing the AIC")

#         Masses = self.masses
#         intern_deviance = self.deviance(Xtrain, self.thresh_train, standardize)
#         return intern_deviance + 2 * len(Masses) / self.number_extremes

#     def select_kappa_min_AIC(self, grid, X, standardize=True,
#                              unstable_kappam_max=0.05,
#                              plot=False,
#                              update_kappa_min=False):
#         """
#         Select the optimal kappa_min value based on AIC.

#         Parameters:
#         ----------

#         - grid (list): Grid of kappa_min values to test.

#         - X (np.ndarray): Input data.

#         - standardize (bool): Whether to standardize the data.

#         - unstable_kappam_max (float): Maximum kappa_min value for unstable
#           solutions.

#         - plot (bool): Whether to plot the AIC values.

#         - update_kappa_min (bool): Whether to update the model's
#             kappa_min value.

#         Returns:
#         --------

#         - kappam_select_aic (float): Selected kappa_min value.

#         - aic_opt (float): Optimal AIC value.

#         - vect_aic (np.ndarray): Vector of AIC values.
#         """
#         old_kappa_min = deepcopy(self.kappa_min)
#         Xt = rank_transform(X) if standardize else X
#         ntests = len(grid)
#         vect_aic = np.zeros(ntests)

#         for counter, kapp in enumerate(grid):
#             _, _ = self.fit(Xt, kapp, standardize=False)
#             vect_aic[counter] = self.get_AIC(Xt, standardize=False)

#         i_maxerr = np.argmax(vect_aic[grid < unstable_kappam_max])
#         kapp_maxerr = grid[i_maxerr]
#         i_mask = grid <= kapp_maxerr
#         i_minAIC = np.argmin(vect_aic + (1e+23) * i_mask)
#         kapp_select_aic = grid[i_minAIC]
#         aic_opt = vect_aic[i_minAIC]

#         if plot:
#             print(f'CLEF: selected kappa_min with AIC criterion: '
#                   f'{round_signif(kapp_select_aic, 2)}')
#             plt.figure(figsize=(10, 5))
#             plt.xlabel('kappa_min')
#             plt.ylabel('AIC')
#             plt.title('CLEF: AIC versus kappa_min')
#             plt.scatter(grid, vect_aic, c='gray', label='AIC')
#             plt.plot([kapp_select_aic, kapp_select_aic], [0, aic_opt],
#                      c='red')
#             plt.grid(True)
#             plt.show()

#         # Update instance's kappa_min value with initial or selected value
#         if update_kappa_min:
#             self.kappa_min = kapp_select_aic
#         else:
#             self.kappa_min = old_kappa_min

#         # Re-fit the model with final kappa_min
#         _, _ = self.fit(Xt,  self.kappa_min, False)

#         return kapp_select_aic, aic_opt, vect_aic

#     def deviance_CV(self, X, standardize=True, kappa_min=None,
#                     thresh_test=None, cv=5, random_state=None):
#         """
#         Estimate the expected  deviance using cross-validation.

#         Parameters:
#         -----------

#         - X (np.ndarray): Input data.

#         - standardize (bool): Whether to standardize the data. Default to True

#         - kappa_min (float): Tolerance level for clustering. Default to
#           None. If None, the instance's attribute will be used.

#         - thresh_test (float, optional): radial threshold for test
#           sets. If None, the instance's attribute will be used. If the
#           latter is also None, the training threshold is used instead.

#         - cv (int): Number of cross-validation folds. Default to 5

#         - random_state (int): Random seed for reproducibility. Default to None

#         Returns:
#         - np.ndarray: Cross-validated deviance scores.
#         """
#         Xt = rank_transform(X) if standardize else X

#         # Update instance's attributes if passed as arguments
#         if kappa_min is not None:
#             self.kappa_min = kappa_min
#         if thresh_test is not None:
#             self.thresh_test = thresh_test
#         if self.thresh_test is None:
#             self.thresh_test = self.thresh_train

#         cv_neglkl_scores = ut.ftclust_cross_validate(
#             Xt, standardize=False, algo='clef', tolerance=self.kappa_min,
#             min_counts=None, use_max_subfaces=None,
#             thresh_train=self.thresh_train, thresh_test=self.thresh_test,
#             include_singletons_train=False,
#             include_singletons_test=False,
#             rate=self.rate, cv=cv, random_state=random_state)

#         return 2 * cv_neglkl_scores

#     def select_kappa_min_CV(self, grid, X, standardize=True,
#                             update_kappa_min=False,
#                             unstable_tol_max=0.05, thresh_test=None,  cv=5,
#                             random_state=None, plot=False):
#         """
#         Select the optimal kappa_min value based on cross-validation.

#         Parameters:
#         ------------

#         - grid (list): Grid of kappa_min values to test.

#         - X (np.ndarray): Input data.

#         - standardize (bool): Whether to standardize the data.

#         - update_kappa_min (bool): Whether to update the model's
#             kappa_min value.

#         - unstable_tol_max (float): Maximum tolerance value for unstable
#           solutions.

#         - cv (int): Number of cross-validation folds.

#         - random_state (int): Random seed for reproducibility.

#         - plot (bool): Whether to plot the CV deviance values.


#         Returns:

#         - tol_cv (float): Selected kappa_min value.

#         - deviance_tol_cv (float): Deviance value for the selected kappa_min.

#         - cv_deviance_vect (np.ndarray): Vector of CV deviance values.
#         """
#         Xt = rank_transform(X) if standardize else X
#         kappa_min_old = deepcopy(self.kappa_min)
#         ntol = len(grid)
#         cv_deviance_vect = np.zeros(ntol)

#         for i, kapp in enumerate(grid):
#             cv_scores = self.deviance_CV(Xt, False, kapp, thresh_test,
#                                          cv, random_state)
#             cv_deviance_vect[i] = np.mean(cv_scores)

#         i_maxerr = np.argmax(cv_deviance_vect[grid < unstable_tol_max])
#         tol_maxerr = grid[i_maxerr]
#         i_mask = grid <= tol_maxerr
#         i_cv = np.argmin(cv_deviance_vect + (1e+23) * i_mask)
#         tol_cv = grid[i_cv]
#         deviance_tol_cv = cv_deviance_vect[i_cv]

#         if plot:
#             print(f'CLEF: selected kappa_min with CV estimate of deviance: '
#                   f'{round_signif(tol_cv, 2)}')
#             plt.scatter(grid, cv_deviance_vect, c='gray', label='CV deviance')
#             plt.plot([tol_cv, tol_cv], [0, deviance_tol_cv], c='red',
#                      label='selected value')
#             plt.grid()
#             plt.legend()
#             plt.title("CLEF: expected  deviance estimated with  K-fold CV")
#             plt.show()

#         if update_kappa_min:
#             self.kappa_min = tol_cv
#         else:
#             self.kappa_min = kappa_min_old

#         # Re-fit the model with final kappa_min
#         _, _ = self.fit(Xt, self.kappa_min, False)

#         return tol_cv, deviance_tol_cv, cv_deviance_vect

#     def deviance_to_true(self,  subfaces_true, weights_true):
#         """
#         Calculate the deviance between the estimated subfaces and the true
#         subfaces.

#         Parameters:
#         -----------

#         - subfaces_true (list): True subfaces.

#         - weights_true (list): Weights of the true subfaces.

#         - use_max_subfaces (bool): Whether to use maximal subfaces.

#         - rate (float, >0): Rate parameter for deviance calculation.

#         Returns:
#         --------

#         - est_to_truth (float): Deviance from estimated to true subfaces.

#         - truth_to_est (float): Deviance from true to estimated subfaces.
#         """
#         if self.subfaces is None:
#             raise RuntimeError("CLEF has not been fitted yet")

#         Subfaces_matrix = ut.subfaces_list_to_matrix(
#             self.subfaces, self.dimension)
#         Masses = self.masses
#         if isinstance(self.masses, list):
#             Masses = np.array(Masses)

#         Subfaces_true_matrix = ut.subfaces_list_to_matrix(
#             subfaces_true, self.dimension)
#         if isinstance(weights_true, list):
#             weights_true = np.array(weights_true)

#         #if not self.include_singletons:
#         id_keep_estim = np.where(np.sum(Subfaces_matrix, axis=1) >= 2)[0]
#         id_keep_true = np.where(np.sum(Subfaces_true_matrix,
#                                        axis=1) >= 2)[0]
#         Subfaces_matrix = Subfaces_matrix[id_keep_estim]
#         Masses = Masses[id_keep_estim]
#         Masses = Masses / np.sum(Masses) if np.sum(Masses) > 0 else Masses
#         Subfaces_true_matrix = Subfaces_true_matrix[id_keep_true]
#         weights_true = weights_true[id_keep_true]
#         weights_true = weights_true / np.sum(weights_true) \
#             if np.sum(weights_true) > 0 else weights_true

#         est_to_truth = 2 * ut.total_deviance_binary_matrices(
#             Subfaces_matrix, Masses, Subfaces_true_matrix, weights_true,
#             self.rate)
#         truth_to_est = 2 * ut.total_deviance_binary_matrices(
#             Subfaces_true_matrix, weights_true, Subfaces_matrix,
#             Masses,  self.rate)

#         return est_to_truth, truth_to_est
