"""
File: mvsets.py
Author: Anne Sabourin
Date: 2025


"""
import numpy as np
from math import isclose
import matplotlib.pyplot as plt
from copy import deepcopy
from MLExtreme.utils.EVT_basics import rank_transform, round_signif



class Angular_bin:
    """Building blocks of angular minimum-volume sets (mvsets)
    constructed from a regular paving of the unit sphere for the
    infinite norm. Geometrically, each grid is a rectangle on the
    positive orthant of the unit sphere defined from the infinite
    norm, of the kind:

    prod_{i in {1, ..., d}, i != i_0} [j_i/J, (j_i+1)/J) x {e_{i_0}}

    where j_i in {0, ..., J-1}, J is the chosen number of basic
    subdivisions of the interval [0, 1) and i_0 is the index of the
    canonical basis vector e_{i_0} normal to the face where the bin is
    located.

    Internally a bin is encoded as a np.array of integers in 0, ..., J-1
    of dimension d, containing the lower end points of the intervals in
    the product, except for the i_0^{th} entry which is equal to J. A hash
    function is provided to enable quick responses to queries about the
    existence of keys in a dictionary of `angular_bin` class instances.
    """

    def __init__(self, coords: np.array):
        ##self.subface = np.argmax(coords)
        discrete_coord = np.floor(coords).astype(int)
        J = np.max(discrete_coord)
        id_max = np.where(discrete_coord == J)[0]
        if len(id_max) > 1:
            raise ValueError("A bin cannot be on an edge.")
        self.coords = discrete_coord

    def __hash__(self):
        coords = np.array(self.coords)
        J = np.max(coords)
        dim = len(coords)
        exponents = (J + 1) ** np.arange(dim)
        return int(np.sum(exponents * coords))

    def __eq__(self, other):
        if isinstance(other, Angular_bin):
            return np.array_equal(self.coords, other.coords)
        return False


def get_angular_bin(point: np.array, J: int) -> Angular_bin:
    """Get the angular bin for a given point.

    Parameters:
    -----------
    point : np.array
        The point for which to find the angular bin.
    J : int
        The number of subdivisions of the interval [0, 1).

    Returns:
    --------
    Angular_bin
        The angular bin corresponding to the given point.
    """
    angle = point / np.max(point)
    edge_indices = np.where(np.isclose(angle, 1))[0]
    # move the point away from the edge by decreasing all but the
    # one random coordinate by a small enough value so that the shifted
    # point belongs to the next adjacent bin.
    if len(edge_indices) > 1:
        j = np.random.randint(0, len(edge_indices))
        other_indices = np.delete(edge_indices, j)
        for i in other_indices:
            angle[i] = angle[i] - 1 / (2 * J)

    return Angular_bin(J * angle)


def sort_bins_by_mass(X: np.array, J: int) -> list[Angular_bin]:
    """Sort angular bins by their mass.

    Parameters:
    -----------
    X : np.array
        The input data.
    J : int
        The number of subdivisions of the interval [0, 1).

    Returns:
    --------
    list[Angular_bin]
        A list of angular bins sorted by mass in descending order.
    """
    bin_dict = {}
    n = np.shape(X)[0]
    # pdb.set_trace()
    for i in range(n):
        x = X[i,]
        ang_bin = get_angular_bin(x, J)
        if ang_bin not in bin_dict:
            bin_dict[ang_bin] = 1
        else:
            bin_dict[ang_bin] += 1

    sorted_list = sorted(bin_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_list


class Xmvset:
    """
    Minimum-volume set estimation and anomaly scoring.

    Implements the methods developed in [1].

    Compared with the implementation proposed in [1] the only  difference resides on the data structure manipulation performs to affect points to the building blocks of the mass-volume sets. This doess not affect the output.

    References
    _________
    [1] Thomas, A., Clémençon, S., Gramfort, A., & Sabourin, A. (2017, April). Anomaly Detection in Extreme Regions via Empirical MV-sets on the Sphere. In AISTATS (Vol. 54).
    """

    def __init__(self, thresh_train=None, thresh_predict=None, J=None):
        """Initialize the Xmvset model.

        Parameters:
        -----------
        thresh_train : float, optional
            Threshold for identifying extreme points during training.
        thresh_predict : float, optional
            Threshold for identifying extreme points during prediction.
        J : int, optional
            Number of subdivisions of the interval [0, 1).
        """
        self.thresh_train = thresh_train
        self.thresh_predict = thresh_predict
        self.J = J
        self.masses = None
        self.coords = None
        self.Xtrain = None

    def set_default_J(self, X):
        """Set the default value for J based on the input data.

        Parameters:
        -----------
        X : np.array
            The input data.
        """
        n = len(X)
        d = np.shape(X)[1]
        default_J = np.ceil((n / d**2) ** (1 / (2 * (d - 1))))
        self.J = default_J

    def fit(self, X):
        """Fit the model to the input data.

        Parameters:
        -----------
        X : np.array
            The input data.
        """
        # # Standardize X if needed
        # if standardize:
        #     Xt = rank_transform(X)
        # else:
        #     Xt = X
        Xt = X
        norm_Xt = np.max(Xt, axis=1)

        # Set the training threshold if not provided
        if self.thresh_train is None:
            self.thresh_train = np.quantile(norm_Xt, 1 - 1 / np.sqrt(len(norm_Xt)))

        # define extreme samples
        X_ext = Xt[norm_Xt > self.thresh_train]
        # Update total mass accordingly
        self.number_extremes = len(X_ext)
        # self.total_mass = self.thresh_train * self.number_extremes / len(norm_Xt)

        # set J if not provided
        if self.J is None:
            self.set_default_J(X_ext)
        angular_bins = sort_bins_by_mass(X_ext, self.J)

        masses = np.array([item[1] for item in angular_bins])
        coords = np.array([item[0].coords for item in angular_bins])
        # pdb.set_trace()
        self.masses = masses
        self.coords = coords
        # self.list_angular_bins = angular_bins
        # self.Xtrain = X

    def predict(self, X_test, alpha):  # , standardize=True):
        """Predicts whether extreme points belong to an extreme angular minimum-volume
        set at level alpha.

        Parameters:
        -----------
        X_test : np.array
            Array of test data.
        alpha : float
            Proportion of extreme points contained in the angular minimum-volume set.
        standardize : bool, optional
            If True, the test points are rank-transformed based on the training data.
            Default is True.

        Returns:
        --------
        np.array
            An array of boolean values of the same length as X_test, indicating
            whether each point belongs to the extreme angular minimum-volume set.
        """
        if self.masses is None:
            raise ValueError("the model has not been trained")
        if self.thresh_predict is None:
            self.thresh_predict = self.thresh_train
        # cumulative masses of bins preliminarily sorted in decreasing order
        cum_rel_masses = np.cumsum(self.masses) / self.number_extremes
        # indices of bins where cumulative mass exceeds alpha
        id_sets_enough_mass = np.where(cum_rel_masses >= alpha)[0]
        # first index such that the cumulative mass exceeds alpha
        id_mvset = id_sets_enough_mass[0]
        # create a set of bins that form the mvset at level alpha.
        # These are the bins with index less than id_mvset in the array of bins
        # represented by self.coords
        set_bins_mvset = {Angular_bin(self.coords[i]) for i in range(id_mvset + 1)}
        # pdb.set_trace()
        # select extreme samples
        Norm_Xtest = np.max(X_test, axis=1)
        mask_test = Norm_Xtest >= self.thresh_predict
        X_test_extreme = X_test[mask_test]
        n_extremes_test = len(X_test_extreme)
        # among extreme samples, test which ones belong to the mvset :
        # a sample belongs to a mvset iff the angular component of the sample
        # belongs to a bin  which is part of  the mvset.
        is_not_member = np.zeros(n_extremes_test)
        for i in range(n_extremes_test):
            is_not_member[i] = (
                get_angular_bin(X_test_extreme[i], self.J) not in set_bins_mvset
            )
        return is_not_member, X_test_extreme, mask_test

    def score(self, X_test, angular_only=False):
        """Return the angular score of extreme points defined simply as an
        un-normalized histogram-based estimated angular density.
        The lower, the more abnormal.

        Parameters:
        -----------
        X_test : np.array
            Array of test data.
        angular_only : bool, optional
            If True, only the angular score is returned. Default is False.

        Returns:
        --------
        np.array
            An array of scores of the same length as X_test.
        """
        if self.masses is None:
            raise ValueError("the model has not been trained")
        if self.thresh_predict is None:
            self.thresh_predict = self.thresh_train

        # create a dictionary (angular_bins, num_points)
        bin_dict = {
            Angular_bin(self.coords[i]): self.masses[i] for i in range(len(self.masses))
        }

        # select extreme samples
        Norm_Xtest = np.max(X_test, axis=1)
        mask_test = Norm_Xtest >= self.thresh_predict
        X_test_extreme = X_test[mask_test]
        n_extremes_test = len(X_test_extreme)

        # compute the angular score
        score = np.zeros(n_extremes_test)
        for i in range(n_extremes_test):
            bin_key = get_angular_bin(X_test_extreme[i], self.J)
            if bin_key in bin_dict:
                ang_score = bin_dict[bin_key]  # number of extreme angles in that bin
                if angular_only:
                    score[i] = ang_score
                else:
                    score[i] = ang_score / np.max(X_test_extreme[i]) ** 2
        return score, X_test_extreme, mask_test


# workflow:

# input: data
# - select extremes and make angles/radii

# - make sorted angular histogram
# - make angular mvsets with of defined (relative) mass level
#          --> OUTPUT: ordered list of cases +
#           ---> indicator function(level, point)
#           --> angular extremal scoring function
#           ---> extremal scoring


# """
# File: mvsets.py
# Author: Anne Sabourin
# Date: 2025-05-02
# """

# import numpy as np
# from math import isclose
# import matplotlib.pyplot as plt
# from copy import deepcopy
# from ...utils.EVT_basics import rank_transform, round_signif


# class Angular_bin:
#     """Building blocks of angular minimum-volume sets (mvsets)
#     constructed from a regular paving of the unit sphere for the
#     infinite norm. Geometrically, each grid is a rectangle on the
#     positive orthant of the unit sphere defined from the infinite
#     norm, of the kind:

#     prod_{i in {1, ..., d}, i != i_0} [j_i/J, (j_i+1)/J) x {e_{i_0}}

#     where j_i in {0, ..., J-1}, J is the chosen number of basic
#     subdivisions of the interval [0, 1) and i_0 is the index of the
#     canonical basis vector e_{i_0} normal to the face where the bin is
#     located.

#     Internally a bin is encoded as a np.array of integers in 0, ..., J-1
#     of dimension d, containing the lower end points of the intervals in
#     the product, except for the i_0^{th} entry which is equal to J. A hash
#     function is provided to enable quick responses to queries about the
#         existence of keys in a dictionary of `angular_bin` class instances.

#     """

#     def __init__(self, coords: np.array):
#         ##self.subface = np.argmax(coords)
#         discrete_coord = np.floor(coords).astype(int)
#         J = np.max(discrete_coord)
#         id_max = np.where(discrete_coord == J)[0]
#         if len(id_max) > 1:
#             raise ValueError("A bin cannot be on an edge.")
#         self.coords = discrete_coord

#     def __hash__(self):
#         coords = np.array(self.coords)
#         J = np.max(coords)
#         dim = len(coords)
#         exponents = (J + 1) ** np.arange(dim)
#         return int(np.sum(exponents * coords))

#     def __eq__(self, other):
#         if isinstance(other, Angular_bin):
#             return np.array_equal(self.coords, other.coords)
#         return False


# def get_angular_bin(point: np.array, J: int) -> Angular_bin:
#     angle = point / np.max(point)
#     edge_indices = np.where(np.isclose(angle, 1))[0]
#     # move the point away from the edge by decreasing all but the
#     # one random coordinate by a small enough value so that the shifted
#     # point belongs to the next adjacent bin.
#     if len(edge_indices) > 1:
#         j = np.random.randint(0, len(edge_indices))
#         other_indices = np.delete(edge_indices, j)
#         for i in other_indices:
#             angle[i] = angle[i] - 1 / (2 * J)

#     return Angular_bin(J * angle)


# def sort_bins_by_mass(X: np.array, J: int) -> list[Angular_bin]:
#     bin_dict = {}
#     n = np.shape(X)[0]
#     # pdb.set_trace()
#     for i in range(n):
#         x = X[i,]
#         ang_bin = get_angular_bin(x, J)
#         if ang_bin not in bin_dict:
#             bin_dict[ang_bin] = 1
#         else:
#             bin_dict[ang_bin] += 1

#     sorted_list = sorted(bin_dict.items(), key=lambda x: x[1], reverse=True)
#     return sorted_list


# class Xmvset:
#     """
#     minimum-volume set estimation and anomaly scoring
#     """

#     def __init__(self, thresh_train=None, thresh_predict=None, J=None):
#         self.thresh_train = thresh_train
#         self.thresh_predict = thresh_predict
#         self.J = J
#         self.masses = None
#         self.coords = None
#         self.Xtrain = None

#     def set_default_J(self, X):
#         n = len(X)
#         d = np.shape(X)[1]
#         default_J = np.ceil((n / d**2) ** (1 / (2 * (d - 1))))
#         self.J = default_J

#     def fit(self, X):
#         # # Standardize X if needed
#         # if standardize:
#         #     Xt = rank_transform(X)
#         # else:
#         #     Xt = X
#         Xt = X
#         norm_Xt = np.max(Xt, axis=1)

#         # Set the training threshold if not provided
#         if self.thresh_train is None:
#             self.thresh_train = np.quantile(norm_Xt, 1 - 1 / np.sqrt(len(norm_Xt)))

#         # define extreme samples
#         X_ext = Xt[norm_Xt > self.thresh_train]
#         # Update total mass accordingly
#         self.number_extremes = len(X_ext)
#         # self.total_mass = self.thresh_train * self.number_extremes / len(norm_Xt)

#         # set J if not provided
#         if self.J is None:
#             self.set_default_J(X_ext)
#         angular_bins = sort_bins_by_mass(X_ext, self.J)

#         masses = np.array([item[1] for item in angular_bins])
#         coords = np.array([item[0].coords for item in angular_bins])
#         # pdb.set_trace()
#         self.masses = masses
#         self.coords = coords
#         # self.list_angular_bins = angular_bins
#         # self.Xtrain = X

#     def predict(self, X_test, alpha):  # , standardize=True):
#         """
#         Predicts whether extreme points belong to an extreme angular minimum-volume
#         set at level alpha.

#         Parameters:
#         -----------
#         Xtest : np.array
#             Array of test data.
#         alpha : float
#             Proportion of extreme points contained in the angular minimum-volume set.
#         standardize : bool, optional
#             If True, the test points are rank-transformed based on the training data.
#             Default is True.

#         Returns: --------
#         np.array *
#             An array of boolean values of the same length as Xtest, indicating
#             whether each point belongs to the extreme angular minimum-volume set.
#         """
#         if self.masses is None:
#             raise ValueError("the model has not been trained")
#         if self.thresh_predict is None:
#             self.thresh_predict = self.thresh_train
#         # cumulative masses of bins preliminarily sorted in decreasing order
#         cum_rel_masses = np.cumsum(self.masses) / self.number_extremes
#         # indices of bins where cumulative mass exceeds alpha
#         id_sets_enough_mass = np.where(cum_rel_masses >= alpha)[0]
#         # first index such that the cumulative mass exceeds alpha
#         id_mvset = id_sets_enough_mass[0]
#         # create a set of bins that form the mvset at level alpha.
#         # These are the bins with index less than id_mvset in the array of bins
#         # represented by self.coords
#         set_bins_mvset = {Angular_bin(self.coords[i]) for i in range(id_mvset + 1)}
#         # pdb.set_trace()
#         # select extreme samples
#         Norm_Xtest = np.max(X_test, axis=1)
#         mask_test = Norm_Xtest >= self.thresh_predict
#         X_test_extreme = X_test[mask_test]
#         n_extremes_test = len(X_test_extreme)
#         # among extreme samples, test which ones belong to the mvset :
#         # a sample belongs to a mvset iff the angular component of the sample
#         # belongs to a bin  which is part of  the mvset.
#         is_not_member = np.zeros(n_extremes_test)
#         for i in range(n_extremes_test):
#             is_not_member[i] = (
#                 get_angular_bin(X_test_extreme[i], self.J) not in set_bins_mvset
#             )
#         return is_not_member, X_test_extreme, mask_test

#     def score(self, X_test, angular_only=False):
#         """return the angular score of extreme points defined simply as an
#         un-normalized histogram-based estimated angular density.
#         The lower, the more abnormal."""
#         if self.masses is None:
#             raise ValueError("the model has not been trained")
#         if self.thresh_predict is None:
#             self.thresh_predict = self.thresh_train

#         # create a dictionary (angular_bins, num_points)
#         bin_dict = {
#             Angular_bin(self.coords[i]): self.masses[i] for i in range(len(self.masses))
#         }

#         # select extreme samples
#         Norm_Xtest = np.max(X_test, axis=1)
#         mask_test = Norm_Xtest >= self.thresh_predict
#         X_test_extreme = X_test[mask_test]
#         n_extremes_test = len(X_test_extreme)

#         # compute the angular score
#         score = np.zeros(n_extremes_test)
#         for i in range(n_extremes_test):
#             bin_key = get_angular_bin(X_test_extreme[i], self.J)
#             if bin_key in bin_dict:
#                 ang_score = bin_dict[bin_key]  # number of extreme angles in that bin
#                 if angular_only:
#                     score[i] = ang_score
#                 else:
#                     score[i] = ang_score / np.max(X_test_extreme[i]) ** 2
#         return score, X_test_extreme, mask_test

