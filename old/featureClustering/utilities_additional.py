from __future__ import division
#import random as rd
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
# import networkx as nx
from . import utilities as ut
from . import clef
from . import ftclust_analysis as fca # setDistance_subface_to_matrix
from ...utils.EVT_basics import rank_transform, round_signif




# #  BELOW THIS POINT: 
# ## CONTAINS FUNCTIONS THAT ARE CURRENTLY UNUSED
# ###################################
# ### previously in utilities.py

def kth_largest_binary_matrix(rank_matrix, k):
    """
    Returns a binary matrix with the kth largest points on each column.

    Args:
    - rank_matrix (np.ndarray): Input matrix with ranked values.
    - k (int): Number of top elements to mark as 1 in each column.

    Returns:
    - np.ndarray: Binary matrix with kth largest points marked as 1.
    """
    num_samples, num_dimensions = rank_matrix.shape
    sorted_indices = np.argsort(rank_matrix, axis=0)[::-1]
    binary_matrix = np.zeros((num_samples, num_dimensions))

    for col in range(num_dimensions):
        binary_matrix[sorted_indices[:k, col], col] = 1.0

    return binary_matrix


def check_errors(true_subfaces, result_subfaces, dimension):
    """
    Compares true subfaces with result subfaces to identify recovered, missed,
    and false positives.

    Args:
    - true_subfaces (list): List of true subfaces.
    - result_subfaces (list): List of result subfaces.
    - dimension (int): Dimensionality of the subfaces.

    Returns:
    - tuple: Lists of recovered, missed, false positives, exact subsets,
             and exact supersets.
    """
    # Convert lists of subfaces into binary matrices where each row represents
    # a subface
    true_subfaces_matrix = ut.list_subfaces_to_vector(true_subfaces, dimension)
    result_subfaces_matrix = ut.list_subfaces_to_vector(result_subfaces,
                                                        dimension)

    # Calculate condition_1: Check if each result subface is a superset of any
    # true subface. This is done by checking if the dot product of result and
    # true subfaces equals the sum of true subfaces
    condition_1 = np.dot(result_subfaces_matrix, true_subfaces_matrix.T) == \
        np.sum(true_subfaces_matrix, axis=1)

    # Calculate condition_2: Check if each true subface is a subset of any
    # result subface. This is done by checking if the dot product of true and
    # result subfaces equals the sum of result subfaces
    condition_2 = np.dot(true_subfaces_matrix, result_subfaces_matrix.T) == \
        np.sum(result_subfaces_matrix, axis=1)

    # Identify exact supersets: Result subfaces that are supersets of true
    # subfaces but not exact matches
    exact_supersets = list(set(np.nonzero(np.sum(condition_1, axis=1))[0]) -
                           set(np.nonzero(np.sum(condition_1 * condition_2.T,
                                                 axis=1))[0]))

    # Identify exact subsets: Result subfaces that are subsets of true
    # subfaces but not exact matches
    exact_subsets = list(set(np.nonzero(np.sum(condition_2.T, axis=1))[0]) -
                         set(np.nonzero(np.sum(condition_1 * condition_2.T,
                                               axis=1))[0]))

    # Identify pure false positives: Result subfaces that are neither subsets
    # nor supersets of true subfaces
    pure_false_indices = list(set(range(len(result_subfaces))) -
                              (set(np.nonzero(np.sum(condition_1 * condition_2.T,
                                                     axis=1))[0]) |
                               set(exact_supersets) | set(exact_subsets)))

    # Recovered subfaces: Result subfaces that exactly match true subfaces
    recovered_subfaces = [result_subfaces[i] for i in
                          np.nonzero(np.sum(condition_1 * condition_2.T,
                                            axis=1))[0]]

    # False positives: Result subfaces that do not match any true subfaces
    false_positives = [result_subfaces[i] for i in pure_false_indices]

    # Exact subset subfaces: Result subfaces that are exact subsets of true
    # subfaces
    exact_subsets_subfaces = [result_subfaces[i] for i in exact_subsets]

    # Exact superset subfaces: Result subfaces that are exact supersets of true
    # subfaces
    exact_supersets_subfaces = [result_subfaces[i] for i in exact_supersets]

    # Missed subfaces: True subfaces that do not match any result subfaces
    missed_subfaces = [true_subfaces[i] for i in
                       np.nonzero(np.sum(condition_1 * condition_2.T, axis=0) == 0)[0]]

    # Return the categorized subfaces
    return recovered_subfaces, missed_subfaces, false_positives, \
        exact_subsets_subfaces, exact_supersets_subfaces



# #####################
# setDistance metric #
# #####################

def setDistance(subface_1, subface_2):
    """
    Calculates the pseudo-Levenshtein distance between two
    subfaces encoded with binary entries.

    Args:
    - subface_1 (np.ndarray): First subface vector.
    - subface_2 (np.ndarray): Second subface vector.

    Returns:
    - float: Pseudo-Levenshtein distance.

    """
    return np.sum(abs(subface_1 - subface_2)) / np.sum(
        subface_1 + subface_2 > 0)




def setDistance_matrix(binary_matrix):
    """
    Computes the pseudo-Levenshtein distance between each row of a
    binary matrix.

    Args:
    - binary_matrix (np.ndarray): Binary matrix.

    Returns:
    - np.ndarray: Distance matrix.
    """
    num_rows = binary_matrix.shape[0]
    distance_matrix = np.zeros((num_rows, num_rows))

    for row_index, row in enumerate(binary_matrix):
        distance_matrix[row_index] = \
            fca.setDistance_subface_to_matrix(row, binary_matrix)

    return distance_matrix

def setDistance_similarity_matrix(binary_matrix):
    """
    Computes the pseudo-Levenshtein similarity between each row of a
    binary matrix.

    Args:
    - binary_matrix (np.ndarray): Binary matrix.

    Returns:
    - np.ndarray: Similarity matrix.
    """
    return 1 - setDistance_matrix(binary_matrix)

###BUG HERE? should be 1 - ?  --> modified. previously setDistance_matrix(binary_matrix) - 1



def setDistance_subfaces_radius(subfaces, radius, rank_matrix):
    """
    Calculates the average pseudo-Levenshtein distance between a list of
    subfaces and rank matrix rows.

    Args:
    - subfaces (list): List of subfaces.
    - radius (int): Radius for binary feature extraction.
    - rank_matrix (np.ndarray): Rank matrix.

    Returns:
    - float: Average distance.
    """
    subfaces_vector = ut.list_subfaces_to_vector(subfaces,
                                                 dim=rank_matrix.shape[1])
    binary_matrix = ut.binary_large_features(rank_matrix, radius)

    return np.mean([np.min(
        fca.setDistance_subface_to_matrix(row, subfaces_vector))
                    for row in binary_matrix])


def setDistance_subfaces_radii(subfaces, radii, rank_matrix, eps=1.0):
    """
    Calculates the average pseudo-Levenshtein distance for different radii.

    Args:
    - subfaces (list): List of subfaces.
    - radii (list): List of radii.
    - rank_matrix (np.ndarray): Rank matrix.
    - eps (float): Epsilon value for feature extraction.

    Returns:
    - tuple: Mean distances and number of extractions.
    """
    subfaces_vector = ut.list_subfaces_to_vector(subfaces, rank_matrix.shape[1])
    mean_distances = []
    num_extracted = []

    for radius in radii:
        distances = []
        binary_matrix = ut.binary_large_features(rank_matrix, radius, eps)
        for row in binary_matrix:
            distances.append(np.min(
                fca.setDistance_subface_to_matrix(row, subfaces_vector)))
        mean_distances.append(np.sum(distances) / len(binary_matrix))
        num_extracted.append(binary_matrix.shape[0])

    return mean_distances, num_extracted



def subfaces_complement(subfaces, dimension):
    """
    Returns the complement of each subface in the list.

    Args:
    - subfaces (list): List of subfaces.
    - dimension (int): Dimensionality of the subfaces.

    Returns:
    - list: List of complement subfaces.
    """
    return [list(set(range(dimension)) - set(subface)) for subface in subfaces]



def dictionary_by_size(all_subfaces):
    """
    Creates a dictionary of subfaces grouped by their size.

    Args:
    - all_subfaces (list): List of subfaces.

    Returns:
    - dict: Dictionary of subfaces by size.
    """
    sizes = np.array(list(map(len, all_subfaces)))
    unique_sizes = list(set(sizes))
    subfaces_dict = {size: np.array(
        all_subfaces, dtype=object)[np.nonzero(sizes == size)]
                  for size in unique_sizes}

    return subfaces_dict



def subfaces_conversion(subfaces):
    """
    Converts subfaces to a format with consecutive feature indices.

    Args:
    - subfaces (list): List of subfaces.

    Returns:
    - list: Converted list of subfaces.
    """
    features = list({j for subface in subfaces for j in subface})
    features_dict = {feature: index for index, feature in enumerate(features)}

    return [[features_dict[j] for j in subface] for subface in subfaces]


def subfaces_reconversion(subfaces, features):
    """
    Reconverts subfaces to their original feature indices.

    Args:
    - subfaces (list): List of converted subfaces.
    - features (list): List of original features.

    Returns:
    - list: Reconverted list of subfaces.
    """
    return [[features[j] for j in subface] for subface in subfaces]


def suppress_sub_subfaces(subfaces):
    """
    Removes subfaces that are subsets of other subfaces.

    Args:
    - subfaces (list): List of subfaces.

    Returns:
    - list: List of subfaces with subsets removed.
    """
    new_list = []
    for subface in subfaces:
        is_subset = False
        for other_subface in subfaces:
            if len(other_subface) > len(subface):
                if (len(set(other_subface) - set(subface)) ==
                    len(other_subface) - len(subface)):
                    is_subset = True
        if not is_subset:
            new_list.append(subface)

    return new_list


def check_if_in_list(subfaces_list, subface):
    """
    Checks if a subface is in the list of subfaces.

    Args:
    - subfaces_list (list): List of subfaces.
    - subface (list): Subface to check.

    Returns:
    - bool: True if subface is in the list, False otherwise.
    """
    return any(set(existing_subface) == set(subface) for
               existing_subface in subfaces_list)


def all_subsets_of_size(subfaces_list, size):
    """
    Returns all subsets of a given size from the list of subfaces.

    Args:
    - subfaces_list (list): List of subfaces.
    - size (int): Size of subsets.

    Returns:
    - list: List of subsets.
    """
    subsets_list = []
    for subface in subfaces_list:
        if len(subface) == size:
            if not check_if_in_list(subsets_list, subface):
                subsets_list.append(subface)
        if len(subface) > size:
            for sub_subface in it.combinations(subface, size):
                if not check_if_in_list(subsets_list, subface):
                    subsets_list.append(list(sub_subface))

    return subsets_list


def subfaces_to_test(subfaces_dict, dimension):
    """
    Generates candidate subfaces to test based on the given
    dictionary of subfaces.

    Args:
    - subfaces_dict (dict): Dictionary of subfaces by size.
    - dimension (int): Dimensionality of the subfaces.

    Returns:
    - dict: Dictionary of candidate subfaces to test.
    """
    all_subfaces = {2: [subface for subface in it.combinations(
        range(dimension), 2)]}
    for size in list(subfaces_dict.keys())[1:]:
        all_subfaces[size] = ut.candidate_subfaces(
            subfaces_dict[size - 1], size - 1, dimension)

    return all_subfaces


def indices_of_true_subfaces(all_subfaces, true_subfaces):
    """
    Returns the indices of true subfaces in the list of all subfaces.

    Args:
    - all_subfaces (list): List of all subfaces.
    - true_subfaces (list): List of true subfaces.

    Returns:
    - np.ndarray: Array of indices.
    """
    indices = []
    for subface in true_subfaces:
        for index, test_subface in enumerate(all_subfaces):
            if set(subface) == set(test_subface):
                indices.append(index)
                break

    return np.array(indices)


def dictionary_of_false_subfaces(true_subfaces_dict, dimension):
    """
    Creates a dictionary of false subfaces based on the true
    subfaces dictionary.

    Args:
    - true_subfaces_dict (dict): Dictionary of true subfaces by size.
    - dimension (int): Dimensionality of the subfaces.

    Returns:
    - dict: Dictionary of false subfaces by size.
    """
    subfaces_to_test_dict = subfaces_to_test(true_subfaces_dict, dimension)
    false_subfaces_dict = {}

    for size in true_subfaces_dict.keys():
        true_indices = indices_of_true_subfaces(subfaces_to_test_dict[size],
                                                true_subfaces_dict[size])
        false_indices = list(set(range(len(subfaces_to_test_dict[size]))) -
                             set(true_indices))
        if np.array(subfaces_to_test_dict[size])[false_indices].size > 0:
            false_subfaces_dict[size] = np.array(
                subfaces_to_test_dict[size])[false_indices]

    return false_subfaces_dict


def all_sub_subfaces(subfaces):
    """
    Returns all sub-subfaces of the given subfaces.

    Args:
    - subfaces (list): List of subfaces.

    Returns:
    - list: List of all sub-subfaces.
    """
    all_subfaces = []
    for subface in subfaces:
        subface_size = len(subface)
        if subface_size == 2:
            all_subfaces.append(subface)
        else:
            for size in range(2, subface_size):
                for subset in it.combinations(subface, size):
                    all_subfaces.append(subset)
            all_subfaces.append(subface)

    sizes = list(map(len, all_subfaces))
    all_subfaces = np.array(all_subfaces)[np.argsort(sizes)]

    return list(map(list, set(map(tuple, all_subfaces))))


# ###########################
# Subfaces frequency analysis #
# ###########################

def rhos_subface_pairs(binary_matrix, subface, k):
    """
    Computes rho(i,j) for each pair (i,j) in the subface.

    Args:
    - binary_matrix (np.ndarray): Binary matrix.
    - subface (list): Subface to evaluate.
    - k (int): Number of samples.

    Returns:
    - dict: Dictionary of rho values for each pair.
    """
    rhos_subface = {}
    for (i, j) in it.combinations(subface, 2):
        rhos_subface[i, j] = ut.rho_value(binary_matrix, [i, j], k)

    return rhos_subface


def r_partial_derivative_centered(matrix_k, matrix_kp, matrix_km, subface, k):
    """
    Computes the centered partial derivative of r for each feature in
    the subface.

    Args:
    - matrix_k (np.ndarray): Base matrix.
    - matrix_kp (np.ndarray): Matrix with incremented values.
    - matrix_km (np.ndarray): Matrix with decremented values.
    - subface (list): Subface to evaluate.
    - k (int): Number of samples.

    Returns:
    - dict: Dictionary of partial derivatives for each feature.
    """
    r_derivatives = {}
    for feature in subface:
        matrix_right = ut.partial_matrix(matrix_k, matrix_kp, feature)
        matrix_left = ut.partial_matrix(matrix_k, matrix_km, feature)
        r_derivatives[feature] = 0.5 * k**0.25 * (
            ut.rho_value(matrix_right, subface, k) -
            ut.rho_value(matrix_left, subface, k))

    return r_derivatives


# ########################################
# ### Displaced from clef.py
# ########################################

# def list_subfaces_to_bin_matrix(subfaces, dimension):
#     """
#     Converts a list of subface indices into a binary matrix.

#     Args:
#     - subfaces (list): List of subfaces.
#     - dimension (int): Dimensionality of the ambient space.

#     Returns:
#     - np.ndarray: Binary matrix representation of subfaces.
#     """
#     num_subfaces = len(subfaces)
#     vector_subfaces = np.zeros((num_subfaces, dimension))

#     for subface_index, subface in enumerate(subfaces):
#         vector_subfaces[subface_index, subface] = 1.0

#     return vector_subfaces
 

def khi(binary_data, face):
    face_vect_tmp = binary_data[:, face]
    face_exist = float(np.sum(np.sum(face_vect_tmp, axis=1) > 0))
    all_face = np.sum(np.prod(face_vect_tmp, axis=1))

    return all_face/face_exist


# ########################################
# ## subfaces frequency analysis from clef.py
# ########################################


def init_freq(x_bin_k, k, f_min):
    """Return all faces of size 2 s.t. frequency > f_min."""
    dim = x_bin_k.shape[1]
    faces = []
    for (i, j) in it.combinations(range(dim), 2):
        face = [i, j]
        r_alph = ut.rho_value(x_bin_k, face, k)
        if r_alph > f_min:
            faces.append(face)

    return faces


def find_faces_freq(x_bin_k, k, f_min):
    """Return all faces s.t. frequency > f_min."""
    dim = x_bin_k.shape[1]
    faces_pairs = init_freq(x_bin_k, k, f_min)
    size = 2
    faces_dict = {}
    faces_dict[size] = faces_pairs
    while len(faces_dict[size]) > size:
        faces_dict[size + 1] = []
        faces_to_try = ut.candidate_subfaces(faces_dict[size], size, dim)
        if faces_to_try:
            for face in faces_to_try:
                r_alph = ut.rho_value(x_bin_k, face, k)
                if r_alph > f_min:
                    faces_dict[size + 1].append(face)
        size += 1

    return faces_dict


def freq_0(x_bin_k, k, f_min):
    """Return maximal faces s.t. frequency > f_min."""
    faces_dict = find_faces_freq(x_bin_k, k, f_min)

    return clef.find_maximal_faces(faces_dict)


###########################
# Discarded functions
def entropy(masses, total_mass):
    """
    Calculates the entropy of a distribution defined by `masses`.

    Parameters:
    - masses (list or np.ndarray): List of masses.
    - total_mass (float): Total mass of the distribution.

    Returns:
    - float: Entropy value.
    """
    k = len(masses)
    if k <= 1:
        return np.inf
    if isinstance(masses, list):
        masses = np.array(masses)
    if total_mass is None:
        total_mass = np.sum(masses)

    distrib = masses[masses > 0] / total_mass
    log_distrib = np.log(distrib)
    neg_product = - distrib * log_distrib
    return np.sum(neg_product)  # / np.log(k)


def AIC_masses(masses, number_extremes, total_mass):
    """
    Calculates the Akaike Information Criterion (AIC) for a given
    distribution of masses.

    Parameters:
    - masses (list or np.ndarray): List of masses.
    - number_extremes (int): Number of extreme points.
    - total_mass (float): Total mass of the distribution.

    Returns:
    - float: AIC value.
    """
    k = len(masses)
    if k <= 1:
        return 0
    normalized_negative_log_likelihood = entropy(masses, total_mass=total_mass)
    return 2 * (normalized_negative_log_likelihood + k/number_extremes)

def clef_select_kappa_AIC(vect_kappa, X, radial_threshold,
                          # include_singletons=True,
                          standardize,
                          unstable_kappa_max=0.05, plot=False):
    """
    Selects the optimal kappa value based on AIC.

    Parameters:
    - vect_kappa (list): List of kappa values to test.
    - X (np.ndarray): Input data.
    - radial_threshold (float): Radius threshold for identifying extreme
      samples.
    - standardize (bool): Whether to standardize the data.
    - unstable_kappa_max (float): Maximum kappa value for unstable solutions.
    - plot (bool): Whether to plot the AIC values.

    Returns:
    - float: Selected kappa value.
    """
    if standardize:
        radii = np.max(rank_transform(X), axis=1)
    else:
        radii = np.max(X, axis=1)
    num_extremes = np.sum(radii >= radial_threshold)
    total_mass = radial_threshold * num_extremes / X.shape[0]
    ntests = len(vect_kappa)
    vect_aic = np.zeros(ntests)
    counter = 0
    for kappa in vect_kappa:
        clef_faces = ut.clef_fit(X, radial_threshold, kappa,
                                 standardize=standardize,
                                 include_singletons=False)
        list_of_masses = ut.clef_estim_subfaces_mass(clef_faces, X,
                                                     radial_threshold,
                                                     standardize=standardize)
        vect_aic[counter] = AIC_masses(list_of_masses, num_extremes,
                                       total_mass)
        # selection based on AIC
        # vect_aic below is
        # proportional to : - log-likelihood(categorical model) + k where k
        # is the number of categories, ie the number of subfaces.
        # vect_aic = vect_entropy + vect_number_faces / num_extremes
        counter += 1

    i_maxerr = np.argmax(vect_aic[vect_kappa < unstable_kappa_max])
    kappa_maxerr = vect_kappa[i_maxerr]
    i_mask = vect_kappa <= kappa_maxerr
    i_minAIC = np.argmin(vect_aic + (1e+23) * i_mask)
    kappa_select_aic = vect_kappa[i_minAIC]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.xlabel('kappa_min')
        plt.ylabel('AIC')
        plt.title('AIC versus kappa_min')
        plt.scatter(vect_kappa, vect_aic, c='gray', label='AIC')
        plt.plot([kappa_select_aic, kappa_select_aic],
                 [0, max(vect_aic)], c='red')
        plt.grid(True)
        plt.show()

    print(f'CLEF: selected kappa_min with AIC method: \
        {round_signif(kappa_select_aic, 2)}')
    return kappa_select_aic

# def setDistance_subface_to_list(subface, subfaces_list, normalize=True):
#     """
#     Computes the pseudo-Subface distance between one subface and a
#     matrix of subfaces.

#     Args:
#     - subface (np.ndarray): Single subface vector.
#     - subfaces_list (list): list of subface vectors

#     Returns:
#     - np.ndarray: Array of distances.
#     """
#     subfaces_matrix = ut.subfaces_list_to_matrix(subfaces_list)
#     return setDistance_subface_to_matrix(subface, subfaces_matrix, normalize)

# def setDistance_error_l2l(subfaces_list, masses, subfaces_reference_list,
#                           reference_masses, total_reference_mass,
#                           dimension,
#                           dispersion_model, rate
#                           ):
#     """
#     Computes the average minimum set distance between
#      all entries of subfaces_list and the reference list
#     subfaces_reference_list.

#     The results is a weighted average of the minimum set distances,
#     weighted by the corresponding normalized `mass` entry in the
#     considered subfaces in `subfaces_list'.

#     The lower, the better.

#     Parameters:
#     - subfaces_list (list): List of subfaces.
#     - masses (list or np.ndarray): Masses associated with subfaces.
#     - subfaces_reference_list (list): Reference list of subfaces.
#     - reference_masses (list or np.ndarray): Masses associated with reference
#       subfaces.
#     - total_reference_mass (float): Total mass of reference subfaces.
#     - dimension (int): Dimensionality of the space.
#     - dispersion_model (bool): Whether to use a dispersion model.
#     - rate (float, >0): Rate parameter for deviance calculation.

#     Returns:
#     - float: Average minimum set distance.
#     """
#     if len(subfaces_list) == 0:
#         return 0
#     if len(subfaces_reference_list) == 0:
#         return float('inf')
#     subfaces_matrix = subfaces_list_to_matrix(subfaces_list, dimension)
#     subfaces_reference_matrix = subfaces_list_to_matrix(
#         subfaces_reference_list, dimension)
#     return total_deviance_binary_matrices(subfaces_matrix, masses,
#                                  subfaces_reference_matrix,
#                                  reference_masses, total_reference_mass,
#                                  dispersion_model, rate)
