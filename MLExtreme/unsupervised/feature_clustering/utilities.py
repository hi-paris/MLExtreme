from __future__ import division
# import random as rd
import numpy as np
from ...utils.EVT_basics import rank_transform
# from .ftclust_analysis import subfaces_list_to_matrix


def subfaces_list_to_matrix(subfaces, dimension=None):
    """
    Converts a list of subfaces into a binary matrix.

    Args:
    - subfaces (list): List of subfaces.

    Returns:
    - np.ndarray: Binary matrix representation of subfaces.
    """
    num_subfaces = len(subfaces)

    if dimension is None:
        features = list({j for subface in subfaces for j in subface})
        dimension = int(max(features))+1

    matrix_subfaces = np.zeros((num_subfaces, dimension))

    for subface_index, subface in enumerate(subfaces):
        matrix_subfaces[subface_index, subface] = 1

    return matrix_subfaces # [:, np.sum(matrix_subfaces, axis=0) > 0]


def remove_zero_rows(binary_array):
    """
    Removes rows that contain only zeros from a binary matrix.

    Parameters:
    - binary_array (np.ndarray): A binary matrix from which to remove zero
        rows.

    Returns:
    - np.ndarray: The binary matrix with zero rows removed.
    """
    # Remove rows where the sum of elements is zero (i.e., all
    # elements are zero)
    return binary_array[np.sum(binary_array, axis=1) > 0]


def binary_large_features(std_data, radial_threshold, epsilon=None):
    """
    Filters extreme samples from the input `std_data`, returns a
    binary matrix of same number of rows as the number of extreme
    samples, where each row is a binary vector indicating which
    components of the extreme sample point are large.  Here the radius
    of a sample point is its infinite norm and a sample point is
    extreme if its radius is greater than `radial_threshold'.  Also a
    sample's component is 'large' means that it is greater than
    `epsilon * radial_threshold` .

    Parameters:

    - std_data (np.ndarray): A data matrix. For meaningful usage,
      columns should be preliminarily sdandardized.

    - radial_threshold (float): The threshold value to compare against.

    - epsilon (float, optional): A tolerance parameter between 0 and 1
        for the radius threshold. If None, treated as 1. A value of
        zero will have the function declare as 'large' every
        components in extreme sample points. On the contrary, with
        `epsilon=1`, a component needs to exceed the radial threshold
        to be declared large.

    Returns:

    - np.ndarray: A binary matrix with values above the
    radius threshold, with zero rows removed.

    Note: For standardizing a raw data input, see mlx.rank_transform,
    mlx.rank_transform_test.

    """
    if epsilon is not None:
        # Create a binary matrix where values are above the radial
        # threshold rescaled by epsilon, after selecting samples which
        # radius exceeds the radial threshold.
        extreme_samples = std_data[np.max(std_data, axis=1) >=
                                   radial_threshold]
        binary_matrix = extreme_samples > radial_threshold * epsilon
    else:
        # Create a binary matrix which values are  above the
        # radius threshold
        binary_matrix = std_data >= radial_threshold

    # Convert the binary matrix to float type and remove zero rows
    return remove_zero_rows(binary_matrix.astype(int))
 

def estim_subfaces_mass(subfaces_list, X, threshold, epsilon,
                        standardize=True):
    if standardize:
        x_norm = rank_transform(X)
    else:
        x_norm = X
    x_bin = binary_large_features(x_norm, threshold, epsilon)
    dimension = np.shape(x_bin)[1]
    mass_list = []
    subfaces_matrix = subfaces_list_to_matrix(subfaces_list, dimension)
    #pdb.set_trace()
    for k in range(len(subfaces_list)):
        subface = subfaces_matrix[k]
        shared_features = np.dot(x_bin, subface.reshape(-1, 1)) #(num_extremes,1)
        num_features_samples = np.sum(x_bin, 1).reshape(-1, 1) #(num_extremes,1)
        num_features_subface = np.sum(subface) #int 
        sample_superset_of_subface = num_features_subface == shared_features
        sample_subset_of_subface = num_features_samples == shared_features
        sample_equal_subface = sample_subset_of_subface * \
            sample_superset_of_subface
        counts_subface = np.sum(sample_equal_subface)
        mass = threshold * counts_subface / X.shape[0]
        mass_list.append(float(mass))
        
    return mass_list


# %%
def entropy(masses, total_mass = None):
    k = len(masses)
    # if k == 0:
    #     raise Exception("empty list found by damex")
    #     # pdb.set_trace()
    if k <= 1:
        return 0
    if isinstance(masses, list):
        masses = np.array(masses)
    # if total_mass is not None:
    #     missing_mass = total_mass - np.sum(masses)
    #     if missing_mass < -1e-7:
    #         raise ValueError('total mass should exceed sum of masses')
    #     if missing_mass > 0:
    #         masses = np.append(masses, missing_mass)
        
    # distrib = masses[masses > 0] / (np.sum(masses) + missing_mass)
    if total_mass is None:
        total_mass = np.sum(masses)

    distrib = masses[masses > 0] / total_mass
    log_distrib = np.log(distrib)
    neg_product = - distrib * log_distrib
    return np.sum(neg_product)  # / np.log(k)

def AIC_clustering(masses, number_extremes, total_mass=None):
    k = len(masses)
    if k <= 1:
        return 0
    negllkl = number_extremes * entropy(masses, total_mass=total_mass)
    return negllkl + k
