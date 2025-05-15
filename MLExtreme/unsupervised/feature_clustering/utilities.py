
from __future__ import division
# import random as rd
#import pdb
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import combinations
import networkx as nx
from sklearn.model_selection import KFold
from MLExtreme.utils.EVT_basics import rank_transform, round_signif
# from .ftclust_analysis import subfaces_list_to_matrix

def subfaces_list_to_matrix(subfaces, dimension):
    """
    Converts a list of subfaces into a binary matrix.

    Parameters:
    -----------
    - subfaces (list): List of subfaces.
    - dimension (int): Dimensionality of the ambient space.

    Returns:
    --------
    - np.ndarray: Binary matrix representation of subfaces.
    """
    num_subfaces = len(subfaces)

    if dimension is None:
        features = list({j for subface in subfaces for j in subface})
        dimension = int(max(features)) + 1

    matrix_subfaces = np.zeros((num_subfaces, dimension))

    for subface_index, subface in enumerate(subfaces):
        matrix_subfaces[subface_index, subface] = 1

    return matrix_subfaces  # [:, np.sum(matrix_subfaces, axis=0) > 0]


def remove_zero_rows(binary_array):
    """
    Removes rows that contain only zeros from a binary matrix.

    Parameters:
    -----------
    - binary_array (np.ndarray): A binary matrix from which to remove zero
        rows.

    Returns:
    --------
    - np.ndarray: The binary matrix with zero rows removed.
    """
    # Remove rows where the sum of elements is zero (i.e., all
    # elements are zero)
    return binary_array[np.sum(binary_array, axis=1) > 0]


def binary_large_features(std_data, radial_threshold, epsilon):
    """
    Filters extreme samples from the input `std_data`, returns a
    binary matrix of same number of rows as the number of extreme
    samples, where each row is a binary vector indicating which
    components of the extreme sample point are large. Here the radius
    of a sample point is its infinite norm and a sample point is
    extreme if its radius is greater than `radial_threshold'. Also a
    sample's component is 'large' means that it is greater than
    `epsilon * radial_threshold'.

    Parameters:
    -----------
    - std_data (np.ndarray): A data matrix. For meaningful usage,
      columns should be preliminarily standardized.
    - radial_threshold (float): The threshold value to compare against.
    - epsilon (float, optional): A tolerance parameter between 0 and 1
        for the radius threshold. If None, treated as 1. A value of
        zero will have the function declare as 'large' every
        components in extreme sample points. On the contrary, with
        `epsilon=1`, a component needs to exceed the radial threshold
        to be declared large.

    Returns:
    -------
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
                        standardize):
    """
    Estimates the mass of each subface in `subfaces_list` based on the
    data `X`.

    Parameters:
    - subfaces_list (list): List of subfaces.
    - X (np.ndarray): Input data.
    - threshold (float): Threshold for identifying extremes.
    - epsilon (float): Tolerance parameter for identifying large
        components.
    - standardize (bool): Whether to standardize the data.

    Returns:
    - list: Estimated masses for each subface.
    """
    if standardize:
        x_norm = rank_transform(X)
    else:
        x_norm = X
    x_bin = binary_large_features(x_norm, threshold, epsilon)
    dimension = np.shape(x_bin)[1]
    mass_list = []
    subfaces_matrix = subfaces_list_to_matrix(subfaces_list, dimension)

    for k in range(len(subfaces_list)):
        subface = subfaces_matrix[k]
        shared_features = np.dot(x_bin, subface.reshape(-1, 1))
        # (num_extremes, 1)
        num_features_samples = np.sum(x_bin, 1).reshape(-1, 1)
        # (num_extremes, 1)
        num_features_subface = np.sum(subface)  # int
        sample_superset_of_subface = num_features_subface == shared_features
        sample_subset_of_subface = num_features_samples == shared_features
        sample_equal_subface = sample_subset_of_subface * \
            sample_superset_of_subface
        counts_subface = np.sum(sample_equal_subface)
        mass = threshold * counts_subface / X.shape[0]
        mass_list.append(float(mass))

    return np.array(mass_list)


##################################################
# DAMEX functions
##################################################

def damex_0(binary_matrix, include_singletons):
    """
    Analyzes a binary matrix to determine the number of points (row) matching
    patterns in {0,1}^d where d is the number of columns in the matrix.

    Parameters:
    - binary_matrix (np.ndarray): A binary matrix where rows represent samples
        and columns represent features.
    - include_singletons (bool): Whether to include singletons in the analysis.

    Returns:
    - faces (list of lists): A list of subfaces of the infinite sphere,
        each represented as a list of feature indices.
    - mass (np.ndarray): An array indicating the number of rescaled
        samples observed within each subface.
    """
    # Set the minimum number of features per subface
    if include_singletons:
        min_features = 1
    else:
        min_features = 2

    # Sample size
    num_samples = binary_matrix.shape[0]
    # Number of features (columns) with value 1 for each
    # sample (row)
    num_features_per_sample = np.sum(binary_matrix, axis=1)

    # Dot product of the binary matrix with its
    # transpose: get shared features between samples.
    # Entry ij is the number of features shared by samples i and j.
    shared_features_matrix = np.dot(binary_matrix, binary_matrix.T)

    # Determine samples with exactly matching features
    # Entry ij is one if (binary) samples i and j are identical
    exact_match_matrix = (
        shared_features_matrix == num_features_per_sample) & (
            shared_features_matrix.T == num_features_per_sample).T

    # Set of samples not yet observed/assigned in any subface
    uncovered_samples = set(range(num_samples))

    # Dictionary to store the number of samples assigned to each subface
    subface_sample_count = {}

    # Iterate over each sample to identify subfaces
    for i in range(num_samples):
        # Find samples with exactly matching features
        matching_samples = list(np.nonzero(exact_match_matrix[i, :])[0])

        # If the current sample has not been assigned yet
        if i in uncovered_samples:
            # Mark these samples as assigned
            uncovered_samples -= set(matching_samples)

            # If the sample has more than one feature, record the
            # subface (add a key to the dictionary with value equal to
            # the number of samples in this subface. The key is the
            # index of the first record of the event that a sample
            # point belongs to the considered subface.
            if num_features_per_sample[i] >= min_features:
                subface_sample_count[i] = len(matching_samples)

    # Sort subfaces by the number of samples they cover, in descending order
    sorted_indices = np.argsort(list(subface_sample_count.values()))[::-1]

    # Create a (sorted) list of subfaces, each represented by the
    # indices of `1` features
    subfaces = [list(np.nonzero(
        binary_matrix[list(subface_sample_count)[i], :])[0])
                for i in sorted_indices]

    converted_subfaces = [[int(item) for item in sublist]
                          for sublist in subfaces]

    # Create an array of the number of samples covered by each subface
    counts = [list(subface_sample_count.values())[i] for i in sorted_indices]

    return converted_subfaces, np.array(counts)

def damex_fit(data, threshold, epsilon, min_counts, standardize,
              include_singletons):
    """
    Implements the methodology proposed in [1].

    Given a data matrix (n,d), damex provides the groups of features that are
    likely to be simultaneously large, while the other features are small. The
    function also returns an estimate of the limit measure associated with each
    subgroup.

    Parameters:
    - data (np.ndarray): A data matrix.
    - threshold (float): The radial threshold for selecting extreme
      samples.
    - epsilon (float): A scaling factor for the radial threshold.
    - min_counts (int): The minimum number of points required per face.
        Defaults to 0.
    - standardize (bool): Defaults to True. If True, the data matrix will be
      standardized using the rank transformation mlx.rank_transform. Meaningful
      usage requires that columns (features) follow a unit Pareto distribution
        or
      that rank-standardization has been applied previously with
      `standardize=False`, or that columns are possibly not standard and
      `standardize=True`.

    Returns:
    - list: A list of faces where each face has more than min_points points.

    References:
    [1] Goix, N., Sabourin, A., & Clémençon, S. (2017). Sparse representation
    of multivariate extremes with applications to anomaly detection.
    Journal of Multivariate Analysis, 161, 12-31.
    (The returned estimator is defined in
    Equation (3.3) from the reference)
    """
    # Standardize data if necessary
    if standardize:
        intern_data = rank_transform(data)
    else:
        intern_data = data

    # Generate binary matrix and determine faces and their masses
    binary_matrix = binary_large_features(intern_data, threshold,
                                          epsilon)
    faces, counts = damex_0(binary_matrix, include_singletons)
    n = intern_data.shape[0]

    limit_mass_estimator = threshold * counts / n
    id_large_mass = (counts >= min_counts)
    number_heavy_faces = np.sum(id_large_mass)
    truncated_faces = faces[:number_heavy_faces]
    truncated_mass_estimator = limit_mass_estimator[:number_heavy_faces]

    # Return faces with mass greater than or equal to min_points
    return truncated_faces, truncated_mass_estimator

# def damex_estim_subfaces_mass(subfaces_list, X, threshold, epsilon,
#                               standardize=True):
#     res = estim_subfaces_mass(subfaces_list, X, threshold, epsilon,
#                               standardize=standardize)
#     return res

##################################################
# # CLEF
##################################################

#############
# Clef algo #
#############

def clef_fit(X, radius, kappa_min, standardize, include_singletons):
    """
    Returns maximal faces such that kappa > kappa_min.

    Parameters:
    - X (np.ndarray): Input data.
    - radius (float): Radius threshold for selecting extreme samples.
    - kappa_min (float): Minimum kappa value.
    - standardize (bool): Whether to standardize the data.
    - include_singletons (bool): Whether to include singletons.

    Returns:
    - list: List of maximal faces.
    """
    if standardize:
        x_norm = rank_transform(X)
    else:
        x_norm = X
    x_bin = binary_large_features(x_norm, radius, epsilon=None)
    result = clef_fit0(x_bin, kappa_min, include_singletons)
    return result

def clef_fit0(x_bin, kappa_min, include_singletons):
    """
    Returns maximal faces such that kappa > kappa_min.

    Parameters:
    - x_bin (np.ndarray): Binary matrix of extreme samples.
    - kappa_min (float): Minimum kappa value.
    - include_singletons (bool): Whether to include singletons.

    Returns:
    - list: List of maximal faces.
    """
    faces_dict = find_faces(x_bin, kappa_min)
    faces = find_maximal_faces(faces_dict)

    if include_singletons:
        dim = np.shape(x_bin)[1]
        missed_singletons = cons_missing_singletons(faces, dim)
        for sing in missed_singletons:
            faces.append(sing)
    return faces

##################
# CLEF functions #
##################

def cons_missing_singletons(faces, d):
    """
    Constructs missing singletons from a list of faces.

    Parameters:
    - faces (list): List of faces.
    - d (int): Dimensionality of the space.

    Returns:
    - list: List of missing singletons.
    """
    # Create a set of all possible indices
    all_indices = set(range(d))
    # Iterate through each sublist in faces and remove its elements
    # from all_indices
    for sublist in faces:
        all_indices.difference_update(sublist)

    # Convert the remaining indices to a list and return
    list_indices = list(all_indices)
    return list([index] for index in list_indices)

def faces_init(x_bin, mu_0):
    """
    Returns faces of size 2 such that kappa > kappa_min.

    Parameters:
    - x_bin (np.ndarray): Binary matrix of extreme samples.
    - mu_0 (float): Minimum kappa value.

    Returns:
    - list: List of asymptotic pairs.
    """
    asymptotic_pair = []
    for (i, j) in combinations(range(x_bin.shape[1]), 2):
        pair_tmp = x_bin[:, [i, j]]
        one_out_of_two = np.sum(np.sum(pair_tmp, axis=1) > 0)
        two_on_two = np.sum(np.prod(pair_tmp, axis=1))
        if one_out_of_two > 0:
            proba = two_on_two / one_out_of_two
            if proba > mu_0:
                asymptotic_pair.append([i, j])

    return asymptotic_pair

def compute_beta(x_bin, face):
    """
    Computes the beta value for a given face.

    Parameters:
    - x_bin (np.ndarray): Binary matrix of extreme samples.
    - face (list): Face to evaluate.

    Returns:
    - int: Beta value.
    """
    return np.sum(np.sum(x_bin[:, face], axis=1) > len(face)-2)

def kappa(x_bin, face):
    """
    Returns the kappa value.

    kappa = #{i | for all j in face, X_ij=1} /
    #{i | at least |face|-1 j, X_ij=1}

    Parameters:
    - x_bin (np.ndarray): Binary matrix of extreme samples.
    - face (list): Face to evaluate.

    Returns:
    - float: Kappa value.
    """
    beta = compute_beta(x_bin, face)
    all_face = np.sum(np.prod(x_bin[:, face], axis=1))
    if beta == 0.:
        kap = 0.
    else:
        kap = all_face / float(beta)

    return kap

def find_faces(x_bin, kappa_min):
    """
    Returns all faces such that kappa > kappa_min.

    Parameters:
    - x_bin (np.ndarray): Binary matrix of extreme samples.
    - kappa_min (float): Minimum kappa value.

    Returns:
    - dict: Dictionary of faces.
    """
    dim = x_bin.shape[1]
    size = 2
    faces_dict = {}
    faces_dict[size] = faces_init(x_bin, kappa_min)
    # print('face size : nb faces')
    while len(faces_dict[size]) > size:
        # print(size, ':', len(faces_dict[size]))
        faces_dict[size + 1] = []
        faces_to_try = candidate_subfaces(faces_dict[size], size, dim)
        if faces_to_try:
            for face in faces_to_try:
                if kappa(x_bin, face) > kappa_min:
                    faces_dict[size + 1].append(face)
        size += 1

    return faces_dict

def find_maximal_faces(faces_dict, lst=True):
    """
    Returns inclusion-wise maximal faces.

    Parameters:
    - faces_dict (dict): Dictionary of faces.
    - lst (bool): Whether to return a list of faces.

    Returns:
    - list: List of maximal faces.
    """
    # k = len(faces_dict.keys()) + 1
    if len(faces_dict) == 0:
        return []
    list_keys = []
    for k in faces_dict.keys():
        list_keys.append(k)
    list_keys.sort(reverse=True)
    k = list_keys[0]
    maximal_faces = [faces_dict[k]]
    faces_used = list(map(set, faces_dict[k]))
    # for i in list_keys[:-1]:  ##range(1, k - 1):
    for j in list_keys[1:]:
        face_tmp = list(map(set, faces_dict[j]))
        for face in faces_dict[j]:
            for face_test in faces_used:
                if len(set(face) & face_test) == j:
                    face_tmp.remove(set(face))
                    break
        maximal_faces.append(list(map(list, face_tmp)))
        faces_used = faces_used + face_tmp
    maximal_faces = maximal_faces[::-1]
    if lst:
        maximal_faces = [face for faces_ in maximal_faces
                         for face in faces_]

    return maximal_faces

##################
# Generate Candidate Subfaces of Increased Size at Each CLEF Iteration
##################

def make_graph(subfaces, size, dimension):
    """
    Creates a graph where nodes represent subfaces and edges exist if subfaces
    differ by at most one feature.

    It is the main building block of the `candidate_sufaces' function
    used in CLEF and variants [1,2].

    Parameters:
    - subfaces (list): List of subfaces.
    - size (int): Size of the subfaces.
    - dimension (int): Dimensionality of the subfaces.

    Returns:
    - nx.Graph: Graph representation of subfaces.

    References
    -----------

    [1] Chiapino, M., & Sabourin, A. (2016,September). Feature clustering
    for extreme events analysis, with application to extreme stream-flow data.
    In International workshop on new frontiers in mining complex patterns
    (pp. 132-147). Cham: Springer International Publishing.

    [2] Chiapino, M., Sabourin, A., & Segers, J. (2019). Identifying groups of
    variables with the potential of being large simultaneously.
    Extremes, 22, 193-222.
    """
    vector_subfaces = subfaces_list_to_matrix(subfaces, dimension)
    num_subfaces = len(vector_subfaces)
    graph = nx.Graph()
    nodes = range(num_subfaces)
    graph.add_nodes_from(nodes)
    edges = np.nonzero(np.triu(
        np.dot(vector_subfaces, vector_subfaces.T) == size - 1))
    graph.add_edges_from([(edges[0][i], edges[1][i])
                          for i in range(len(edges[0]))])

    return graph

def candidate_subfaces(subfaces, size, dimension):
    """
    Generates a list A_{s+1} of candidate subfaces of size s+1
    from a list A_s = `subfaces' of subfaces of size s, with `s=size`.
    Candidate subfaces are all subfaces of size s+1 which are
    supersets of each of subface of size s in the current list
    `subfaces'.

    This is a key step in CLEF algorithm [1] and variants [2]

    Parameters:
    - subfaces (list): List of subfaces.
    - size (int): Size of the subfaces.
    - dimension (int): Dimensionality of the subfaces.

    Returns:
    - list: List of candidate subfaces.

    References
    -----------

    [1] Chiapino, M., & Sabourin, A. (2016,September). Feature clustering
    for extreme events analysis, with application to extreme stream-flow data.
    In International workshop on new frontiers in mining complex patterns
    (pp. 132-147). Cham: Springer International Publishing.

    [2] Chiapino, M., Sabourin, A., & Segers, J. (2019). Identifying groups of
    variables with the potential of being large simultaneously.
    Extremes, 22, 193-222.
    """
    graph = make_graph(subfaces, size, dimension)
    candidate_subfaces_list = []
    cliques = list(nx.find_cliques(graph))
    indices_to_try = np.nonzero(np.array(
        list(map(len, cliques))) == size + 1)[0]

    for index in indices_to_try:
        clique_features = set([])
        for subface_index in range(len(cliques[index])):
            clique_features = clique_features | set(
                subfaces[cliques[index][subface_index]])
        clique_features = list(clique_features)
        if len(clique_features) == size + 1:
            candidate_subfaces_list.append(clique_features)

    return candidate_subfaces_list

############################
# Subfaces frequency analysis #
############################

# def rho_value(binary_matrix, subface, k):
#     """
#     Calculates the rho value of a subface.
#     (notation r_a(1) in [1], where a is the subface)

#     Args:
#     - binary_matrix (np.ndarray): Binary matrix.
#     - subface (list): Subface to evaluate.
#     - k (int): Number of samples.

#     Returns:
#     - float: Rho value.

#     References
#     -----------

#     [1] Chiapino, M., Sabourin, A., & Segers, J. (2019).
#       Identifying groups of variables with the potential of being
#        large simultaneously. Extremes, 22, 193-222.

#     """
#     return np.sum(np.sum(
#         binary_matrix[:, subface], axis=1) == len(subface)) / float(k)

def partial_matrix(base_matrix, partial_matrix, column_index):
    """
    Replaces the column_index column of base_matrix with the corresponding
    column from partial_matrix.

    Used in asymptotic variants of CLEF [1] to compute partial derivatives
    of the 'kappa' and the 'r' functions

    Parameters:
    - base_matrix (np.ndarray): Base matrix.
    - partial_matrix (np.ndarray): Partial matrix.
    - column_index (int): Index of the column to replace.

    Returns:
    - np.ndarray: Modified matrix.

    References
    -----------

    [1] Chiapino, M., Sabourin, A., & Segers, J. (2019). Identifying groups of
    variables with the potential of being large simultaneously.
    Extremes, 22, 193-222.
    """
    matrix_copy = np.copy(base_matrix)
    matrix_copy[:, column_index] = partial_matrix[:, column_index]

    return matrix_copy

def clef_estim_subfaces_mass(subfaces_list, X, threshold,
                             standardize):
    """
    Estimates the mass of each subface in `subfaces_list` based on the
    data `X`.

    Parameters:
    - subfaces_list (list): List of subfaces.
    - X (np.ndarray): Input data.
    - threshold (float): Threshold for identifying extremes.
    - standardize (bool): Whether to standardize the data.

    Returns:
    - list: Estimated masses for each subface.
    """
    res = estim_subfaces_mass(subfaces_list, X, threshold, epsilon=None,
                              standardize=standardize)
    return res


##################################################
# Feature clusters analysis: unit deviance, total deviance,
# dispersion model ...
# previously in ftclust_analysis
##################################################

def unit_deviance(subface, subfaces_matrix):
    """Calculates unit deviances `d` between one subface and a matrix of
    (binary encoded) subfaces.

    The unit deviance `d` between any two subfaces is defined as the
    ratio between the cardinalities of the symmetric difference and
    the union of the two sets.

    This is the key ingredient of the unsupervised performance
    criterion s(list_of_subsets, binary_encoded_new_point) proposed in [1]

    In the present implementation it is the basic building block of
    the total deviance, which serves as convenient unsupervised
    criterion for parameter tuning.

    Parameters:
    -----------
    subface : np.ndarray)
        Single subface vector (binary vector), typically
        a test vector.
    - subfaces_matrix : np.ndarray)
        Matrix of reference subface vectors
      (binary entries) (typically, the ones estimated by damex or clef).

    Returns:
    --------
    - np.ndarray: Array of distances.

    References:
    -----------

    [1] Chiapino, M., Sabourin, A., & Segers, J. (2019). Identifying groups of
    variables with the potential of being large simultaneously.
    Extremes, 22, 193-222.

    """
    result = (np.sum(abs(subface - subfaces_matrix), axis=1) /
              np.sum(subface + subfaces_matrix > 0, axis=1))

    return result

def total_deviance_binary_matrices(subfaces_matrix, masses,
                                   subfaces_reference_matrix,
                                   reference_masses, 
                                   rate):
    """Computes the total deviance between all rows of subfaces_matrix
    and the matrix subfaces_reference_matrix, and returns a normalized
    version of it. If `mass` is not provided, the normalization is
    done by dividing the total deviance by the sample size (i.e. the
    number of rows in subfaces_matrix).  Alternatively, if `mass` is
    provided, the contribution of each row is weighted by the relative
    weight of the corresponding entry in `mass`.

    The total deviance is relative to a dispersion_model. It is the
    (normalized) negative log-likelihood of the rows in
    subfaces_matrix, within a mixture model defined in the set of
    subfaces, as follows:

        - each row in subfaces_reference_matrix is a 'center of mass'
    
        - the references_masses are the (unnormalized) weights of the mixture

    Given that row_i in subfaces_matrix was generated by the mixture
            component reference_row_j in subfaces_reference_matrix,
            the unit deviance (see `unit_deviance`) between row_i and
            reference_row_j follows an exponential distribution with
            rate `rate`. The likelihood term disregards any necessary
            normalizing constants, such as those accounting for the
            truncation of the exponential model or the combinatorial
            number of subfaces at a given set distance from
            reference_row_j, which would be necessary in principle to
            define a proper dispersion model `à la Jorgensen'.

    In unsupervised usages:
        - subfaces_reference_matrix is typically issued from prior
    estimation from DAMEX or CLEF
        - subfaces_matrix is typically a dataset

    In supervised usage:
        Both subfaces_matrix and subfaces_reference matrix are issued from a
        list of true or estimated subfaces and the goal is to assess the
        'distance' between the two.

    The lower, the better fit.

    Parameters:
    - subfaces_matrix (np.ndarray): Matrix of subfaces.
    - masses (list or np.ndarray): Masses associated with subfaces.
    - subfaces_reference_matrix (np.ndarray): Reference matrix of subfaces.
    - reference_masses (list or np.ndarray): Masses associated with reference
      subfaces.
    - rate (float, >0): Rate parameter for deviance calculation.

    Returns:
    - float: Average aggregated set distance.

    """
    n_subfaces = np.shape(subfaces_matrix)[0]
    if n_subfaces == 0:
        return 0
    neg_log_likelihood = 10 * np.ones(n_subfaces)
    if masses is None:
        weights = np.ones(n_subfaces)/n_subfaces
    else:
        if isinstance(masses, list):
            masses = np.array(masses)
        weights = masses / np.sum(masses)
    if reference_masses is None:
        warnings.warn('No reference masses provided. \
        Using uniform weights by default.')
        nsub_ref = subfaces_reference_matrix.shape[0]
        reference_masses = np.ones(nsub_ref)/nsub_ref

    if isinstance(reference_masses, list):
        reference_masses = np.array(reference_masses)

    total_reference_mass = np.sum(reference_masses)
    ref_weights = reference_masses/total_reference_mass \
        if total_reference_mass > 0 else reference_masses

    for i in range(n_subfaces):
        unit_deviance_vect = unit_deviance(subfaces_matrix[i],
                                           subfaces_reference_matrix)
        mixture_likelihood = np.sum(ref_weights *
                                    np.exp(- rate * unit_deviance_vect))
        
        if mixture_likelihood > 0:
            neg_log_likelihood[i] = - np.log(mixture_likelihood)
        else:
            # Handle the case where mixture_likelihood is zero
            neg_log_likelihood[i] = np.inf
            
    # return negative log-likelihood, normalized by sample size or
    # sample's weights
            
    return np.sum(neg_log_likelihood * weights)


##################################
# For unsupervised evaluation: unit deviance, total deviance in a
# dispersion model###
#################################

def total_deviance(subfaces_list, masses, std_data, threshold,
                   include_singletons_test, epsilon, rate):
    """Calculates the total deviance for the estimated parameter
    (list of subfaces, list of masses), evaluated on  the multivariate
    peaks-over-threshold of a preliminary standardized dataset
    std_data.  The total deviance is normalized by the number of
    extreme samples in std_data

    Parameters:
    - subfaces (list): List of subfaces.

    - masses (array): associated masses

    - std_data (np.ndarray): Standardized data.

    - threshold (float): Radius for selection of extremes in std_data

    - include_singletons_test: if False, disregards events where a
      single feature is large (suitable for analysising concomittant
      events)

    Returns:
    - float: Average distance.

    """
    if len(subfaces_list) == 0:
        return float('inf')

    subfaces_matrix = subfaces_list_to_matrix(subfaces_list,
                                              dimension=std_data.shape[1])
    binary_data = binary_large_features(std_data, threshold,
                                        epsilon=epsilon)

    if isinstance(masses, list):
        masses = np.array(masses)

    if not include_singletons_test:
        id_keep_data = np.where(np.sum(binary_data, axis=1) >= 2)[0]
        binary_data = binary_data[id_keep_data]

    if masses is None:
        warnings.warn('No masses given for subfaces_list.\
        Estimating on test dataset and proceeding ...')
        masses = estim_subfaces_mass(subfaces_list, std_data, threshold,
                                     epsilon, standardize=False)

    Deviance = total_deviance_binary_matrices(
        subfaces_matrix=binary_data, masses=None,
        subfaces_reference_matrix=subfaces_matrix,
        reference_masses=masses, rate=rate)

    return Deviance


#################################
# ##### previously in damex
#################################

def list_to_dict(faces_list, mass_list):
    """
    Converts a list of faces into a dictionary where keys are face
    sizes and values are lists of faces, orderted by sizes.

    Useful for inspection of large lists of subfaces.

    Parameters:
    - faces_list (list of lists): A list where each element is a list
      of points representing a face.
    - mass_list (list): Optional. If provided, the function
      returns a dictionary with the same keys as faces_dict and the values are
      the masses associated with the corresponding subfaces in faces_dict.

    Returns:
    - faces_dict: A dictionary where the key is the size of the face and the
      value is a list of faces of that size.
    - mass_dict: A dictionary with the same keys as faces_dict and the values
      are the masses associated with the corresponding subfaces in faces_dict.
    """
    if len(faces_list) == 0:
        return {}, {}
    # Initialize dictionary with sizes ranging from 1 to the maximum face size
    faces_dict = {size: [] for size in range(1, max(map(len, faces_list)) + 1)}
    mass_dict = {size: [] for size in range(1, max(map(len, faces_list)) + 1)}
    # Populate the dictionary with faces based on their sizes
    for index, face in enumerate(faces_list):
        faces_dict[len(face)].append(face)
        if mass_list is not None:
            mass_dict[len(face)].append(round_signif(mass_list[index], 2))

    return faces_dict, mass_dict

def ftclust_cross_validate(X, standardize, algo, tolerance,
                           min_counts, use_max_subfaces,
                           thresh_train,
                           thresh_test, include_singletons_train,
                           include_singletons_test, rate, cv,
                           random_state):
    """
    Performs cross-validation for feature clustering.

    Parameters:
    - X (np.ndarray): Input data.
    - standardize (bool): Whether to standardize the data.
    - algo (str): Algorithm to use ('damex' or 'clef').
    - tolerance (float): Tolerance level for clustering.
    - min_counts (int): Minimum number of points required to form a cluster.
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

    Returns:
    - np.ndarray: Cross-validated deviance scores.
    """
    if standardize:
        Xt = rank_transform(X)
    else:
        Xt = X

    scores = []
    Norm_Xt = np.max(Xt, axis=1)

    if thresh_train is None:
        thresh_train = np.quantile(Norm_Xt, (1 - 1 / np.sqrt(len(Norm_Xt))))

    if thresh_test is None:
        thresh_test = thresh_train

    # which data are considered extreme for training and testing:
    # logical mask vectors of size len(y)
    id_extreme_train = (Norm_Xt >= thresh_train)
    id_extreme_predict = (Norm_Xt >= thresh_test)

    # K-fold train/test indices
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    # CV loop
    for train_index, test_index in kf.split(X):
        size_train_ex = np.sum(id_extreme_train[train_index])
        size_predict_ex = np.sum(id_extreme_predict[test_index])
        if size_train_ex <= 2 or size_predict_ex <= 0:
            continue
        # Split the data into training and testing sets
        X_train, X_test = Xt[train_index], Xt[test_index]
        # total_mass_train = thresh_train * size_train_ex/len(train_index)

        #  fit the model
        if algo == 'clef':
            Subfaces = clef_fit(X_train, thresh_train, kappa_min=tolerance,
                                standardize=False,
                                include_singletons=include_singletons_train)
            Masses = clef_estim_subfaces_mass(Subfaces, X_train,
                                              thresh_train,
                                              standardize=False)
            epsilon_val = None
            # include_singletons_test = False
            # ## include_singletons_test is hard set to False
            # # here because  setting it to True has undesirable behaviour,
            # # namely it always selects a large kappa.

        elif algo == 'damex':
            # min_counts = kwargs.pop('min_counts', 0)
            faces_i, mass_i = damex_fit(
                X_train, thresh_train, epsilon=tolerance,
                min_counts=min_counts, standardize=False,
                include_singletons=include_singletons_train
            )
            Subfaces = faces_i
            Masses = mass_i

            if use_max_subfaces:
                faces_dict, _ = list_to_dict(faces_i, mass_i)
                Subfaces = find_maximal_faces(faces_dict, lst=True)
                Masses = estim_subfaces_mass(Subfaces, X_train,
                                             thresh_train,
                                             epsilon=tolerance,
                                             standardize=False)

            epsilon_val = tolerance
        else:
            raise ValueError("algo must be either`damex` or `clef`")

        clust_error = total_deviance(
            Subfaces, Masses,  X_test, thresh_test,
            include_singletons_test=include_singletons_test,
            epsilon=epsilon_val,
            rate=rate)

        scores.append(clust_error)

    return np.array(scores)

# thresh_train=None,
# thresh_test=None, include_singletons_train=False,
# include_singletons_test=False,  **kwargs):


def ftclust_choose_tolerance_cv(tolerance_grid, X, standardize, algo,
                                unstable_tol_max, min_counts,
                                use_max_subfaces,
                                thresh_train, thresh_test,
                                include_singletons_train,
                                include_singletons_test, rate, cv,
                                random_state, plot=False):
    """
    Chooses the optimal tolerance value based on cross-validation.

    Parameters:
    - tolerance_grid (list): List of tolerance values to test.
    - X (np.ndarray): Input data.
    - standardize (bool): Whether to standardize the data.
    - algo (str): Algorithm to use ('damex' or 'clef').
    - unstable_tol_max (float): Maximum tolerance value for unstable solutions.
    - min_counts (int): Minimum number of points required to form a cluster.
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

    Returns:
    - float: Selected tolerance value.
    - float: Deviance value for the selected tolerance.
    - np.ndarray: Vector of CV deviance values.
    """
    ntol = len(tolerance_grid)
    cv_error_vect = np.zeros(ntol)
    for i in range(ntol):
        cv_scores = ftclust_cross_validate(
            X, standardize, algo, tolerance=tolerance_grid[i],
            min_counts=min_counts,
            use_max_subfaces=use_max_subfaces,
            thresh_train=thresh_train,
            thresh_test=thresh_test,
            include_singletons_train=include_singletons_train,
            include_singletons_test=include_singletons_test,
            rate=rate,
            cv=cv,
            random_state=random_state
            )

        cv_error_vect[i] = np.mean(cv_scores)

    i_maxerr = np.argmax(cv_error_vect[tolerance_grid < unstable_tol_max])
    tol_maxerr = tolerance_grid[i_maxerr]
    i_mask = tolerance_grid <= tol_maxerr
    i_cv = np.argmin(cv_error_vect +
                     (1e+23) * i_mask)
    tol_cv = tolerance_grid[i_cv]
    error_tol_cv = cv_error_vect[i_cv]
    if plot:
        plt.scatter(tolerance_grid, cv_error_vect, c='gray', label='CV error')
        plt.plot([tol_cv, tol_cv], [0, error_tol_cv], c='red',
                 label='selected value')
        plt.grid()
        plt.legend()
        if algo == 'damex':
            plt.title("DAMEX: clustering pseudo-deviance, K-fold CV scores")
        else:
            plt.title("CLEF: clustering pseudo-deviance, K-fold CV scores")
        plt.show()
    return tol_cv, error_tol_cv, cv_error_vect

