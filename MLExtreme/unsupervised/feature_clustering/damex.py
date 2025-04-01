import numpy as np
import matplotlib.pyplot as plt
from . import utilities as ut  # #import binary_large_features
from ...utils.EVT_basics import rank_transform, round_signif
# TEMPORARY
import pdb



def damex_0(binary_matrix, include_singletons=True):
    """Analyzes a binary matrix to determine the number of points per subface.

    Parameters:
    - binary_matrix (np.ndarray):
        A binary matrix where rows represent samples
        and columns represent features.

    Returns:
        - faces (list of lists):
            A list of subfaces of the infinite sphere,
            each represented as a list of feature indices.
    
        -  mass (np.ndarray):
            An array indicating the number of rescaled
            samples observed within each subface.

    """
    # set the minimum number of features per subface: 
    if include_singletons:
        min_features = 1
    else:
        min_features = 2
        
    # sample size 
    num_samples = binary_matrix.shape[0]
    # number of features (columns) with value 1 for each
    # sample (row)
    num_features_per_sample = np.sum(binary_matrix, axis=1)

    # dot product of the binary matrix with its
    # transpose: get shared features between samples.
    # entry ij is the number of features shared by samples i and j. 
    shared_features_matrix = np.dot(binary_matrix, binary_matrix.T)

    # Determine samples with exactly matching features
    # entry ij is one if (binary) samples i and j are identical
    exact_match_matrix = (
        shared_features_matrix == num_features_per_sample) & (
            # shared_features_matrix.T == num_features_per_sample).T
            shared_features_matrix == num_features_per_sample).T   # MODIF!!

    # Set of samples not yet observed / assigned in any subface 
    uncovered_samples = set(range(num_samples))

    # Dictionary to store the number of samples assigned to each  subface
    subface_sample_count = {}

    # Iterate over each sample to identify subfaces
    for i in range(num_samples):
        # Find samples with exactly matching features
        matching_samples = list(np.nonzero(exact_match_matrix[i, :])[0])

        # If the current sample has not been assigned yet (this would
        # happen if it belongs to the same subface as another sample j<i
        if i in uncovered_samples:
            # Mark these samples as assigned 
            uncovered_samples -= set(matching_samples)

            # If the sample has more than one feature, record the
            # subface (add a key to the dictionary with value equal to
            # the number of samples in this subface.  The key is the
            # index of the first record of the event that a sample
            # point belongs to the considered subface.
            if num_features_per_sample[i] >= min_features:   # MODIFIED. previously >1
                subface_sample_count[i] = len(matching_samples)

    # Sort subfaces by the number of samples they cover, in descending order
    sorted_indices = np.argsort(list(subface_sample_count.values()))[::-1]

    # Create a (sorted) list of subfaces, each represented by the
    # indices of `1` features
    subfaces = [list(np.nonzero(binary_matrix[list(subface_sample_count)[i], :])[0])
                for i in sorted_indices]

    converted_subfaces = [[int(item) for item in sublist]
                          for sublist in subfaces]

    # Create an array of the number of samples covered by each subface
    counts = [list(subface_sample_count.values())[i] for i in sorted_indices]

    return converted_subfaces, np.array(counts)


def damex(data, radial_threshold, epsilon, min_counts=0, standardize=True,
          include_singletons=True):
    """
    Implements the methodology proposed in [1].

    Given a data matrix (n,d), damex provides the groups of features that are
    likely to be simultaneously large, while the other features are small. The
    function also returns an estimate of the limit measure associated with each
    subgroup.

    Formally: an estimation of the vector
    $$ M_a = lim_{t to infty} t * P(V in t *  C_a) $$
    where V is a random vector from which (data) represents an iid sample and
     C_a is the truncated subcone associated with the subface a of the unit sphere.
    If `standardize=True`, standardization of `data` columns is performed internally
    and the vector V above represents the standardized version of an entry of
    `data`.

    Parameters:

    - data (np.ndarray): A data matrix.

    - radial_threshold (float): The radial threshold for selecting extreme
      samples.

    - epsilon (float): A scaling factor for the radial threshold.

    - min_counts (int): The minimum number of points required per face.
        Defaults to 0.

    - standardize (bool): Defaults to True. If True, the data matrix will be
      standardized using the rank transformation mlx.rank_transform. Meaningful
      usage requires that columns (features) follow a unit Pareto distribution or
      that rank-standardization has been applied previously with
      `standardize=False`, or that columns are possibly not standard and
      `standardize=True`.

    Returns:

    - list: A list of faces where each face has more than min_points points.

    Note:

    For standardizing a raw data input, see mlx.rank_transform,
    mlx.rank_transform_test.

    References:
    [1] Goix, N., Sabourin, A., & Clémençon, S. (2017). Sparse representation of
    multivariate extremes with applications to anomaly detection. Journal of
    Multivariate Analysis, 161, 12-31. (The returned estimator is defined in
    Equation (3.3) from the reference)
    """
    # Standardize data if necessary
    if standardize:
        intern_data = rank_transform(data)
    else:
        intern_data = data
        
    # Generate binary matrix and determine faces and their masses
    binary_matrix = ut.binary_large_features(intern_data, radial_threshold,
                                             epsilon)
    faces, counts = damex_0(binary_matrix, include_singletons)
    n = intern_data.shape[0]

    limit_mass_estimator = radial_threshold * counts / n
    id_large_mass = (counts >= min_counts)
    number_heavy_faces = np.sum(id_large_mass)
    truncated_faces = faces[:number_heavy_faces]
    truncated_mass_estimator = limit_mass_estimator[:number_heavy_faces]
    
    # Return faces with mass greater than or equal to min_points
    return truncated_faces, truncated_mass_estimator

def damex_estim_subfaces_mass(subfaces_list, X, threshold, epsilon, 
                              standardize=True):
    res = ut.estim_subfaces_mass(subfaces_list, X, threshold, epsilon,
                                 standardize=standardize)
    return res


def damex_select_epsilon_AIC(vect_eps, X, radial_threshold,
                             plot=False, standardize=True,
                             unstable_eps_max=0.1):
    """
    The goal is to avoid trivial clusterings, such as a
    uniform scattering of points on many subfaces.  Here, a
    cluster is identified with a subface.

    **Key Idea**: A well-chosen
    epsilon should yield an informative clustering of points, characterized
    by low entropy. This is directly related to the AIC of a categorical model where each identified subface corresponds to a category. 
    To avoid unstable solutions   for small epsilon with one large cluster (the central face) and a many small clusters (low dimensional subfaces), the algorithm searches for a maximizer of the AIC eps_max_aic (ie a bad epsilon) in the range [0, unsafe_eps_max] and then looks for a minimum [eps_max_aic, vect_eps[-1]], where
    vect_eps is the grid passed in argument for the search. Values of the latter should be sorted in increasing order. 
    """
    if standardize:
        radii = np.max(rank_transform(X), axis=1)
    else:
        radii = np.max(X, axis=1)
    num_extremes = np.sum(radii >= radial_threshold)
    total_mass = radial_threshold * num_extremes / X.shape[0]
    ntests = len(vect_eps)
    vect_aic = np.zeros(ntests)
    # vect_number_faces = np.zeros(ntests)
    counter = 0
    for eps in vect_eps:
        _, list_of_masses = damex(X, radial_threshold, eps,
                                  min_counts=0, 
                                  standardize=standardize,
                                  include_singletons=True)
        # selection based on AIC:
        # vect_aic below is proportional to :
        # - log-likelihood(categorical model) + k where k
        # is the number of categories, ie the number of subfaces.
        # vect_aic = num_extremes * vect_entropy + vect_number_faces 
        vect_aic[counter] = ut.AIC_clustering(list_of_masses, num_extremes,
                                              total_mass)
        # vect_number_faces[counter] = len(list_of_masses)
        counter += 1
        
    i_maxerr = np.argmax(vect_aic[vect_eps < unstable_eps_max])
    eps_maxerr = vect_eps[i_maxerr]
    i_mask = vect_eps <= eps_maxerr
    i_minAIC = np.argmin(vect_aic + (1e+23) * i_mask)
    # #i_candidates_aic[i_minAIC_candidates]
    eps_select_aic = vect_eps[i_minAIC]

    # selection based on minimum number of subfaces
    # i_select_numfaces = np.argmin(vect_number_faces + (1e+23) * i_mask)
    # eps_select_numfaces = vect_eps[i_select_numfaces]

    if plot:
        #fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        plt.figure(figsize=(10, 5))
        plt.xlabel('epsilon')
        plt.ylabel('AIC')
        plt.title('DAMEX- AIC versus epsilon')
        # axs[0].plot([eps_select_entropy, eps_select_entropy],
        #             [0, max(vect_entropy)], c='blue')
        plt.scatter(vect_eps, vect_aic, c='gray', label='AIC')
        plt.plot([eps_select_aic, eps_select_aic],
                    [0, max(vect_aic)], c='red')
        plt.grid(True)
        # axs[1].scatter(vect_eps, vect_number_faces, c='gray')
        # axs[1].plot([eps_select_numfaces, eps_select_numfaces],
        #             [0, max(vect_number_faces)], c='red')

        # axs[1].set_xlabel('epsilon')
        # axs[1].set_ylabel('number of faces')
        # axs[1].set_title('Number of faces versus epsilon')
        # axs[1].grid(True)
       #plt.tight_layout()
        plt.show()

    print(f'DAMEX: selected epsilon with AIC method: \
        {round_signif(eps_select_aic, 2)}')
        
        # print(f'size based selection of epsilon:\
        # {round_signif(eps_select_numfaces, 2)}')
        # eps_select_entropy,
    return eps_select_aic  # np.array([eps_select_aic, eps_select_numfaces])


