import numpy as np
from  . import utilities as ut  # #import binary_large_features
from ...utils.EVT_basics import rank_transform
# TEMPORARY
import pdb


###    TEMPORARY: ORIGINAL VERSION: singleton faces are discarded. 
def damex_0(binary_matrix):
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
    # Number of samples (rows) in the binary matrix
    num_samples = binary_matrix.shape[0]
    # Calculate the number of features (columns) with value 1 for each
    # sample (row)
    num_features_per_sample = np.sum(binary_matrix, axis=1)

    # Calculate the dot product of the binary matrix with its
    # transpose to get shared features between samples
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
            if num_features_per_sample[i] >= 1:   # MODIFIED. previously >=1
                subface_sample_count[i] = len(matching_samples)

    # Sort subfaces by the number of samples they cover, in descending order
    sorted_indices = np.argsort(list(subface_sample_count.values()))[::-1]

    # Create a (sorted) list of subfaces, each represented by the
    # indices of `1` features
    subfaces = [list(np.nonzero(binary_matrix[list(subface_sample_count)[i], :])[0])
                for i in sorted_indices]

    # Create an array of the number of samples covered by each subface
    counts = [list(subface_sample_count.values())[i] for i in sorted_indices]

    return subfaces, np.array(counts)


def damex(data, radial_threshold, epsilon, min_counts=0, standardize=True):
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
    faces, counts = damex_0(binary_matrix)
    n = intern_data.shape[0]

    limit_mass_estimator = radial_threshold * counts / n
    id_large_mass = (counts >= min_counts)
    number_heavy_faces = np.sum(id_large_mass)
    truncated_faces = faces[:number_heavy_faces]
    truncated_mass_estimator = limit_mass_estimator[:number_heavy_faces]
    
    # Return faces with mass greater than or equal to min_points
    return truncated_faces, truncated_mass_estimator


## USEFUL? for CLEF? 
def list_to_dict_size(faces_list):
    """Converts a list of faces into a dictionary where keys are face
    sizes and values are lists of faces.

    Parameters:

    - faces_list (list of lists): A list where each element is a list
      of points representing a face.

    Returns:

    - dict: A dictionary where the key is the size of the face and the
      value is a list of faces of that size.

    """
    # Initialize dictionary with sizes ranging from 2 to the maximum face size
    faces_dict = {size: [] for size in range(2, max(map(len, faces_list)) + 1)}

    # Populate the dictionary with faces based on their sizes
    for face in faces_list:
        faces_dict[len(face)].append(face)

    return faces_dict

