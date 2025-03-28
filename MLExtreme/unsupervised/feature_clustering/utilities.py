from __future__ import division
import random as rd
import itertools as it
import numpy as np
import networkx as nx


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
    return remove_zero_rows(binary_matrix.astype(float))


# #################
# Generate Candidate Subfaces of Increased Size at Each CLEF Iteration
# #################

def list_subfaces_to_bin_matrix(subfaces, dimension):
    """
    Converts a list of subface indices into a binary matrix.

    Args:
    - subfaces (list): List of subfaces.
    - dimension (int): Dimensionality of the ambient space.

    Returns:
    - np.ndarray: Binary matrix representation of subfaces.
    """
    num_subfaces = len(subfaces)
    vector_subfaces = np.zeros((num_subfaces, dimension))

    for subface_index, subface in enumerate(subfaces):
        vector_subfaces[subface_index, subface] = 1.0

    return vector_subfaces


def make_graph(subfaces, size, dimension):
    """
    Creates a graph where nodes represent subfaces and edges exist if subfaces
    differ by at most one feature.

    It is the main building block of the `candidate_sufaces' function
    used in CLEF and variants [1,2].

    Args:
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
    vector_subfaces = list_subfaces_to_bin_matrix(subfaces, dimension)
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
    """Generates a list A_{s+1} of candidate subfaces of size s+1
    from a list A_s = `subfaces' of subfaces of size s, with `s=size`.
    Candidate subfaces are all subfaces of size s+1 which are
    supersets of each of subface of size s in the current list
    `subfaces'.

    This is a key step in CLEF algorithm [1] and variants [2]

    Args:
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


# ###########################
# Subfaces frequency analysis #
# ###########################


def rho_value(binary_matrix, subface, k):
    """
    Calculates the rho value of a subface.
    (notation r_a(1) in [1], where a is the subface)

    Args:
    - binary_matrix (np.ndarray): Binary matrix.
    - subface (list): Subface to evaluate.
    - k (int): Number of samples.

    Returns:
    - float: Rho value.

    References
    -----------
    
    [1] Chiapino, M., Sabourin, A., & Segers, J. (2019). Identifying groups of
    variables with the potential of being large simultaneously.
    Extremes, 22, 193-222.

    """
    return np.sum(np.sum(
        binary_matrix[:, subface], axis=1) == len(subface)) / float(k)


def partial_matrix(base_matrix, partial_matrix, column_index):
    """
    Replaces the column_index column of base_matrix with the corresponding
    column from partial_matrix.

    Used in asymptotic variants of CLEF [1] to compute partial derivatives
    of the 'kappa' and the 'r' functions

    Args:
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

