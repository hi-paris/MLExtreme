import itertools as it
import numpy as np
import matplotlib.pyplot as plt
# import itertools as it
import networkx as nx
from . import utilities as ut
from ...utils.EVT_basics import rank_transform, round_signif
# from .damex import entropy, AIC_clustering
import pdb
# from . import ftclust_analysis as fca



#############
# Clef algo #
#############


def clef(X, radius, kappa_min, standardize=True, include_singletons=False):
    """Returns maximal faces s.t. kappa > kappa_min."""
    if standardize:
        x_norm = rank_transform(X)
    else:
        x_norm = X
    x_bin = ut.binary_large_features(x_norm, radius)
    result = clef_0(x_bin, kappa_min, include_singletons)
    return result


def clef_0(x_bin, kappa_min, include_singletons):
    """Return maximal faces s.t. kappa > kappa_min."""
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
    # Create a set of all possible indices
    all_indices = set(range(d))
    # Iterate through each sublist in faces and remove its elements from all_indices
    for sublist in faces:
        all_indices.difference_update(sublist)

    # Convert the remaining indices to a list and return
    list_indices =  list(all_indices)
    return list([index] for index in list_indices)



def faces_init(x_bin, mu_0):
    """Returns faces of size 2 s.t. kappa > kappa_min."""
    asymptotic_pair = []
    for (i, j) in it.combinations(range(x_bin.shape[1]), 2):
        pair_tmp = x_bin[:, [i, j]]
        one_out_of_two = np.sum(np.sum(pair_tmp, axis=1) > 0)
        two_on_two = np.sum(np.prod(pair_tmp, axis=1))
        if one_out_of_two > 0:
            proba = two_on_two / one_out_of_two
            if proba > mu_0:
                asymptotic_pair.append([i, j])

    return asymptotic_pair


def compute_beta(x_bin, face):
    return np.sum(np.sum(x_bin[:, face], axis=1) > len(face)-2)


def kappa(x_bin, face):
    """Returns kappa value.

    kappa = #{i | for all j in face, X_ij=1} /  #{i | at least |face|-1 j, X_ij=1}

    """
    beta = compute_beta(x_bin, face)
    all_face = np.sum(np.prod(x_bin[:, face], axis=1))
    if beta == 0.:
        kap = 0.
    else:
        kap = all_face / float(beta)

    return kap



def find_faces(x_bin, kappa_min):
    """Returns all faces s.t. kappa > kappa_min."""
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
    """Return inclusion-wise maximal faces."""
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


# #################
# Generate Candidate Subfaces of Increased Size at Each CLEF Iteration
# #################

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
    vector_subfaces = ut.subfaces_list_to_matrix(subfaces, dimension)
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
    
#     [1] Chiapino, M., Sabourin, A., & Segers, J. (2019). Identifying groups of
#     variables with the potential of being large simultaneously.
#     Extremes, 22, 193-222.

#     """
#     return np.sum(np.sum(
#         binary_matrix[:, subface], axis=1) == len(subface)) / float(k)


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

def clef_estim_subfaces_mass(subfaces_list, X, threshold,
                             standardize=True):
    res = ut.estim_subfaces_mass(subfaces_list, X, threshold, epsilon=None,
                                 standardize=standardize)
    return res

def clef_select_kappa_AIC(vect_kappa, X, radial_threshold,
#                          include_singletons=True,
                          plot=False, standardize=True,
                          unstable_kappa_max=0.1):
    if standardize:
        radii = np.max(rank_transform(X), axis=1)
    else:
        radii = np.max(X, axis=1)
    num_extremes = np.sum(radii >= radial_threshold)
    total_mass =  radial_threshold * num_extremes / X.shape[0]
    ntests = len(vect_kappa)
    vect_aic = np.zeros(ntests)
    counter = 0
    for kappa in vect_kappa:
        clef_faces = clef(X, radial_threshold, kappa, standardize=standardize,
                          include_singletons=True)
        list_of_masses = clef_estim_subfaces_mass(clef_faces, X,
                                                  radial_threshold,
                                                  standardize=standardize)
        vect_aic[counter] = ut.AIC_clustering(list_of_masses, num_extremes,
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
