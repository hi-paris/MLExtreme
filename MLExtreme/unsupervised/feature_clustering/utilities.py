from __future__ import division
import random as rd
import itertools as it
import numpy as np
import networkx as nx


def remove_zero_row(binary_array):
    """
    Removes rows that contain only zeros from a binary matrix.

    Parameters:
    - binary_array (np.ndarray): A binary matrix from which to remove zero rows.

    Returns:
    - np.ndarray: The binary matrix with zero rows removed.
    """
    # Remove rows where the sum of elements is zero (i.e., all
    # elements are zero)
    return binary_array[np.sum(binary_array, axis=1) > 0]


# def remove_zero_row(v_bin):
#     """Remove rows that contain only zeros in matrix v_bin."""
#     return v_bin[np.sum(v_bin, axis=1) > 0]


import numpy as np

def binary_large_features(std_data, radial_threshold, epsilon=None):
    """Filters extreme samples from the input `std_data`, returns a
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
        # Create a binary matrix where values are above the scaled
        # radius threshold
        extreme_samples = std_data[np.max(std_data, axis=1) >=
                                   radial_threshold]
        binary_matrix = extreme_samples > radial_threshold * epsilon
    else:
        # Create a binary matrix which values are  above the
        # radius threshold
        binary_matrix = std_data >= radial_threshold

    # Convert the binary matrix to float type and remove zero rows
    return remove_zero_row(binary_matrix.astype(float))

# def  above_radius_bin(v_rank, radius, eps=None):
    # """Returns binary matrix with values above radius.
    # case eps=None is treated as if eps=1"""
    # if eps:
    #     v_bin = v_rank[np.max(v_rank, axis=1) > radius] > radius*eps
    # else: 
    #     v_bin = v_rank > radius

    # return remove_zero_row(v_bin.astype(float))


def kth_largest_bin(v_rank, k):
    """Returns binary matrix with kth largest points on each column."""
    n_sample, n_dim = v_rank.shape
    mat_rank = np.argsort(v_rank, axis=0)[::-1]
    v_bin = np.zeros((n_sample, n_dim))
    for j in range(n_dim):
        v_bin[mat_rank[:k, j], j] = 1.

    return v_bin


# def rank_transformation(x_raw): ## moved in utils/EVT_basics 'rank_transform'
#     """Standardize each column to Pareto : v_rank_ij = n_sample/(rank(x_raw_ij) + 1)."""
#     n_sample, n_dim = x_raw.shape
#     mat_rank = np.argsort(x_raw, axis=0)[::-1]
#     x_rank = np.zeros((n_sample, n_dim))
#     for i in range(n_dim):
#         x_rank[mat_rank[:, i], i] = np.arange(n_sample) + 1
#     v_rank = n_sample/x_rank

#     return v_rank


def check_errors(charged_faces, result_faces, dim):
    """Faces found -> (recovered, misseds, falses)"""
    faces_mat_true = list_faces_to_vect(charged_faces, dim)
    faces_mat = list_faces_to_vect(result_faces, dim)
    # Find supsets of real face
    cond_1 = np.dot(faces_mat, faces_mat_true.T) == np.sum(faces_mat_true, axis=1)
    # Find subsets of a real face
    cond_2 = np.dot(faces_mat_true, faces_mat.T) == np.sum(faces_mat, axis=1)
    # Intersect sub and supsets to get recovered faces
    ind_exct_supsets = list(set(np.nonzero(np.sum(cond_1, axis=1))[0]) -
                            set(np.nonzero(np.sum(cond_1 * cond_2.T,
                                                  axis=1))[0]))
    ind_exct_subsets = list(set(np.nonzero(np.sum(cond_2.T, axis=1))[0]) -
                            set(np.nonzero(np.sum(cond_1 * cond_2.T,
                                                  axis=1))[0]))
    ind_pure_false = list(set(range(len(result_faces))) -
                          (set(np.nonzero(np.sum(cond_1 * cond_2.T,
                                                 axis=1))[0]) |
                           set(ind_exct_supsets) | set(ind_exct_subsets)))
    # Results
    founds = [result_faces[i]
              for i in np.nonzero(np.sum(cond_1 * cond_2.T, axis=1))[0]]
    falses_pure = [result_faces[i] for i in ind_pure_false]
    exct_subsets = [result_faces[i] for i in ind_exct_subsets]
    exct_supsets = [result_faces[i] for i in ind_exct_supsets]
    misseds = [charged_faces[i]
               for i in np.nonzero(np.sum(cond_1 * cond_2.T, axis=0) == 0)[0]]

    return founds, misseds, falses_pure, exct_subsets, exct_supsets


######################
# Levenshtein metric #
######################


def levenshtein_dist(face_1, face_2):
    """Pseudo-Levenshtein distance between array_1 and array_2: diff_vals/(sim_vals + diff_vals)"""
    return np.sum(abs(face_1 - face_2))/(np.sum(face_1 + face_2 > 0))


def levenshtein_dist_mat(face_1, faces):
    """Pseudo-Levenshtein distance between array_1 and mat. diff_vals/(sim_vals + diff_vals)"""
    return (np.sum(abs(face_1 - faces), axis=1) /
            np.sum(face_1 + faces > 0, axis=1))


def levenshtein_similarity(v_bin):
    """Pseudo-Levenshtein similarity between each row of v_bin."""
    return levenshtein_kernel_dist(v_bin) - 1


def levenshtein_kernel_dist(v_bin):
    """Pseudo-Levenshtein distance between each row of v_bin."""
    n_row = v_bin.shape[0]
    dist = np.zeros((n_row, n_row))
    for j, v_row in enumerate(v_bin):
        dist[j] = levenshtein_dist_mat(v_row, v_bin)

    return dist


def levenshtein_faces_radius(faces, radius, v_rank):
    """Average pseudo-Levenshtein distance between list of faces and v_rank rows."""
    faces = list_faces_to_vect(faces, dim=v_rank.shape[1])
    v_bin = binary_large_features(v_rank, radius)

    return np.mean([np.min(levenshtein_dist_mat(v_row, faces)) for v_row in v_bin])


def levenshtein_faces_radiuss(faces, radiuss, v_rank, eps=1.):
    """Average pseudo-Levenshtein distance for different radius."""
    faces = list_faces_to_vect(faces, v_rank.shape[1])
    mean_dist = []
    nb_extr = []
    for radius in radiuss:
        dist = []
        v_bin = binary_large_features(v_rank, radius, eps)
        for v_row in v_bin:
            dist.append(np.min(levenshtein_dist_mat(v_row, faces)))
        mean_dist.append(np.sum(dist)/len(v_bin))
        nb_extr.append(v_bin.shape[0])

    return mean_dist, nb_extr


##################
# Generate faces #
##################


def gen_random_faces(dim, nb_faces,
                     max_size=8,
                     p_geom=0.25,
                     with_singlet=False):
    """Returns list of random subsets of {1,...,dim}"""
    faces = np.zeros((nb_faces, dim))
    size_face = min(np.random.geometric(p_geom) + 1, max_size)
    faces[0, rd.sample(range(dim), size_face)] = 1
    k = 1
    loop = 0
    while k < nb_faces and loop < 1e4:
        size_face = min(np.random.geometric(p_geom) + 1, max_size)
        face = np.zeros(dim)
        face_ind = rd.sample(range(dim), size_face)
        face[face_ind] = 1
        if np.sum(np.prod(faces[:k]*face == face, axis=1)) == 0:
            if np.sum(np.prod(faces[:k]*face == faces[:k], axis=1)) == 0:
                faces[k, face_ind] = 1
                k += 1
        loop += 1
    faces_list = [list(np.nonzero(f)[0]) for f in faces]
    feats = list({j for face in faces_list for j in face})
    missing_feats = list(set(range(dim)) - {j for face in faces_list for j in face})
    singlets = []
    if missing_feats:
        if with_singlet:
            singlets = [[j] for j in missing_feats]
        else:
            if len(missing_feats) > 1:
                faces_list.append(missing_feats)
            if len(missing_feats) == 1:
                missing_feats.append(list(set(range(dim)) -
                                          set(missing_feats))[0])
                faces_list.append(missing_feats)
    if with_singlet:
        return faces_list, feats, singlets
    return faces_list


def faces_complement(faces, dim):
    return [list(set(range(dim)) - set(face)) for face in faces]


def faces_matrix(faces):
    n_faces = len(faces)
    feats = list({j for face in faces for j in face})
    d_max = int(max(feats))
    mat_faces = np.zeros((n_faces, d_max+1))
    for k, face in enumerate(faces):
        mat_faces[k, face] = 1

    return mat_faces[:, np.sum(mat_faces, axis=0) > 0]


def faces_conversion(faces):
    feats = list({j for face in faces for j in face})
    feats_dict = {feat: j for j, feat in enumerate(feats)}

    return [[feats_dict[j] for j in face] for face in faces]


def faces_reconvert(faces, feats):
    return [[feats[j] for j in face] for face in faces]


def suppress_sub_faces(faces):
    new_list = []
    for face in faces:
        subset = False
        for face_t in faces:
            if len(face_t) > len(face):
                if len(set(face_t) - set(face)) == len(face_t) - len(face):
                    subset = True
        if not subset:
            new_list.append(face)

    return new_list


def check_if_in_list(list_faces, face):
    val = False
    for face_test in list_faces:
        if set(face_test) == set(face):
            val = True

    return val


def all_subsets_size(list_faces, size):
    subsets_list = []
    for face in list_faces:
        if len(face) == size:
            if not check_if_in_list(subsets_list, face):
                subsets_list.append(face)
        if len(face) > size:
            for sub_face in it.combinations(face, size):
                if not check_if_in_list(subsets_list, face):
                    subsets_list.append(list(sub_face))

    return subsets_list


def indexes_true_faces(all_faces_2, faces_2):
    ind = []
    for face in faces_2:
        cpt = 0
        for face_test in all_faces_2:
            if set(face) == set(face_test):
                ind.append(int(cpt))
            cpt += 1

    return np.array(ind)


def all_sub_faces(faces):
    all_faces = []
    for face in faces:
        k_face = len(face)
        if k_face == 2:
            all_faces.append(face)
        else:
            for k in range(2, k_face):
                for beta in it.combinations(face, k):
                    all_faces.append(beta)
            all_faces.append(face)
    sizes = list(map(len, all_faces))
    all_faces = np.array(all_faces)[np.argsort(sizes)]

    return list(map(list, set(map(tuple, all_faces))))


def dict_size(all_faces):
    sizes = np.array(list(map(len, all_faces)))
    size_set = list(set(sizes))
    dict_faces = {k: np.array(all_faces, dtype=object)[np.nonzero(sizes == k)]
                  for k in size_set}

    return dict_faces


def faces_to_test(dict_all_faces, dim):
    all_faces = {2: [face for face in it.combinations(range(dim), 2)]}
    for size in dict_all_faces.keys()[1:]:
        all_faces[size] = candidate_faces(dict_all_faces[size-1], size-1, dim)

    return all_faces


def dict_falses(dict_true_faces, dim):
    dict_faces_test = faces_to_test(dict_true_faces, dim)
    dict_false_faces = {}
    for size in dict_true_faces.keys():
        ind_s = indexes_true_faces(dict_faces_test[size], dict_true_faces[size])
        ind_s_c = list(set(range(len(dict_faces_test[size]))) - set(ind_s))
        if np.array(dict_faces_test[size])[ind_s_c]:
            dict_false_faces[size] = np.array(dict_faces_test[size])[ind_s_c]

    return dict_false_faces


def make_graph(faces, size, dim):
    """Returns graph.

    -nodes represent faces.
    -edges exist if faces have at most one different feature.

    """
    vect_faces = list_faces_to_vect(faces, dim)
    nb_faces = len(vect_faces)
    graph_s = nx.Graph()
    nodes = range(nb_faces)
    graph_s.add_nodes_from(nodes)
    edges = np.nonzero(np.triu(np.dot(vect_faces, vect_faces.T) == size - 1))
    graph_s.add_edges_from([(edges[0][i], edges[1][i])
                            for i in range(len(edges[0]))])

    return graph_s


def candidate_faces(faces, size, dim):
    """Generates candidate faces of size s+1 from list of faces of size s."""
    graph_s = make_graph(faces, size, dim)
    faces_to_try = []
    cliques = list(nx.find_cliques(graph_s))
    ind_to_try = np.nonzero(np.array(list(map(len, cliques))) == size + 1)[0]
    for j in ind_to_try:
        clique_feature = set([])
        for i in range(len(cliques[j])):
            clique_feature = clique_feature | set(faces[cliques[j][i]])
        clique_feature = list(clique_feature)
        if len(clique_feature) == size + 1:
            faces_to_try.append(clique_feature)

    return faces_to_try


def list_faces_to_vect(faces, dim):
    """List of faces indices -> bin matrix."""
    nb_faces = len(faces)
    vect_faces = np.zeros((nb_faces, dim))
    for i, face in enumerate(faces):
        vect_faces[i, face] = 1.

    return vect_faces


############################
# faces frequency analysis #
############################


def rho_value(x_bin, face, k):
    """Returns rho value of face."""
    return np.sum(np.sum(x_bin[:, face], axis=1) == len(face))/float(k)


def rhos_face_pairs(x_bin, face, k):
    """Computes rho(i,j) for each (i,j) in face."""
    rhos_face = {}
    for (i, j) in it.combinations(face, 2):
        rhos_face[i, j] = rho_value(x_bin, [i, j], k)

    return rhos_face


def partial_matrix(x_bin_base, x_bin_partial, j):
    """Returns x_bin_base with its jth colomn replaced by the jth column of x_bin_partial."""
    x_bin_copy = np.copy(x_bin_base)
    x_bin_copy[:, j] = x_bin_partial[:, j]

    return x_bin_copy


def r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, face, k):
    """Returns dictionary : {j: derivative of r in j}"""
    r_p = {}
    for j in face:
        x_r = partial_matrix(x_bin_k, x_bin_kp, j)
        x_l = partial_matrix(x_bin_k, x_bin_km, j)
        r_p[j] = 0.5*k**0.25*(rho_value(x_r, face, k) - rho_value(x_l, face, k))

    return r_p
