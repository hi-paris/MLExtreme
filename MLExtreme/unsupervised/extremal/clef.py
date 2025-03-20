import itertools as it
import numpy as np
import utilities as ut


#############
# Clef algo #
#############


def clef(x_norm, radius, kappa_min):
    """Return maximal faces s.t. kappa > kappa_min."""
    x_bin = ut.above_radius_bin(x_norm, radius)
    faces_dict = find_faces(x_bin, kappa_min)
    faces = find_maximal_faces(faces_dict)

    return faces


def clef_0(x_bin, kappa_min):
    """Return maximal faces s.t. kappa > kappa_min."""
    faces_dict = find_faces(x_bin, kappa_min)
    faces = find_maximal_faces(faces_dict)

    return faces


##################
# CLEF functions #
##################


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


def compute_beta(x_bin, face):
    return np.sum(np.sum(x_bin[:, face], axis=1) > len(face)-2)


def khi(binary_data, face):
    face_vect_tmp = binary_data[:, face]
    face_exist = float(np.sum(np.sum(face_vect_tmp, axis=1) > 0))
    all_face = np.sum(np.prod(face_vect_tmp, axis=1))

    return all_face/face_exist


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
        faces_to_try = ut.candidate_faces(faces_dict[size], size, dim)
        if faces_to_try:
            for face in faces_to_try:
                if kappa(x_bin, face) > kappa_min:
                    faces_dict[size + 1].append(face)
        size += 1

    return faces_dict


def find_maximal_faces(faces_dict, lst=True):
    """Return inclusion-wise maximal faces."""
    k = len(faces_dict.keys()) + 1
    maximal_faces = [faces_dict[k]]
    faces_used = list(map(set, faces_dict[k]))
    for i in range(1, k - 1):
        face_tmp = list(map(set, faces_dict[k - i]))
        for face in faces_dict[k - i]:
            for face_test in faces_used:
                if len(set(face) & face_test) == k - i:
                    face_tmp.remove(set(face))
                    break
        maximal_faces.append(list(map(list, face_tmp)))
        faces_used = faces_used + face_tmp
    maximal_faces = maximal_faces[::-1]
    if lst:
        maximal_faces = [face for faces_ in maximal_faces
                         for face in faces_]

    return maximal_faces


############################
# faces frequency analysis #
############################


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
        faces_to_try = ut.candidate_faces(faces_dict[size], size, dim)
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

    return find_maximal_faces(faces_dict)
