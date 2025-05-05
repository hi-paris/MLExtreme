import itertools as it
import numpy as np
import scipy.stats as st

import clef as clf
import utilities as ut


##############
# Kappa algo #
##############


def clef_as(x_rank, delta, k, kappa_min):
    """Returns maximal faces s.t. kappa pass the test."""
    x_bin_k = ut.kth_largest_bin(x_rank, k)
    x_bin_kp = ut.kth_largest_bin(x_rank, k + int(k**(3./4)))
    x_bin_km = ut.kth_largest_bin(x_rank, k - int(k**(3./4)))
    faces_dict = find_faces_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km,
                                  delta, k, var_max=1e3, verbose=0)
    faces = clf.find_maximal_faces(faces_dict)

    return faces


def clef_as_0(x_bin_k, x_bin_kp, x_bin_km, delta, k, kappa_min, var_max=1e3, verbose=0):
    """Returns maximal faces s.t. kappa pass the test."""
    faces_dict = find_faces_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km,
                                  delta, k, var_max, verbose)
    faces = clf.find_maximal_faces(faces_dict)

    return faces


def faces_init_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max, verbose):
    """Returns faces of size 2 s.t. kappa pass the test."""
    n_dim = x_bin_k.shape[1]
    faces = []
    for (i, j) in it.combinations(range(n_dim), 2):
        face = [i, j]
        kap = clf.kappa(x_bin_k, face)
        if kap > 0.:
            var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, face, k)
            if verbose and var > var_max:
                print(f'var={var} for {face}')
            if 0 < var < var_max:
                test = kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k))
                if kap > test:
                    faces.append(face)

    return faces


def find_faces_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max, verbose):
    """Returns all faces s.t. kappa pass the test."""
    dim = x_bin_k.shape[1]
    faces = faces_init_kappa(kappa_min, x_bin_k, x_bin_kp, x_bin_km,
                             delta, k, var_max, verbose)
    size = 2
    faces_dict = {}
    faces_dict[size] = faces
    # print('face size : nb faces')
    while len(faces_dict[size]) > size:
        # print(size, ':', len(faces_dict[size]))
        faces_dict[size + 1] = []
        faces_to_try = ut.candidate_subfaces(faces_dict[size], size, dim)
        if faces_to_try:
            for face in faces_to_try:
                kap = clf.kappa(x_bin_k, face)
                if kap > 0:
                    var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, face, k)
                    if verbose and var > var_max:
                        print(f'var={var} for {face}')
                    if 0 < var < var_max:
                        test = kappa_min + \
                            st.norm.ppf(delta) * np.sqrt(var/float(k))
                        if kap > test:
                            faces_dict[size + 1].append(face)
        size += 1

    return faces_dict


###################
# Kappa estimator #
###################


def kappa_partial_derivs(x_bin_k, x_bin_kp, x_bin_km, face, k):
    """Returns partial derivative of kappa function."""
    kappa_p = {}
    for j in face:
        x_r = ut.partial_matrix(x_bin_k, x_bin_kp, j)
        x_l = ut.partial_matrix(x_bin_k, x_bin_km, j)
        kappa_p[j] = 0.5*k**0.25 * (clf.kappa(x_r, face) -
                                    clf.kappa(x_l, face))

    return kappa_p


def rhos(x_bin, face, k):
    """Computes rho(i,j) for each (i,j) in face and rho(-j)."""
    rhos_face = {}
    for j in face:
        face_tronq = [i for i in face]
        del face_tronq[face_tronq.index(j)]
        rhos_face[j] = ut.rho_value(x_bin, face_tronq, k)
    for (i, j) in it.combinations(face, 2):
        rhos_face[i, j] = ut.rho_value(x_bin, [i, j], k)

    return rhos_face


def var_kappa(x_bin_k, x_bin_kp, x_bin_km, face, k):
    """Returns the variance of the kappa estimator."""
    kappa_face = clf.kappa(x_bin_k, face)
    kappa_p = kappa_partial_derivs(x_bin_k, x_bin_kp, x_bin_km, face, k)
    rhos_face = rhos(x_bin_k, face, k)
    beta_face = float(clf.compute_beta(x_bin_k, face))
    var = ((1 - kappa_face) * kappa_face *
           (beta_face**-1 - sum([kappa_p[j] for j in face])) +
           2*sum([kappa_p[i] * kappa_p[j] * rhos_face[i, j]
                  for (i, j) in it.combinations(face, 2)]) +
           sum([kappa_p[i]**2 for i in face]) +
           kappa_face * sum([kappa_p[j] * (1 - rhos_face[j] * beta_face**-1)
                             for j in face]))

    return var


def kappa_test(x_bin_k, x_bin_kp, x_bin_km, face, k, kappa_min, delta):
    var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, face, k)
    kap = clf.kappa(x_bin_k, face)

    return (kap - (kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k))),
            np.sqrt(var/float(k)))
