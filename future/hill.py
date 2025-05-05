import itertools as it
import numpy as np
import scipy.stats as st

import clef as clf
import utilities as ut


#############
# Hill algo #
#############


def hill(x_rank, delta, k, var_max=1e3, verbose=0):
    """Returns maximal faces s.t. eta pass the test."""
    x_bin_k = ut.kth_largest_bin(x_rank, k)
    x_bin_kp = ut.kth_largest_bin(x_rank, k=k + int(k**(3./4)))
    x_bin_km = ut.kth_largest_bin(x_rank, k=k - int(k**(3./4)))
    faces_dict = find_faces_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                 delta, k, var_max, verbose)
    faces = clf.find_maximal_faces(faces_dict)

    return faces


def hill_0(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max=1e3, verbose=0):
    """Returns maximal faces s.t. eta pass the test."""
    faces_dict = find_faces_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                 delta, k, var_max, verbose)
    faces = clf.find_maximal_faces(faces_dict)

    return faces


def faces_init_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max, verbose):
    """Returns faces of size 2 s.t. eta pass the test."""
    n_dim = np.shape(x_bin_k)[1]
    faces = []
    for (i, j) in it.combinations(range(n_dim), 2):
        face = [i, j]
        rho = ut.rho_value(x_bin_k, face, k)
        if rho > 0.:
            var = variance_eta_hill(rho, x_bin_k, x_bin_kp, x_bin_km,
                                    face, k)
            if verbose and var >= var_max:
                print(f'var={var} for {face}')
            if 0 < var < var_max:
                eta = eta_hill(x_rank, face, k)
                if verbose and eta <= 0:
                    print(f'eta={eta} for {face}')
                else:
                    test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                    if eta > test:
                        faces.append(face)

    return faces


def find_faces_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max, verbose):
    """Returns all faces s.t. eta pass the test."""
    dim = x_bin_k.shape[1]
    faces_pairs = faces_init_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                  delta, k, var_max, verbose)
    size = 2
    faces_dict = {}
    faces_dict[size] = faces_pairs
    # print('face size : nb faces')
    while len(faces_dict[size]) > size:
        # print(size, ':', len(faces_dict[size]))
        faces_dict[size + 1] = []
        faces_to_try = ut.candidate_subfaces(faces_dict[size], size, dim)
        if faces_to_try:
            for face in faces_to_try:
                rho = ut.rho_value(x_bin_k, face, k)
                if rho > 0.:
                    var = variance_eta_hill(rho, x_bin_k, x_bin_kp, x_bin_km, face, k)
                    if verbose and var >= var_max:
                        print(f'var={var} for {face}')
                    if 0 < var < var_max:
                        eta = eta_hill(x_rank, face, k)
                        if eta <= 0 and verbose:
                            print(f'eta){eta} for {face}')
                        else:
                            test = 1 - \
                                st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                            if eta > test:
                                faces_dict[size + 1].append(face)
        size += 1

    return faces_dict


##################
# Hill estimator #
##################


def eta_hill(x_rank, face, k):
    """Hill estimator of eta."""
    t_vect = np.min(x_rank[:, face], axis=1)
    t_vect_ordered = t_vect[np.argsort(t_vect)][::-1]
    eta_h = (sum([np.log(t_vect_ordered[j])
                  for j in range(k)])/float(k) -
             np.log(t_vect_ordered[k]))

    return eta_h


def variance_eta_hill(rho, x_bin_k, x_bin_kp, x_bin_km, face, k):
    """Variance of the Hill estimator."""
    rhos = ut.rhos_face_pairs(x_bin_k, face, k)
    r_p = ut.r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km,
                                     face, k)
    var = 1 - 2*rho + (2*sum([r_p[i]*r_p[j]*rhos[i, j]
                              for (i, j) in it.combinations(face, 2)]) +
                       sum(r_p[j]**2 for j in face))/rho

    return var


def hill_test(x_rank, x_bin_k, x_bin_kp, x_bin_km, face, k, delta):
    rho = ut.rho_value(x_bin_k, face, k)
    var = variance_eta_hill(rho, x_bin_k, x_bin_kp, x_bin_km, face, k)
    eta = eta_hill(x_rank, face, k)
    test = eta - (1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k)))

    return test, np.sqrt(var/float(k))
