import itertools as it
import numpy as np
import scipy.stats as st

import clef as clf
import utilities as ut


#############
# Peng algo #
#############


def peng(x_lgtc, delta, k, rho_min):
    """Returns maximal faces s.t. eta pass the test."""
    x_bin_k = ut.kth_largest_bin(x_lgtc, k)
    x_bin_kp = ut.kth_largest_bin(x_lgtc, k + int(k**(3./4)))
    x_bin_km = ut.kth_largest_bin(x_lgtc, k - int(k**(3./4)))
    x_bin_2k = ut.kth_largest_bin(x_lgtc, 2*k)
    faces_dict = find_faces_peng(x_bin_k, x_bin_2k, x_bin_kp,
                                 x_bin_km, delta, k, rho_min, verbose=0)
    faces = clf.find_maximal_faces(faces_dict)

    return faces


def peng_0(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k, rho_min=0, verbose=0):
    """Returns maximal faces s.t. eta pass the test."""
    faces_dict = find_faces_peng(x_bin_k, x_bin_2k, x_bin_kp,
                                 x_bin_km, delta, k, rho_min, verbose)
    faces = clf.find_maximal_faces(faces_dict)

    return faces


def faces_init_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k, rho_min, verbose):
    """Returns faces of size 2 s.t. eta pass the test."""
    n_dim = np.shape(x_bin_k)[1]
    faces = []
    for (i, j) in it.combinations(range(n_dim), 2):
        face = [i, j]
        r_k = ut.rho_value(x_bin_k, face, k)
        if verbose and r_k <= rho_min:
            print(f'rho={r_k} for {face}')
        if r_k > rho_min:
            r_2k = ut.rho_value(x_bin_2k, face, k)
            if r_2k > r_k:
                eta = np.log(2)/np.log(r_2k/float(r_k))
                var = var_eta_peng(x_bin_k, x_bin_2k,
                                   x_bin_kp, x_bin_km, face, k)
                if var > 0:
                    test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                    if eta > test:
                        faces.append(face)

    return faces


def find_faces_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k, rho_min, verbose):
    """Returns all faces s.t. eta pass the test."""
    dim = x_bin_k.shape[1]
    faces_pairs = faces_init_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                                  delta, k, rho_min, verbose)
    size = 2
    faces_dict = {}
    faces_dict[size] = faces_pairs
    # print('face size : nb faces')
    while len(faces_dict[size]) > size:
        # print(size, ':', len(faces_dict[size]))
        faces_dict[size + 1] = []
        faces_to_try = ut.candidate_faces(faces_dict[size], size, dim)
        if faces_to_try:
            for face in faces_to_try:
                r_k = ut.rho_value(x_bin_k, face, k)
                if verbose and r_k <= rho_min:
                    print(f'rho={r_k} for {face}')
                if r_k > rho_min:
                    r_2k = ut.rho_value(x_bin_2k, face, k)
                    if r_2k > r_k:
                        eta = np.log(2)/np.log(r_2k/float(r_k))
                        var = var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp,
                                           x_bin_km,
                                           face, k)
                        if var > 0:
                            test = 1 - \
                                st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                            if eta > test:
                                faces_dict[size + 1].append(face)
        size += 1

    return faces_dict


##################
# Peng estimator #
##################


def eta_peng(x_bin_k, x_bin_2k, face, k):
    """Returns Peng estimator of eta."""
    r_k = ut.rho_value(x_bin_k, face, k)
    r_2k = ut.rho_value(x_bin_2k, face, k)
    if (r_k == 0 or r_2k == 0):
        eta_face = 0.
    elif r_k == r_2k:
        eta_face = 0.
    else:
        eta_face = np.log(2)/np.log(r_2k/float(r_k))

    return eta_face


def var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                 face, k):
    """Returns the variance of the Peng estimator."""
    rho = ut.rho_value(x_bin_k, face, k)
    rhos = ut.rhos_face_pairs(x_bin_k, face, k)
    r_p = ut.r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km,
                                     face, k)
    r_ij = {(i, j): ut.rho_value(ut.partial_matrix(x_bin_2k, x_bin_k, j),
                                 [i, j], k)
            for (i, j) in it.combinations(face, 2)}
    r_ji = {(i, j): ut.rho_value(ut.partial_matrix(x_bin_k, x_bin_2k, j),
                                 [i, j], k)
            for (i, j) in it.combinations(face, 2)}
    var = ((2 * (rho * np.log(2))**2)**-1 *
           (rho +
            sum([r_p[j] * (-4*rho +
                           2*ut.rho_value(ut.partial_matrix(x_bin_k,
                                                            x_bin_2k, j),
                                          face, k)) for j in face]) +
            sum([r_p[i]*r_p[j] * (3*rhos[i, j] - 2*r_ij[i, j])
                 for (i, j) in it.combinations(face, 2)]) +
            sum([r_p[i]*r_p[j] * (3*rhos[i, j] - 2*r_ji[i, j])
                 for (i, j) in it.combinations(face, 2)]) +
            sum([r_p[i]**2 for i in face])))

    return var


def peng_test(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, face, k, delta):
    var = var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, face, k)
    eta = eta_peng(x_bin_k, x_bin_2k, face, k)

    return (eta - (1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))),
            np.sqrt(var/float(k)))
