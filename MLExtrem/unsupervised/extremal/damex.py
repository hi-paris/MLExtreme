import numpy as np
import utilities as ut


#############
# Functions #
#############


def damex_0(x_bin):
    """binary x_bin -> nb of points per subfaces"""
    n_sample = x_bin.shape[0]
    n_extr_feats = np.sum(x_bin, axis=1)
    n_shared_feats = np.dot(x_bin, x_bin.T)
    exact_extr_feats = (n_shared_feats == n_extr_feats) * (
        n_shared_feats.T == n_extr_feats).T
    feat_non_covered = set(range(n_sample))
    samples_nb = {}
    for i in range(n_sample):
        feats = list(np.nonzero(exact_extr_feats[i, :])[0])
        if i in list(feat_non_covered):
            feat_non_covered -= set(feats)
            if n_extr_feats[i] > 1:
                samples_nb[i] = len(feats)
    ind_sort = np.argsort(list(samples_nb.values()))[::-1]
    faces = [list(np.nonzero(x_bin[list(samples_nb)[i], :])[0])
             for i in ind_sort]
    mass = [list(samples_nb.values())[i] for i in ind_sort]

    return faces, np.array(mass)


def damex(x_norm, radius, eps, nb_min):
    """Returns faces s.t. points per faces > nb_min."""
    x_damex = ut.above_radius_bin(x_norm, radius, eps)
    faces, mass = damex_0(x_damex)

    return faces[:np.sum(mass >= nb_min)]


def list_to_dict_size(list_faces):
    """List of faces -> Dict[size] = [faces,..]"""
    faces_dict = {s: [] for s in range(2, max(list(map(len, list_faces)))+1)}
    for face in list_faces:
        faces_dict[len(face)].append(face)

    return faces_dict
