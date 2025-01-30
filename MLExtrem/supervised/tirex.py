from sklearn.decomposition import PCA
import numpy as np
from scipy.linalg import sqrtm

def tirex_transform(X, y, n_components, k=None, method="FO", mode="TIREX", get_SDR_X=False):
    if mode !="CUME" and mode!="TIREX":
        ValueError("mode must be 'TIREX' or 'CUME'.")
    if mode == "CUME":
        k = X.shape[0]
    if X.shape[0] != y.shape[0] or k > X.shape[0] or k <= 0:
        raise ValueError("X and y must have the same length and k must be within a valid range.")

    first_cov = np.cov(X.T)
    q, r = np.linalg.qr(first_cov) #PA : QR décomposition, rename  ?
    independent_column = [ii for ii in range(len(first_cov)) if (first_cov[ii, ii] > 0 and abs(r[ii, ii]) > 1e-7)]
    X_f = X[:, independent_column]

    if mode == "TIREX":
        Y = -y
    else:
        Y = y

    sorted_X = X_f[Y.argsort()]
    sorted_X -= np.mean(sorted_X, axis=0)

    if len(independent_column) == X.shape[1]:
        inv_root = sqrtm(np.linalg.pinv(first_cov))
        sorted_Z = np.matmul(inv_root, sorted_X.T).T
    else:
        sorted_Z = PCA(whiten=True).fit_transform(sorted_X)

    if method == "FO":
        cumsum = np.cumsum(sorted_Z[:k, :], axis=0)
        M_n = np.sum(np.apply_along_axis(lambda x: np.matmul(x.reshape(-1, 1), x.reshape(-1, 1).T) * (1 / k), -1, cumsum), axis=0) / (k ** 2)
    else:  # method == "SO", error to print ?
        B = np.apply_along_axis(lambda x: np.identity(X_f.shape[1]) - np.outer(x, x), -1, sorted_Z[:k, :])
        cumsum = np.cumsum(B[:k, :], axis=0)
        M_n = np.sum(np.matmul(cumsum, cumsum.transpose((0, 2, 1))) / k, axis=0) / (k ** 2)

    w, v = np.linalg.eigh(M_n)
    extreme_space = v[:, (-np.abs(w) / np.sqrt(np.sum(w ** 2))).argsort()[:n_components]]

    if get_SDR_X and len(independent_column) == X.shape[1]:
        inv_root = sqrtm(np.linalg.pinv(first_cov))
        extreme_space_X = np.matmul(inv_root, extreme_space)
        return extreme_space, extreme_space_X
    else:
        return extreme_space



