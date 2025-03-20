
## Author : Anass Aghbalou
## Contributor : Anne Sabourin, Pierre-Antoine Amiand-Leroy


from sklearn.decomposition import PCA
import numpy as np
from scipy.linalg import sqrtm

def tirex_transform(X, y, n_components, k=None, method="FO", mode="TIREX", get_SDR_X=False):
    """
    Perform the TIREX transformation on the input data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target values.

    n_components : int
        The number of components to extract.

    k : int, optional
        The number of samples to consider. If None, it is set to the number of samples in X.

    method : str, optional
        The method to use for transformation. Must be "FO" (First Order) or "SO" (Second Order).

    mode : str, optional
        The mode to use for transformation. Must be "TIREX" or "CUME".

    get_SDR_X : bool, optional
        Whether to return the extreme space in the original feature space.

    Returns
    -------
    extreme_space : array-like of shape (n_features, n_components)
        The extreme space components.

    extreme_space_X : array-like of shape (n_features, n_components), optional
        The extreme space components in the original feature space, if get_SDR_X is True.

    Citation
    -------
    Aghbalou, A. , Portier, F., Sabourin, A., Zhou, C. (2024) Tail inverse regression for dimension reduction with extreme response. Bernoulli.
    """
    if mode not in ["CUME", "TIREX"]:
        raise ValueError("mode must be 'TIREX' or 'CUME'.")

    if mode == "CUME":
        k = X.shape[0]

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    if k is None:
        k = X.shape[0]

    if k > X.shape[0] or k <= 0:
        raise ValueError("k must be within the range (0, n_samples].")

    # Compute the covariance matrix
    first_cov = np.cov(X.T)

    # QR decomposition
    q, r = np.linalg.qr(first_cov)

    # Identify independent columns
    independent_column = [ii for ii in range(len(first_cov)) if (first_cov[ii, ii] > 0 and abs(r[ii, ii]) > 1e-7)]
    X_f = X[:, independent_column]

    # Adjust y based on mode
    Y = -y if mode == "TIREX" else y

    # Sort X_f based on sorted Y
    sorted_X = X_f[Y.argsort()]
    sorted_X -= np.mean(sorted_X, axis=0)

    # Whiten the sorted data
    if len(independent_column) == X.shape[1]:
        inv_root = sqrtm(np.linalg.pinv(first_cov))
        sorted_Z = np.matmul(inv_root, sorted_X.T).T
    else:
        sorted_Z = PCA(whiten=True).fit_transform(sorted_X)

    # Compute the cumulative sum based on the method
    if method == "FO":
        cumsum = np.cumsum(sorted_Z[:k, :], axis=0)
        M_n = np.sum(np.apply_along_axis(lambda x: np.outer(x, x) * (1 / k), -1, cumsum), axis=0) / (k ** 2)
    elif method == "SO":
        B = np.apply_along_axis(lambda x: np.identity(X_f.shape[1]) - np.outer(x, x), -1, sorted_Z[:k, :])
        cumsum = np.cumsum(B[:k, :], axis=0)
        M_n = np.sum(np.matmul(cumsum, cumsum.transpose((0, 2, 1))) / k, axis=0) / (k ** 2)
    else:
        raise ValueError("method must be 'FO' or 'SO'.")

    # Eigen decomposition
    w, v = np.linalg.eigh(M_n)

    # Select the extreme space components
    extreme_space = v[:, (-np.abs(w) / np.sqrt(np.sum(w ** 2))).argsort()[:n_components]]

    if get_SDR_X and len(independent_column) == X.shape[1]:
        inv_root = sqrtm(np.linalg.pinv(first_cov))
        extreme_space_X = np.matmul(inv_root, extreme_space)
        return extreme_space, extreme_space_X
    else:
        return extreme_space
