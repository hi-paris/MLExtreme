# Author: Anne Sabourin
# Date: 2025
"""
Submodule for Dataset Generation and Plotting Utilities

This submodule provides essential functions for generating datasets and plotting utilities used throughout the MLExtreme package. It focuses on the Pareto Margins convention for modeling multivariate extremes, featuring an angular component that follows a flexible Dirichlet Mixture distribution. Additionally, the submodule includes implementations of the multivariate symmetric Logistic model.

For supervised learning scenarios, the submodule supports the generation of datasets consisting of covariate-target pairs, where the target can be either binary or continuous. The covariates are designed to be regularly varying, and the submodule also includes functionality for rescaling targets.

"""

from scipy.special import loggamma  # gamma, betaln
import numpy as np
import random
import matplotlib.pyplot as plt


def normalize_param_dirimix(Mu, wei):
    """
    Modifies the Dirichlet centers matrix `Mu` and the weights `wei`
    for the Dirichlet mixture model in Pareto margins in order to
    satisfy a moments constraint on the angular distribution of
    extremes.

    The constraint is that :math:`E(\\Theta) = (1/d, \\ldots, 1/d)`, where
    :math:`d` is the dimension of the ambient space. For a Dirichlet
    mixture density with parameters :math:`w = wei, M=Mu`, this is equivalent to

    .. math::

        \\sum_{j=1}^k w_j M_{j, \\cdot} = \\left(\\frac{1}{d}, \\ldots, \\frac{1}{d}\\right)

    The function accepts a matrix `Mu` of dimension (k, d) and a weights
    vector `wei` of dimension (k,). It normalizes these inputs to satisfy
    the moment constraint on the angular distribution, which is derived from
    Pareto-margins standardization. This normalization ensures that each row
    of the resulting `Mu_modif` matrix sums to 1, and the barycenter of the
    rows, weighted by `wei_modif`, is :math:`(1/d, \\ldots, 1/d)`. For further
    details, refer to Boldi & Davison and Sabourin & Naveau. The normalization
    process follows the method described in Chiapino et al. (see references below).

    Parameters
    ----------
    Mu : np.ndarray
        A k x p matrix with positive entries, with rows summing to one.
    wei : np.ndarray
        A weights vector of length k, with nonnegative entries, summing to one.

    Returns
    -------
    tuple
        A tuple containing:
            - Mu1 (np.ndarray): The normalized matrix where columns sum to 1.
            - wei1 (np.ndarray): The updated weights vector of length p.

    Example
    -------
    >>> import numpy as np
    >>> p = 3; k = 2
    >>> Mu0 = 3*np.random.random(2*3).reshape(k, p)
    >>> wei0 = 4*np.random.random(k)
    >>> Mu, wei = normalize_param_dirimix(Mu0, wei0)
    >>> print(f'row sums of Mu : {np.sum(Mu, axis=1)}')
    >>> print(f'barycenter of `Mu` rows with weights `wei` : {np.sum(Mu*wei.reshape(-1, 1), axis=0)}')

    References
    ----------
    [1] Boldi, M. O., & Davison, A. C. (2007). A mixture model for multivariate extremes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(2), 217-229.

    [2] Sabourin, A., & Naveau, P. (2014). Bayesian Dirichlet mixture model for multivariate extremes: a re-parametrization. Computational Statistics & Data Analysis, 71, 542-567.

    [3] Chiapino, M., Clémençon, S., Feuillard, V., & Sabourin, A. (2020). A multivariate extreme value theory approach to anomaly clustering and visualization. Computational Statistics, 35(2), 607-628.
    """
    Rho = np.diag(wei) @ Mu
    p = Rho.shape[1]
    Rho_csum = np.sum(Rho, axis=0)  # length p. Barycenter of the input.
    if any(x == 0 for x in Rho_csum):
        raise ValueError("One column of Mu is zero")
    Rho1 = Rho / (Rho_csum * p)  # Rho1 has columns summing to 1/p
    wei1 = np.sum(Rho1, axis=1)  # Updated weights vector of length p
    Mu1 = Rho1 / wei1.reshape(-1, 1)

    return Mu1, wei1


def gen_dirichlet(a, size=1):
    """
    Generate angular random samples from a `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_ with parameter `a`.

    Parameters
    ----------
    a : 1D or 2D np.ndarray, shape (n, p) or (p,)
        Parameters for the Dirichlet distribution of each sample.
    size : int, optional
        Number of samples to generate. Default is 1.

    Returns
    -------
    ndarray, shape (n, p)
        An array of Dirichlet samples, each normalized to sum to 1.
    """
    n = size
    if a.ndim == 1:
        # a is a 1D array
        p = a.shape[0]
    elif a.ndim == 2:
        # a is a 2D array
        p = a.shape[1]
        if n != a.shape[0]:
            raise ValueError(
                "if a is a 2D array, \
            n should be equal to the number of rows of a"
            )
    else:
        raise ValueError("a must be either a 1D or 2D array")
    x = np.random.gamma(a, size=(n, p))
    sm = np.sum(x, axis=1)
    return x / sm[:, None]


def pdf_dirichlet(x, a):
    """
    Evaluate the probability density function of a `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_ with parameter `a`, at each point represented as a row in the data matrix `x`.

    Parameters
    ----------
    x : ndarray, shape (n, d) or (d,)
        Data matrix where each row is a point on the (d-1)-dimensional simplex.
    a : ndarray, shape (d,)
        Parameters for the Dirichlet distribution.

    Returns
    -------
    ndarray, shape (n,)
        Probability density at each row of `x` for the given Dirichlet
        distribution.
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]

    assert np.allclose(np.sum(x, axis=1), 1), "Each row of x must sum to 1"
    assert x.shape[1] == len(a), (
        "Each row of x and a must have \
                                      the same length"
    )

    log_numerator = np.sum((a - 1) * np.log(x), axis=1)
    log_denominator = np.sum(loggamma(a)) - loggamma(np.sum(a))

    log_pdf = log_numerator - log_denominator
    return np.exp(log_pdf)


def pdf_dirimix(x, Mu, wei, lnu):
    """
    Density function of a Dirichlet mixture distribution on the (d-1)-dimensional simplex evaluated at each
    row in the data matrix `x`.

    Each Dirichlet mixture component has weight given by the jth entry
    of `wei`, and Dirichlet parameter (see :func:`pdf_dirichlet`) :math:`a_j=\\nu_j \\mu_j`, where
    :math:`\\nu_j = \\exp(\\textrm{lnu}_j)`, :math:`\\mu_j` is the jth
    row of matrix `Mu`, :math:`\\textrm{lnu}_j` is the jth entry of `lnu`.
    The Dirichlet mixture density writes, for `x` in the unit simplex,

    .. math::

        f(x, \\textrm{Mu}, \\textrm{wei}, \\textrm{lnu}) = \\sum_{j=0}^{k-1} \
    \\textrm{wei}[j] \\textrm{dirichlet}(x | \\textrm{Mu}[j, :], \\exp(\\textrm{lnu}[j])),

    where "dirichlet" is the Dirichlet density with parameter \
    :math:`a=\\nu\\mu`,
    
    .. math::

        \\textrm{dirichlet}(x | \\mu, \\nu) = \\frac{\\Gamma(\\nu)}{\\prod_{i=1}^d \\Gamma(\\nu \\mu_i)}\\prod_{i=1}^d x_i^{\\nu \\mu_i - 1}.

    

    Parameters
    ----------
    x : ndarray, shape (n, d) or (d,)
        Data matrix where each row is a point on the (d-1)-dimensional simplex.
    Mu : ndarray, shape (k, d)
        Matrix of means for the Dirichlet concentration parameters.
        Each row must contain non-negative entries and sum to one.
        A moments constraint arises from standardization to unit Pareto margins.
        This constraint is that

        .. math::

            \\sum_{j=1}^{k} w_j \\mu_j = \\left(\\frac{1}{d}, \\ldots, \\frac{1}{d}\\right).

        No warning nor error is issued if that constraint is not satisfied, to encompass more general situations.
    wei : ndarray, shape (k,)
        Vector of weights for each mixture component. Nonnegative entries, summing to one.
    lnu : ndarray, shape (k,)
        Vector of log-scales for each mixture component.

    
    
    Returns
    -------
    ndarray, shape (n,)
        Density at each row of `x` on the (d-1)-dimensional simplex.
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]

    k = len(wei)
    density = np.zeros(x.shape[0])
    for i in range(k):
        a = Mu[i, :] * np.exp(lnu[i])
        density += wei[i] * pdf_dirichlet(x, a)

    return density


def gen_dirimix(Mu, wei, lnu, size=1):
    """
    Generate angular samples from a mixture of Dirichlet distributions \
    described in :func:`pdf_dirimix`

    Parameters
    -----------
    size : int (optional)
        Number of samples to generate.
    Mu : ndarray, shape (k, p)
        Matrix of means for the Dirichlet components.
    wei : ndarray, shape (k,)
        Vector of weights for each mixture component.
    lnu : ndarray, shape (k,)
        Vector of log-scales for each mixture component.

    Returns
    --------
    ndarray, shape (size, p)
        Array of Dirichlet samples generated from the mixture.
    """
    n = size
    k, p = Mu.shape
    if len(lnu) != k or len(wei) != k:
        raise ValueError("Length of lnu and wei must be equal to k")

    u = np.random.rand(n)
    cum_wei = np.cumsum(wei)

    ms = np.array(
        [
            np.where(cum_wei <= u[i])[0].max() + 1 if np.any(cum_wei <= u[i]) else 0
            for i in range(n)
        ]
    )

    matpars = np.array([Mu[ms[i], :] * np.exp(lnu[ms[i]]) for i in range(n)])
    return gen_dirichlet(a=matpars, size=n)


def plot_pdf_dirimix_2D(Mu, wei, lnu, n_points=500):
    """
    Plot the Dirichlet mixture density on the 1-D simplex in \
ambient dimension 2, see :func:`pdf_dirimix`.

    Parameters
    -----------
    Mu : ndarray, shape (2, k)
        Matrix of means for the Dirichlet concentration parameters.
    lnu : ndarray, shape (k,)
        Vector of log-scales for each mixture component.
    wei : ndarray, shape (k,)
        Vector of weights for each mixture component.
    n_points : int, optional
        Number of points to use for evaluating the density. Default is 500.
    """
    x_values1 = np.linspace(10 ** (-5), 1 - 10 ** (-5), n_points)
    x = np.column_stack((x_values1, 1 - x_values1))
    density_values = pdf_dirimix(x, Mu, wei, lnu)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values1, density_values, label="Mixture Density", color="blue")
    plt.fill_between(x_values1, density_values, alpha=0.3, color="blue")
    plt.title("Mixture Density on the 1-D Simplex")
    plt.xlabel("x[1] first component")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_pdf_dirimix_3D(Mu, wei, lnu, n_points=500):
    """
    Plot the mixture density on the 2D simplex in ambient dimension 3, \
see :func:`pdf_dirimix`.

    Parameters
    -----------
    Mu : ndarray, shape (3, k)
        Matrix of means for the Dirichlet concentration parameters.
    lnu : ndarray, shape (k,)
        Vector of log-scales for each mixture component.
    wei : ndarray, shape (k,)
        Vector of weights for each mixture component.
    n_points : int, optional
        Number of points to use for evaluating the density. Default is 500.
    """
    x1_values = np.linspace(10 ** (-5), 1 - 10 ** (-5), n_points)
    x2_values = np.linspace(10 ** (-5), 1 - 10 ** (-5), n_points)

    X1, X2 = np.meshgrid(x1_values, x2_values)
    X1 = X1.flatten()
    X2 = X2.flatten()

    valid_points = X1 + X2 <= 1 - 10 ** (-5)
    X1 = X1[valid_points]
    X2 = X2[valid_points]
    X_full = np.column_stack((X1, X2, 1 - X1 - X2))
    density_values = pdf_dirimix(X_full, Mu, wei, lnu)

    plt.figure(figsize=(8, 6))
    plt.tricontourf(X1, X2, density_values, 20, cmap="viridis")
    plt.colorbar(label="Density")
    plt.title("Mixture Density on the 2D Simplex")
    plt.xlabel("x1 (first component)")
    plt.ylabel("x2 (second component)")

    plt.plot([0, 1, 0], [1, 0, 0], "k--", lw=2)
    plt.grid(True)
    plt.show()


# ##############################################
# ## Multivariate symmetric / asymmetric logistic Model
# ##############################################
def gen_PositiveStable(alpha, size=1):
    """
    Generate positive stable random variables, useful for generating \
    Multivariate Logistic variables, see :func:`gen_multilog`.

    Parameters
    -----------
    size : int (optional)
        Sample size.
    alpha : float
        Dependence parameter.
  
    Returns
    --------
    ndarray
        Sample of positive stable random variables.
    """
    U = np.pi * np.random.random(size)
    #    U = stat.uniform.rvs(0, np.pi, size=size, random_state=seed)
    W = np.random.exponential(scale=1, size=size)
    # stat.expon.rvs(size=size, random_state=seed)
    Term_1 = (np.sin((1 - alpha) * U) / W) ** ((1 - alpha) / alpha)
    Term_2 = np.sin(alpha * U) / (np.sin(U) ** (1 / alpha))
    return Term_1 * Term_2


def gen_multilog(dim, alpha, size=1):
    """
    Generate multivariate symmetric logistic random variables via Algorithm 2.1 in [1].

    [1]: Stephenson, A. (2003). Simulating multivariate extreme value distributions of logistic type.\
    Extremes, 6, 49-59.

    Parameters
    -----------
    size : int (optional)
        Sample size.
    dim : int
        Dimension.
    alpha : float
        Dependence parameter.
  
    Returns
    --------
    ndarray
        Sample of multivariate logistic random variables.
    """
    W = np.random.exponential(scale=1, size=(size, dim))
    #    W = stat.expon.rvs(size=(size, dim), random_state=seed)
    S = gen_PositiveStable(alpha=alpha, size=size)
    Result = np.zeros((size, dim))
    for ii in range(dim):
        Result[:, ii] = (S / W[:, ii]) ** alpha
    return Result


# ###########################################
# ### Prediction of a missing component
# ###########################################


def transform_target_lin(y, X, norm_func):
    """
    Transform the target vector to achieve approximate independence from the
    norm of `X` and stabilize the learning algorithms by preventing large target values

    This function rescales the original target vector `y` into
    `y' = y / ||X||` to mitigate the influence of the magnitude of `X` on `y`,
    particularly when `||X||` is large. This transformation is useful for
    predicting missing components in heavy-tailed multivariate data.

    Parameters
    ----------
    y : array-like
        The original target vector to be transformed.
    X : array-like
        The matrix whose norm is used to rescale `y`.
    norm_func : callable
        A function that computes the norm of `X`.

    Returns
    -------
    y1 : array-like
        The rescaled target vector.
    """
    norm_X = norm_func(X)
    y1 = y / norm_X
    return y1


def inv_transform_target_lin(y, X, norm_func):
    """
    Inverse transform the rescaled target vector to its original scale.

    This function performs the inverse operation of :func:`transform_target_lin`,
    converting the rescaled target vector back to its original scale by
    multiplying it by the norm of `X`.

    Parameters
    ----------
    y : array-like
        The rescaled target vector to be inverse-transformed.
    X : array-like
        The matrix whose norm was used to rescale `y`.
    norm_func : callable
        The same norm function used in `transform_target_lin`.

    Returns
    -------
    y_orig : array-like
        The original target vector.
    """
    norm_X = norm_func(X)
    y_orig = y * norm_X
    return y_orig


def transform_target_nonlin(y, X, norm_order):
    """Non-linearly transform the target vector to achieve \
    approximate independence from the norm of `X` and stabilize the
    learning algorithms by preventing large target values

    This function rescales the original target vector `y` into `y' = y\
    / ||[X, y]||_q` to mitigate the influence of the magnitude of `X`\
    on `y`, particularly when `||[X, y]||_q` is large. This\
    transformation is useful for predicting missing components in
    heavy-tailed multivariate data.

    Parameters
    ----------
    y : array-like
        The original target vector to be transformed.
    X : array-like
        The matrix whose norm, combined with `y`, is used to rescale `y`.
    norm_order : int
        The order of the norm to be used for the transformation.

    Returns
    -------
    array-like
        The rescaled target vector.

    """
    q = norm_order
    joint = np.hstack((X, y.reshape(-1, 1)))
    norm_full = np.linalg.norm(joint, ord=q, axis=1)
    return y / norm_full


def inv_transform_target_nonlin(y, X, norm_order):
    """
    Inverse transform the rescaled target vector to its original scale in a nonlinear fashion.

    This function performs the inverse operation of
    :func:`transform_target_nonlin`,
    converting the rescaled target vector back to its original scale.

    Parameters
    ----------
    y : array-like
        The rescaled target vector to be inverse-transformed.
    X : array-like
        The matrix whose norm was used to rescale `y`.
    norm_order : int
        The order of the norm used in the transformation.

    Returns
    -------
    array-like
        The original target vector.
    """
    q = norm_order
    norm_x = np.linalg.norm(X, ord=q, axis=1)
    predicted_x = y / (1 - y**q) ** (1 / q) * norm_x
    return predicted_x


def gen_rv_dirimix(
    alpha=1,
    Mu=np.array([[0.5, 0.5]]),
    wei=np.array([1]),
    lnu=np.array([2]),
    scale_weight_noise=1,
    index_weight_noise=None,
    Mu_bulk=None,
    wei_bulk=None,
    lnu_bulk=None,
    size=1,
):
    """Generate `n` points according to a multivariate regularly varying \
    distribution (heavy-tailed) with regular variation `alpha`.

    Each point is generated as `X_i = R_i * Theta_i`, where R_i and
    Theta_i are not independent in the bulk.

    The noise influence decreases proportionally to
    `(scale_weight_noise/R)**(index_weight_noise)`, where `R` is the 1-norm of the sample point.
    This means that as the radius `R` increases, the noise's effect diminishes, controlled by the
    `scale_weight_noise` and `index_weight_noise` parameters.

    Theta is generated as a weighted average of two components:
    - A "bulk" component, which dominates for small to moderate radii.
    - A "tail" component, which becomes more significant as the radius increases.

    Parameters
    -----------
    size : int, optional
        Number of samples to generate. Default is 1.
    alpha : float, optional
        Shape parameter of the Pareto distribution. Default is 1.
    Mu : ndarray, shape (k,d)
        Means for the Dirichlet mixture components.
    wei : ndarray, shape (k,)
        Weights for each mixture component.
    lnu : ndarray, shape (k,)
        Log-scale parameters for the Dirichlet mixture components.
    Mu_bulk : ndarray, shape (k,d), optional
        Parameter matrix of same dimension as Mu, ruling the angular distribution for small to moderate radii.
    scale_weight_noise : float, optional
        Scaling factor (>0) for the angular noise. It controls the overall magnitude of the noise added to the angular component.
        A larger value increases the noise influence, making points more dispersed, while a smaller value reduces noise effect.
        Default is 1.
    index_weight_noise : float, optional
        Exponent ruling how the noise vanishes for large R. It determines how quickly the noise's influence diminishes as the radius `R` increases.
        A higher value makes the noise vanish more rapidly, resulting in less noisy points for larger radii.
        Default is -alpha.
    wei_bulk : ndarray, shape (k,), optional
        Weights for each mixture component in the bulk distribution.
    lnu_bulk : ndarray, shape (k,), optional
        Log-scale parameters for the Dirichlet mixture components in the bulk distribution.

    Returns
    --------
    ndarray, shape (n, d)
        Array of points generated from the Pareto and Dirichlet mixture.

    """
    n = size
    if index_weight_noise is None:
        index_weight_noise = alpha

    R = np.random.pareto(alpha, size=n) + 1
    Theta = gen_dirimix(Mu=Mu, wei=wei, lnu=lnu, size=n)
    dim = np.shape(Mu)[1]
    # k_comp = np.shape(Mu)[0]
    # if Mu_bulk is None:
    #     Mu_bulk = (1 - Mu)/(dim - 1)
    if Mu_bulk is None:
        Mu_bulk_0 = np.maximum((2 / dim - Mu), 10 ** (-5))
        r_0 = np.sum(Mu_bulk_0, axis=1)
        Mu_bulk = Mu_bulk_0 / r_0.reshape(-1, 1)

    if wei_bulk is None:
        wei_bulk = wei  # np.ones(k_comp)/k_comp

    if lnu_bulk is None:
        lnu_bulk = lnu[::-1]

    Noise = gen_dirimix(Mu=Mu_bulk, wei=wei_bulk, lnu=lnu_bulk, size=n)
    # gen_dirichlet(n, ) np.ones((n, np.shape(Mu)[0])))
    w = np.minimum(1, (scale_weight_noise / R) ** (index_weight_noise))
    newTheta = (1 - w[:, np.newaxis]) * Theta + w[:, np.newaxis] * Noise
    X = R[:, np.newaxis] * newTheta

    return X


def gen_classif_data_diriClasses(
    mu0=np.array([0.7, 0.3]), lnu=None, alpha=4, index_weight_noise=1, size=10
):
    """
    Generate synthetic classification data with two classes using a Dirichlet mixture model.

    This function generates synthetic data for binary classification tasks. It creates two classes
    of data points using a Dirichlet mixture model, where each class is characterized by different
    mean parameters. The data points are generated using the `gen_rv_dirimix` function, which
    combines Pareto-distributed radii with Dirichlet-distributed angular components.

    Parameters
    -----------
    mu0 : ndarray, shape (d,), optional
        Mean parameter for the first class. Default is np.array([0.7, 0.3]).
    lnu : ndarray, shape (k,), optional
        Log-scale parameters for the Dirichlet mixture components. If not provided, it is calculated
        based on the mean parameters.
    alpha : float, optional
        Shape parameter of the Pareto distribution. Default is 4.
    index_weight_noise : float, optional
        Exponent ruling how the noise vanishes for large R. Default is 1.
    size : int, optional
        Total number of samples to generate. Default is 10.

    Returns
    --------
    X : ndarray, shape (n, d)
        Array of generated data points.
    y : ndarray, shape (n,)
        Array of class labels corresponding to the data points.

    Notes
    -----
    The function first normalizes the mean parameters and weights for the Dirichlet mixture model.
    It then generates data points for each class using the `gen_rv_dirimix` function, combining them
    into a single dataset. The class labels are randomly permuted to ensure a balanced dataset.
    """
    mu1 = 1 - mu0
    Mu_temp = np.array([mu0, mu1])
    wei_temp = 0.5 * np.ones(2)
    Mu, wei = normalize_param_dirimix(Mu_temp, wei_temp)
    if lnu is None:
        lnu = np.log(2 / np.min(Mu, axis=1))
    size0 = int(size * wei[0])
    size1 = int(size * wei[1])
    Mu0 = Mu[0, :].reshape(1, -1)
    Mu1 = Mu[1, :].reshape(1, -1)
    # Dim = np.shape(Mu)[1]
    data0 = gen_rv_dirimix(
        alpha,
        Mu0,
        wei=np.array([1]),
        lnu=np.array([lnu[0]]),
        scale_weight_noise=10 ** (1 / alpha),  # np.sqrt(Dim),
        index_weight_noise=index_weight_noise,
        size=size0,
    )
    data1 = gen_rv_dirimix(
        alpha,
        Mu1,
        wei=np.array([1]),
        lnu=np.array([lnu[1]]),
        scale_weight_noise=10 ** (1 / alpha),  # np.sqrt(Dim),
        index_weight_noise=index_weight_noise,
        size=size1,
    )
    y = np.vstack(
        (np.zeros(size0).reshape(-1, 1), np.ones(size1).reshape(-1, 1))
    ).flatten()
    X = np.vstack((data0, data1))
    permut = np.random.permutation(size0 + size1)
    X = X[permut, :]
    y = y[permut]
    return X, y


# ## target generation for regression models:
# specific instance of additive noise nodel considered in Huet et al.
def tail_reg_fun_default(angle):
    """Default tail regression function for generating target values.

    This function is used internally as an argument for
    :func:`gen_target_CovariateRV`.  The regression function in the
    tail is defined in a linear model, y = < beta, angle >. `angle` is
    meant to be the angular component of the original input, and
    `beta` is a hard-set vectors with the first half of its entries
    equal to 10, and  the rest equal to 0.1.



    Parameters
    ----------
    angle : np.ndarray
        A 1D or 2D array representing angular components of the data.
        Each element corresponds to an angle in the covariate space. If
        `angle` is a 2D array, each row represents a different observation.

    Returns
    -------
    np.ndarray
        The result of the dot product between the `angle` array and the
        vector `beta`. This output represents the tail regression values
        for the input angles, which can be used to generate target values
        in a tail regression model.
    """
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([10 * np.ones(p1), 0.1 * np.ones(p - p1)])
    return np.dot(angle, beta)


def bulk_reg_fun_default(angle):
    """Default bulk regression function for generating target values.

    This function is used internally as an argument for
    :func:`gen_target_CovariateRV`.  The regression function in the
    bulk is defined in a linear model, y = < beta, angle >. `angle` is
    meant to be the angular component of the original input, and
    `beta` is a hard-set vectors with the first half of its entries
    equal to 0.1, and  the rest equal to 10.

    Parameters
    ----------
    angle : np.ndarray
        A 1D or 2D array representing angular components of the data.
        Each element corresponds to an angle in the covariate space. If
        `angle` is a 2D array, each row represents a different
        observation.

    Returns
    -------
    np.ndarray
        The result of the dot product between the `angle` array and the
        vector `beta`. This output represents the bulk regression values
        for the input angles, which can be used to generate target values
        in a regression model.

    """
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([0.1 * np.ones(p1), 10 * np.ones(p - p1)])
    return np.dot(angle, beta)


def bulk_decay_fun_default(radius, rv_index):
    """Default bulk decay function for generating continuous target values
    in a regression model. 

    This function applies a decay transformation to the input radii,
    and returns : `1 / (radius) ** (rv_index)`

    The ouput is used in function :func:`gen_target_CovariateRV` as a\
     weight to be attributed to the bulk regression function, and the\
     final regression function in :func:`gen_target_CovariateRV` is a\
     weighted average between a bulk and a tail component.
    
    Parameters
    ----------
    radius : np.ndarray
        A 1D array representing the radial distances. Each element
        corresponds to the radius of a data point in the covariate space.

    rv_index : float
        A positive float that determines the rate of decay. Higher values
        result in faster decay. This parameter allows for adjustment of
        the decay function's sensitivity to changes in radius.

    Returns
    -------
    np.ndarray
        An array containing the result of applying the decay function to
        each element in the `radius` array. The output represents the
        decayed values, which can be used to weight the contributions of
        different data points in a regression model.

    """
    return 1 / (radius) ** (rv_index)


def gen_target_CovariateRV(
    X,
    tail_reg_fun=tail_reg_fun_default,
    bulk_reg_fun=bulk_reg_fun_default,
    bulk_decay_fun=bulk_decay_fun_default,
    param_decay_fun=2,
):
    """Generate target values for covariate random variables.

    This function generates target values for a given set of covariate data
    by combining tail and bulk regression functions with a decay function.
    The resulting target values are influenced by both the tail and bulk
    regression functions, weighted by the decay function.

    Parameters
    ----------
    X : np.ndarray
        A 2D array representing covariate data. Each row corresponds to a
        different observation, and each column corresponds to a different
        covariate.

    tail_reg_fun : function, optional
        A function that computes the tail regression values based on the
        angular components of the covariate data. Default is
        `tail_reg_fun_default`.

    bulk_reg_fun : function, optional
        A function that computes the bulk regression values based on the
        angular components of the covariate data. Default is
        `bulk_reg_fun_default`.

    bulk_decay_fun : function, optional.
        A function that computes the
        weight of the bulk component of the regression function,
        based on the radial components of the covariate data. Default
        is `bulk_decay_fun_default`.

    param_decay_fun : any, optional.
        An additional parameter to be
        passed to `bulk_decay_fun`. If `bulk_decay_fun_default` is
        used, then `param_decay_fun` is the regular variation index of
        the weight, as a function of the radius. Default is 2. Larger
        values induce a more rapidly decaying bulk weight.

    Returns
    -------
    y : np.ndarray
        An array of generated target values corresponding to the input
        covariate data. The target values are computed as a weighted
        combination of the tail and bulk regression values, with added
        Gaussian noise.

    """
    rad = np.linalg.norm(X, axis=1)
    ang = X / rad[:, np.newaxis]
    n = len(rad)
    noise = np.random.normal(loc=0, scale=0.01, size=n)
    tail_mean = tail_reg_fun(ang).flatten()
    bulk_mean = bulk_reg_fun(ang).flatten()
    bulk_weight = bulk_decay_fun(rad, param_decay_fun).flatten()
    ystar = (1 - bulk_weight) * tail_mean + bulk_weight * bulk_mean
    y = ystar + noise
    return y


# ## toy example for functional PCA
def gen_rv_functional_data(
    num_samples, grid, alpha, alphanoise, scalenoise=5, om1=1, om2=2, om3=3, om4=4
):
    """
    Generate random regularly varying functions on a grid using
    Pareto-distributed coefficients and noise components.

    Parameters
    -----------
    num_samples : int
        Number of samples to generate.
    grid : array-like
        1D array representing the grid over which the basis functions are
        evaluated.
    alpha : float
        Shape parameter for the 'signal' Pareto-distributed variables.
    alphanoise : float
        Shape parameter for the noise Pareto-distributed variables.
    scalenoise : float, optional
        Scaling factor for the noise components. Default is 5.
    om1, om2, om3, om4 : float, optional
        Frequencies used in the sine and cosine basis functions.
        Defaults are 1, 2, 3, 4.

    Returns
    --------
    ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row
    represents a generated functional data sample.
    """
    a1 = (np.random.pareto(alpha, size=num_samples) + 1) * (
        2 * (np.random.uniform(size=num_samples) > 0.5) - 1
    )
    a2 = (np.random.pareto(alpha, size=num_samples) + 1) * (
        2 * (np.random.uniform(size=num_samples) > 0.5) - 1
    )
    a3 = (
        scalenoise
        * (np.random.pareto(alphanoise, size=num_samples) + 1)
        * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    )
    a4 = (
        scalenoise
        * (np.random.pareto(alphanoise, size=num_samples) + 1)
        * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    )

    result = (
        np.dot(a1[:, None], (1 + np.sin(2 * np.pi * om1 * grid))[None, :])
        + np.dot(a2[:, None], (1 + np.cos(2 * np.pi * om2 * grid))[None, :])
        + np.dot(a3[:, None], (1 + np.sin(2 * np.pi * om3 * grid))[None, :])
        + np.dot(a4[:, None], (1 + np.cos(2 * np.pi * om4 * grid))[None, :])
    )

    return result


def gen_rv_functional_data_gaussianNoise(
    num_samples, grid, alpha, sd, scalenoise=5, om1=1, om2=2, om3=3, om4=4, om5=5, om6=6
):
    """
    Generate random regularly varying functions on a grid using
    Pareto-distributed coefficients and Gaussian noise components.

    Parameters
    -----------
    num_samples : int
        Number of samples to generate.
    grid : array-like
        1D array representing the grid over which the basis functions are
        evaluated.
    alpha : float
        Shape parameter for the Pareto-distributed variables.
    sd : float
        Standard deviation for the Gaussian noise components.
    scalenoise : float, optional
        Scaling factor for the noise components. Default is 5.
    om1, om2, om3, om4, om5, om6 : float, optional
        Frequencies used in the sine and cosine basis functions.
        Defaults are 1, 2, 3, 4, 5, 6.

    Returns
    --------
    ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row
        represents a generated data sample.
    """
    a1 = np.random.pareto(alpha, size=num_samples) + 1
    a2 = 0.8 * (np.random.pareto(alpha, size=num_samples) + 1)
    a3 = scalenoise * np.random.normal(0, sd, size=num_samples)
    a4 = 0.8 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a5 = 0.6 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a6 = 0.4 * scalenoise * np.random.normal(0, sd, size=num_samples)

    result = (
        np.dot(a1[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om1 * grid))[None, :])
        + np.dot(a2[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om2 * grid))[None, :])
        + np.dot(a3[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om3 * grid))[None, :])
        + np.dot(a4[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om4 * grid))[None, :])
        + np.dot(a5[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om5 * grid))[None, :])
        + np.dot(a6[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om6 * grid))[None, :])
    )

    return result


# ######
# subface generation for feature clustering
# #####
# #################
# Generate subfaces #
# #################


def gen_subfaces(
    dimension,
    num_subfaces,
    max_size=8,
    p_geometric=0.25,
    prevent_inclusions=True,
    # include_singletons=True,
    seed=None,
):
    """
    Generates a list of random subsets (subfaces) of the set {1, ..., dimension}.

    This function is used for data generation in feature clustering algorithms.
    It creates random subsets of a specified dimension, ensuring that each subset
    meets certain size and inclusion criteria.

    Parameters
    -----------
    dimension : int
        Dimensionality of the ambient space, i.e., the range of indices for the subsets.
    num_subfaces : int
        Number of subfaces of size >= 2 to generate.
    max_size : int, optional
        Maximum size of a subface. Default is 8.
    p_geometric : float, optional
        Parameter for the geometric distribution ruling the size of subfaces.
        Default is 0.25.
    prevent_inclusions : bool, optional
        If True, ensures that no subface is a subset or superset of another.
        Default is True.
    seed : int, optional
        Seed for random number generation to ensure reproducibility.
        Default is None.

    Returns
    --------
    list
        List of generated subfaces, where each subface is represented as a list of integers.

    Notes
    -----
    The function uses a geometric distribution to determine the size of each subface.
    It ensures that each feature (index) is included in at least one subface to allow
    for standardized components in the associated random vector (dataset).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # create a matrix of subfaces
    subfaces = np.zeros((num_subfaces, dimension))
    # subface_size = min(np.random.geometric(p_geometric) + 1, max_size)
    # subfaces[0, random.sample(range(dimension), subface_size)] = 1
    count = 0
    loop_count = 0
    while count < num_subfaces and loop_count < 1e4:
        subface_size = min(np.random.geometric(p_geometric) + 1, max_size)
        subface = np.zeros(dimension)
        subface_indices = random.sample(range(dimension), subface_size)
        subface[subface_indices] = 1
        if count >= 1:
            valid_cond = (not prevent_inclusions) or (
                # subface is not a subset of one of subfaces
                (np.sum(np.prod(subfaces[:count] * subface == subface, axis=1)) == 0)
                and (
                    # subface is not a superset either
                    np.sum(
                        np.prod(subfaces[:count] * subface == subfaces[:count], axis=1)
                    )
                    == 0
                )
            )
        else:
            valid_cond = True
        if valid_cond:
            subfaces[count, subface_indices] = 1
            count += 1
        loop_count += 1

    idkeep = np.sum(subfaces, axis=1) > 0
    subfaces = subfaces[idkeep]
    # convert the subfaces matrix to a list
    subfaces_list = [list(np.nonzero(f)[0]) for f in subfaces]
    # features = list({int(j) for subface in subfaces_list for j in subface})

    # Each feature must be in at least one subface because otherwise
    # the associated random vector (dataset) cannot have standardized
    # components (columns). The last step is to add potentially
    # missing features:

    missing_features = list(
        set(range(dimension)) - {j for subface in subfaces_list for j in subface}
    )
    # singletons = []

    if missing_features:
        #       if include_singletons:
        # singletons = [[int(j)] for j in missing_features]
        for j in missing_features:
            subfaces_list.append([int(j)])
        # else:
        #     if len(missing_features) > 1:
        #         subfaces_list.append(missing_features)
        #     if len(missing_features) == 1:
        #         missing_features.append(list(set(range(dimension)) -
        #                                      set(missing_features))[0])
        #         subfaces_list.append(missing_features)

    converted_subfaces = [[int(item) for item in sublist] for sublist in subfaces_list]

    # if include_singletons:
    #     return converted_subfaces, features, singletons
    return converted_subfaces
