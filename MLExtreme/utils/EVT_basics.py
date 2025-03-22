import numpy as np
from scipy.special import loggamma  # gamma, betaln
#import scipy.stats as stat
import matplotlib.pyplot as plt

def hill_estimator(x, k):
    """
    Hill estimator for the tail index based on thee data x, using the
    k largest order statistics

    Parameters:
    -----------
    x : array-like
        The input data sample from which to estimate the tail index.

    k : int  >= 2 
        The number of extreme values to consider for the estimation.

    Returns:
    --------
    dict
        A dictionary containing the Hill estimator value ('estimate')
    and its standard deviation ('sdev').
    """
    sorted_data = np.sort(x)[::-1]  # Sort data in decreasing order
    hill_estimate = 1 / (k-1) * np.sum(np.log(sorted_data[:k] / sorted_data[k - 1]))
    standard_deviation = hill_estimate / np.sqrt(k-1)
    return {"estimate": hill_estimate, "sdev": standard_deviation}



# ## rank transformation to approximate unit  pareto margins
def rank_transform(x_raw):
    """
    Standardize each column of the input matrix using a Pareto-based
    rank transformation.

    This function transforms each element in the input matrix by assigning
    a rank based on its position within its column and then applying a
    Pareto transformation.
    The transformation is defined as:

    v_rank_ij = (n_sample + 1) / (rank(x_raw_ij) + 1) =
            1 / ( 1 - (n+1)/n F_emp(x_raw_ij) )

    where `rank(x_raw_ij)` is the rank of the element `x_raw_ij` within its
    column in decreasing order and F_emp is the usual empirical cdf. 

    Parameters:
    -----------
    x_raw : numpy.ndarray
        A 2D array where each column represents a feature and each row
        represents a sample.
        The function assumes that the input is a numerical matrix.

    Returns:
    --------
    v_rank : numpy.ndarray
        A transformed matrix of the same shape as `x_raw`, where each element
        has been standardized using the Pareto-based rank transformation.

    Example:
    --------
    >>> import numpy as np
    >>> x_raw = np.array([[10, 20], [30, 40], [50, 60]])
    >>> rank_transform(x_raw)
    array([[1.33333333, 1.33333333],
           [2.        , 2.        ],
           [4.        , 4.        ]])
    """
    n_sample, n_dim = x_raw.shape
    mat_rank = np.argsort(x_raw, axis=0)[::-1]
    x_rank = np.zeros((n_sample, n_dim))
    for j in range(n_dim):
        x_rank[mat_rank[:, j], j] = np.arange(n_sample) + 1
    v_rank = (n_sample + 1) / x_rank

    return v_rank


def rank_transform_test(x_train, x_test):
    """
    Transform each column  in x_test to approximate unit pareto distribution based on the empirical cumulative distribution function (ECDF) resulting from  the corresponding column in x_train.

    Parameters:
    -----------
    x_train : numpy.ndarray
        A 2D array of shape (n, d) representing the training data.
    x_test : numpy.ndarray
        A 2D array of shape (m, d) representing the test data to be transformed.

    Returns:
    --------
    x_transf : numpy.ndarray
        A transformed version of x_test of shape (m, d), where each element is
        transformed using the formula: 1 / (1 -  n/(n+1) *  F_emp(x_test_ij)).
    Example:
    --------
    >>> x_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> x_test = np.array([[2, 3], [4, 5]])
    >>> rank_transform_test(x_train, x_test)
    array([[1.33333333, 1.33333333],
       [2.        , 2.        ]])
    """
    n_samples, n_dim = x_train.shape
    m_samples, _ = x_test.shape

    # Initialize the transformed array
    x_transf = np.zeros((m_samples, n_dim))

    # Iterate over each column
    for j in range(n_dim):
        # Compute   ECDF*n/(n+1) for the j-th column of x_train
        sorted_col = np.sort(x_train[:, j])
        ecdf = np.searchsorted(sorted_col, x_test[:, j],
                               side='right') / (n_samples+1)

        # Apply the transformation
        x_transf[:, j] = 1 / (1 - ecdf)

    return x_transf

# # Example usage
# x_train = np.array([[1, 2], [3, 4], [5, 6]])
# x_test = np.array([[2, 3], [4, 5]])

# x_transf = rank_transform_test(x_train, x_test)
# print(x_transf)


def normalize_param_dirimix(Mu, wei):
    """
    Modifies the dirichlet centers matrix Mu and the weights wei for the Dirichlet
    mixture model in Pareto margins

    Takes a matrix Mu (k,p) and a weights vector wei (,p),
    and normalizes  both of them  so that
    each column of the resulting matrix sums to 1, and the barycenter of
    the rows with weights `wei` is (1/p, ... 1/p).
    See Boldi & Davison, Sabourin & Naveau. 
    The normalization is performed according to the method described in
    Chiapino et al.

    Parameters:
    -------------
    Mu (np.ndarray): A k x p matrix with positive entries, with rows summing to one. 
    wei (np.ndarray): A weights vector of length p., with nonnegative entries,
    summing to one. 

    Returns:
    --------
    tuple: A tuple containing:
        - Mu1 (np.ndarray): The normalized matrix where columns sum to 1.
        - wei1 (np.ndarray): The updated weights vector of length p.

    Example usage:
    _______________
        p = 3; k = 2
        Mu0 = 3*np.random.random(2*3).reshape(k, p)
        wei0 = 4*np.random.random(k)  
        Mu, wei = normalize_param_dirimix(Mu0, wei0)
        print(f'row sums of Mu : {np.sum(Mu, axis=1)}')
        print(f'barycenter of `Mu` rows with weights `wei` : {\
              np.sum(Mu*wei.reshape(-1, 1), axis=0)}')

    """
    Rho = np.diag(wei) @ Mu
    p = Rho.shape[1]
    Rho_csum = np.sum(Rho, axis=0)
    Rho1 = Rho / (Rho_csum * p)  # Rho1 has columns summing to 1/p
    wei1 = np.sum(Rho1, axis=1)  # Updated weights vector of length p
    Mu1 =  Rho1/wei1.reshape(-1,1)

    return Mu1, wei1


def gen_dirichlet(a, size=1):
    """
    Generate angular random samples from a Dirichlet distribution.

    Parameters:
    -----------
    size : int (optional)
        Number of samples to generate.
    a : 1D or 2D np.array, shape (n, p) or (p, )
        parameters for the Dirichlet distribution of each sample.

    Returns:
    --------
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
            raise ValueError("if a is a 2D array, \
            n should be equal to the number of rows of a")
    else:
        raise ValueError("a must be either a 1D or 2D array")
    x = np.random.gamma(a, size=(n, p))
    sm = np.sum(x, axis=1)
    return x / sm[:, None]


def pdf_dirichlet(x, a):
    """
    Compute the Dirichlet probability density function for each row in the
    data matrix `x`.

    Parameters:
    -----------
    x : ndarray, shape (n, d) or (d,)
        Data matrix where each row is a point on the (d-1)-dimensional simplex.
    a : ndarray, shape (d,)
        parameters for the Dirichlet distribution.

    Returns:
    --------
    ndarray, shape (n,)
        Probability density at each row of `x` for the given Dirichlet
        distribution.
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]

    assert np.allclose(np.sum(x, axis=1), 1), "Each row of x must sum to 1"
    assert x.shape[1] == len(a), "Each row of x and a must have \
                                      the same length"

    log_numerator = np.sum((a - 1) * np.log(x), axis=1)
    log_denominator = np.sum(loggamma(a)) - loggamma(np.sum(a))

    log_pdf = log_numerator - log_denominator
    return np.exp(log_pdf)


def pdf_dirimix(x, Mu,  wei, lnu):
    """
    Compute the mixture density on the (d-1)-dimensional simplex for each row
    in the data matrix `x`.

    Parameters:
    -----------
    x : ndarray, shape (n, d) or (d,)
        Data matrix where each row is a point on the (d-1)-dimensional simplex.
    Mu : ndarray, shape (k, d)
        Matrix of means for the Dirichlet concentration parameters.
    wei : ndarray, shape (k,)
        Vector of weights for each mixture component.
    lnu : ndarray, shape (k,)
        Vector of log-scales for each mixture component.

    Returns:
    --------
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
    Generate angular samples from a mixture of Dirichlet distributions.

    Parameters:
    -----------
    size : int (optional)
        Number of samples to generate.
    Mu : ndarray, shape (k, p)
        Matrix of means for the Dirichlet components.
    wei : ndarray, shape (k,)
        Vector of weights for each mixture component.
    lnu : ndarray, shape (k,)
        Vector of log-scales for each mixture component.

    Returns:
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

    ms = np.array([
        np.where(cum_wei <= u[i])[0].max()+1 if np.any(cum_wei <= u[i]) else 0
        for i in range(n)
    ])

    matpars = np.array([Mu[ms[i], :] * np.exp(lnu[ms[i]]) for i in range(n)])
    return gen_dirichlet(a=matpars, size=n)


def plot_pdf_dirimix_2D(Mu, wei, lnu, n_points=500):
    """
    Plot the mixture density on the 1-D simplex in ambient dimension 2.

    Parameters:
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
    x_values1 = np.linspace(10**(-5), 1-10**(-5), n_points)
    x = np.column_stack((x_values1, 1 - x_values1))
    density_values = pdf_dirimix(x, Mu,  wei, lnu)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values1, density_values, label="Mixture Density", color='blue')
    plt.fill_between(x_values1, density_values, alpha=0.3, color='blue')
    plt.title("Mixture Density on the 1-D Simplex")
    plt.xlabel("x[1] first component")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_pdf_dirimix_3D(Mu,  wei, lnu,  n_points=500):
    """
    Plot the mixture density on the 2D simplex (represented as an equilateral
    triangle).

    Parameters:
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
    x1_values = np.linspace(10**(-5), 1-10**(-5), n_points)
    x2_values = np.linspace(10**(-5), 1-10**(-5), n_points)

    X1, X2 = np.meshgrid(x1_values, x2_values)
    X1 = X1.flatten()
    X2 = X2.flatten()

    valid_points = X1 + X2 <= 1
    X1 = X1[valid_points]
    X2 = X2[valid_points]
    X_full = np.column_stack((X1, X2, 1 - X1 - X2))
    density_values = pdf_dirimix(X_full, Mu,  wei, lnu)

    plt.figure(figsize=(8, 6))
    plt.tricontourf(X1, X2, density_values, 20, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title("Mixture Density on the 2D Simplex")
    plt.xlabel("x1 (first component)")
    plt.ylabel("x2 (second component)")

    plt.plot([0, 1, 0], [1, 0, 0], 'k--', lw=2)
    plt.grid(True)
    plt.show()


# Multivariate symmetric logistic Model
def gen_PositiveStable(alpha, size=1):
    """
    Generate positive stable random variables.
    See "Simulating Multivariate Extreme Value Distributions of Logistic Type",
    2003, A. Stephenson for more details.

    Parameters:
    -----------
    size : int (optional)
        Sample size.
    alpha : float
        Dependence parameter.
  
    Returns:
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
    Generate multivariate logistic random variables.

    Parameters:
    -----------
    size : int (optional)
        Sample size.
    dim : int
        dimension.
    alpha : float
        Dependence parameter.
  
    Returns:
    --------
    ndarray
        Sample of multivariate logistic random variables.
    """
    W = np.random.exponential(scale=1, size=(size,dim))
#    W = stat.expon.rvs(size=(size, dim), random_state=seed)
    S = gen_PositiveStable(alpha=alpha, size=size)
    Result = np.zeros((size, dim))
    for ii in range(dim):
        Result[:, ii] = (S / W[:, ii]) ** alpha
    return Result
