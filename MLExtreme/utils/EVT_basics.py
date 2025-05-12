import numpy as np
from scipy.special import loggamma  # gamma, betaln
import matplotlib.pyplot as plt
import warnings

def round_signif(numbers, digits):
    """
    Round each number in a list, NumPy array, or a single float to a specified
    number of significant digits.

    Parameters: 
    ----------
    numbers : float, list, or np.ndarray
        The number, list, or array of numbers to round.
    
    digits : int
        The number of significant digits to round to.

    Returns:
    -------
    float, list, or np.ndarray
        The rounded numbers in the same format as the input.

    Examples:
    --------
    >>> import numpy as np
    >>> statistics_array = np.array([0.12345, 6.789, 0.00567])
    >>> statistics_list = [0.12345, 6.789, 0.00567]
    >>> single_float = 0.12345

    # Round each element to two significant digits

    >>> rounded_statistics_array = round_signif(statistics_array, 2)
    >>> rounded_statistics_list = round_signif(statistics_list, 2)
    >>> rounded_single_float = round_signif(single_float, 2)

    # Print the rounded statistics

    >>> print(f'Statistics (array): {rounded_statistics_array}')
    Statistics (array): [0.12 6.8 0.0057]
    >>> print(f'Statistics (list): {rounded_statistics_list}')
    Statistics (list): [0.12, 6.8, 0.0057]
    >>> print(f'Single float: {rounded_single_float}')
    Single float: 0.12
    """
    if isinstance(numbers, float):
        # Round a single float
        return float(round(numbers, digits -
                           int(np.floor(np.log10(abs(numbers)))) - 1)) if \
                           numbers != 0 else 0.0
    elif isinstance(numbers, np.ndarray):
        # Apply rounding element-wise and return as a NumPy array
        return np.array([float(round(num, digits -
                                     int(np.floor(np.log10(abs(num)))) - 1)) if
                         num != 0 else 0.0 for num in numbers])
    elif isinstance(numbers, list):
        # Apply rounding element-wise and return as a list
        return [float(round(num, digits -
                            int(np.floor(np.log10(abs(num)))) - 1)) if
                num != 0 else 0.0 for num in numbers]
    else:
        raise TypeError(
            "Input must be a float, list, or NumPy array of floats.")


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
        A dictionary containing the Hill estimator value ('estimate') and its \
    standard deviation ('sdev').
    """
    sorted_data = np.sort(x)[::-1]  # Sort data in decreasing order
    hill_estimate = 1 / (k-1) * np.sum(np.log(sorted_data[:k] / sorted_data[k - 1]))
    standard_deviation = hill_estimate / np.sqrt(k-1)
    return {"estimate": hill_estimate, "sdev": standard_deviation}


def rank_transform(x_raw):
    """
    Standardize each column of the input matrix using a Pareto-based
    rank transformation.

    This function transforms each element in the input matrix by assigning
    a rank based on its position within its column and then applying a
    Pareto transformation.
    The transformation is defined as:

    v_rank_ij = (n_sample + 1) / (rank(x_raw_ij) + 1) =
            1 / ( 1 -  n/(n+1) * F_emp(x_raw_ij) )

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
    numpy.ndarray
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
    Transform each column  in x_test to approximate unit pareto distribution \
    based on the empirical cumulative distribution function (ECDF) resulting \
    from  the corresponding column in x_train.

    Parameters:
    -----------
    x_train : numpy.ndarray
        A 2D array of shape (n, d) representing the training data.
    x_test : numpy.ndarray
        A 2D array of shape (m, d) representing the test data to be \
    transformed.

    Returns:
    --------
    numpy.ndarray
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
    Modifies the Dirichlet centers matrix `Mu` and the weights `wei`
    for the Dirichlet mixture model in Pareto margins in order to
    satisfy a moments constraint on the angular distribution of
    extremes.

    The constraint is that :math:`E(\\Theta) = (1/d, \\ldots, 1/d)`, where
    :math:`d` is the dimension of the ambient space. For a Dirichlet
    mixture density with parameters :math:`w = wei, M=Mu`, this is equivalent to

    .. math::

        \\sum_{j=1}^k w_j M_{j, \\, \\cdot \\,} = (1/d, \\ldots, 1/d)
    

    The function takes as argument a matrix Mu of dimension (k,d) and a weights
    vector wei of dimension (,k), and normalizes them in order to satisfy moments
    constraint on the angular distribution, which arises from
    Pareto-margins standardization both of them so that each row of
    the resulting Mu_modif matrix sums to 1, and the barycenter of the
    rows with weights `wei_modif` is :math:`(1/d, ... 1/d)`.  See Boldi &
    Davison, Sabourin & Naveau.  The normalization is performed
    according to the method described in Chiapino et al. (see references below)

    Parameters:
    -------------
    Mu : np.ndarray
        A k x p matrix with positive entries, with rows summing to one.
    
    wei : np.ndarray
        A weights vector of length k., with nonnegative entries,summing to one.

    Returns:
    ----------
    A tuple containing:
        - Mu1 (np.ndarray): The normalized matrix where columns sum to 1.
        - wei1 (np.ndarray): The updated weights vector of length p.

    Example usage:
    -----------------
        >>> p = 3; k = 2
        >>> Mu0 = 3*np.random.random(2*3).reshape(k, p)
        >>> wei0 = 4*np.random.random(k)  
        >>> Mu, wei = normalize_param_dirimix(Mu0, wei0)
        >>> print(f'row sums of Mu : {np.sum(Mu, axis=1)}')
        >>> print(f'barycenter of `Mu` rows with weights `wei` : \
{np.sum(Mu*wei.reshape(-1, 1), axis=0)}')

    References:
    _______________
    [1] Boldi, M. O., & Davison, A. C. (2007). A mixture model for multivariate extremes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(2), 217-229.
    
    [2] Sabourin, A., & Naveau, P. (2014). Bayesian Dirichlet mixture model for multivariate extremes: a re-parametrization. Computational Statistics & Data Analysis, 71, 542-567.
    
    [3] Chiapino, M., Clémençon, S., Feuillard, V., & Sabourin, A. (2020). A multivariate extreme value theory approach to anomaly clustering and visualization. Computational Statistics, 35(2), 607-628.
    
    """
    Rho = np.diag(wei) @ Mu
    p = Rho.shape[1]
    Rho_csum = np.sum(Rho, axis=0) # length p. Barycenter of the input.
    if any(x == 0 for x in Rho_csum):
        raise ValueError("One column of Mu is zero")
    Rho1 = Rho / (Rho_csum * p)  # Rho1 has columns summing to 1/p
    wei1 = np.sum(Rho1, axis=1)  # Updated weights vector of length p
    Mu1 =  Rho1/wei1.reshape(-1, 1)

    return Mu1, wei1


def gen_dirichlet(a, size=1):
    """
    Generate angular random samples from a `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_ with parameter `a`.

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
    Evaluate the probability density function of a `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_ with parameter `a`, at each point represented as a row in the data matrix `x`.

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
    """Density function of a Dirichlet mixture distribution on the (d-1)-dimensional simplex evaluated at each \
    row in the data matrix `x`.

    Each Dirichlet mixture component has weight given by the jth entry
    of `wei`,  and Dirichlet parameter (see :func:`pdf_dirichlet`) :math:`a_j=\\nu_j \\mu_j`,  where
    :math:`\\nu_j = \\exp(\\textrm{lnu}_j)`, :math:`\\mu_j` is the jth
    row of matrix `Mu`, :math:`\\textrm{lnu}_j` is the jth entry of `lnu`. 

    Parameters:
    -----------
    x : ndarray, shape (n, d) or (d,).
        Data matrix where each row is a point on the (d-1)-dimensional simplex.

    Mu : ndarray, shape (k, d).
         Matrix of means for the Dirichlet
    concentration parameters. Each row must contain non-negative
    entries and sum to one. A  moments constraint arises  from \
    standardization to unit Pareto margins. This constrain is that 
    
    .. math::
    
        \\sum_{j=1}^{k} w_j \\mu_j = (1/d, \\ldots, 1/d).

    No warning nor error is issued is that constraint is not satisfied,
    to encompass more general situations.

    wei : ndarray, shape (k,).
    Vector of  weights for each mixture component.
    Nonnegative entries, summing to one. 

    lnu : ndarray, shape (k,).
    Vector of log-scales for each mixture component.

    Details:
    _______
    The Dirichlet mixture density writes, for `x` in the unit simplex,
    
    .. math::

        f(x, \\textrm{Mu}, \\textrm{wei}, \\textrm{lnu}) = \
\\sum_{j=0}^{k-1}  \\textrm{wei}[j] \\textrm{dirichlet}(x | \
\\textrm{Mu}[j, :],\\exp(\\textrm{lnu}[j]) ), 
    
    where "dirichlet" is the dirichlet density with parameter
    :math:`a=\\nu\\mu`,

    .. math::
    
        \\textrm{dirichlet}(x, | \\mu, \\nu ) = \\frac{\\Gamma(\\nu)}{\
    \\prod_{i=1}^d \\Gamma(\\nu \\mu_i)} x_i^{\\nu\\mu_i - 1}. 
    
    
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
    Generate angular samples from a mixture of Dirichlet distributions described in :func:`pdf_dirimix`

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

    Details:
    ________
    See :func:`pdf_dirimix`

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
    Plot the Dirichlet mixture density on the 1-D simplex in \
ambient dimension 2, see :func:`pdf_dirimix`.

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
    Plot the mixture density on the 2D simplex in ambient dimension 3, \
see :func:`pdf_dirimix`.

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

    valid_points = X1 + X2 <= 1-10**(-5)
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


# ##############################################
# ## Multivariate symmetric / asymmetric logistic Model
# ##############################################
def gen_PositiveStable(alpha, size=1):
    """
    Generate positive stable random variables, useful for generating Multivariate Logistic variables,\
see :func:`gen_multilog'.

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
    Generate multivariate symmetric logistic random variables via Algorithm 2.1 in [1].

    [1]: Stephenson, A. (2003). Simulating multivariate extreme value distributions of logistic type.\
    Extremes, 6, 49-59.

    Parameters:
    -----------
    size : int (optional)
        Sample size.
    dim : int
        Dimension.
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


# ###########################################
# ### Prediction of a missing component
# ###########################################

def transform_target_lin(y, X, norm_func):
    """
    Transform the target vector to achieve approximate independence from the
    norm of X.

    This function rescales the original target vector `y` into
    `y' = y / ||X||` to mitigate the influence of the magnitude of `X` on `y`,
    particularly when `||X||` is large. This transformation is useful for
    predicting missing components in heavy-tailed multivariate data.

    Parameters:
    - y (array-like): The original target vector to be transformed.
    - X (array-like): The matrix whose norm is used to rescale `y`.
    - norm_func (callable): A function that computes the norm of `X`.

    Returns:
    - y1 (array-like): The rescaled target vector.
    """
    norm_X = norm_func(X)
    y1 = y / norm_X
    return y1


def inv_transform_target_lin(y, X, norm_func):
    """
    Inverse transform the rescaled target vector to its original scale.

    This function performs the inverse operation of `transform_target_lin`,
    converting the rescaled target vector back to its original scale by
    multiplying it by the norm of `X`.

    Parameters:
    - y (array-like): The rescaled target vector to be inverse-transformed.
    - X (array-like): The matrix whose norm was used to rescale `y`.
    - norm_func (callable): The same norm function used in
                            `transform_target_lin`.

    Returns:
    - y_orig (array-like): The original target vector.
    """
    norm_X = norm_func(X)
    y_orig = y * norm_X
    return y_orig


def transform_target_nonlin(y, X, norm_order):
    q = norm_order
    joint = np.hstack((X, y.reshape(-1, 1)))
    norm_full = np.linalg.norm(joint, ord=q, axis=1)
    return y / norm_full


def inv_transform_target_nonlin(y, X, norm_order):
    q = norm_order
    norm_x = np.linalg.norm(X, ord=q, axis=1)
    predicted_x = y/(1-y**q)**(1/q) * norm_x
    return predicted_x

# %% 
def test_indep_radius_rest(X, y, ratio_ext, norm_func,
                           random_state=np.random.randint(10**5)):
    from dcor.independence import distance_covariance_test
    from dcor.independence import distance_correlation_t_test

    # 
    if isinstance(ratio_ext, float):
        ratio_ext = [ratio_ext]
    norm_X = norm_func(X)
    Theta = X/norm_X.reshape(-1, 1)
    if y is  None:
        Z = Theta
    else:
        Z = np.column_stack((Theta, y.reshape(-1, 1)))
    pvalues = []
    count = 0
    for ratio in ratio_ext:
        count += 173
        threshold = np.quantile(norm_X, 1-ratio)
        id_extreme = (norm_X >= threshold)
        r_ext = norm_X[id_extreme]
        Z_ext = Z[id_extreme, :]
        if len(r_ext) < 100:
            # perform a distance covariance test with permutation-based
            # computation of the p-value
            test = distance_covariance_test(
                x=Z_ext,
                y=np.log(1+r_ext).reshape(-1, 1),
                num_resamples=500,
                random_state=random_state + count,
            )
        else:
            # perform a distance covariance test with asymptotic p-value
            test = distance_correlation_t_test(
                x=Z_ext,
                y=np.log(1+r_ext).reshape(-1, 1)
                )

        pvalues.append(test.pvalue)
        
    pvalues = np.array(pvalues)
    # Find indices where the condition is satisfied
    indices = np.where(pvalues > 0.05)[0]

    # Check if any indices satisfy the condition
    if len(indices) > 0:
        i_max = np.max(indices)
    else:
        warnings.warn("No indices satisfy the condition pvalues > 0.05.",
                      UserWarning)
        
        i_max = 0
    
    ratio_max = ratio_ext[i_max]
    return pvalues, ratio_max 


def plot_indep_radius_rest(pvalues, ratio_ext, ratio_max, n):
    kk = (ratio_ext * n).astype(int)
    k_max = int(ratio_max * n)
    fig, ax = plt.subplots()
#    colors = ['green' if p > 0.05 else 'red' for p in pvalues]
    ax.scatter(kk, pvalues, c='black', label='distance correlation pvalues')
    ax.plot(k_max*np.ones(2), np.linspace(0, np.max(pvalues), num=2),
            c='red',
            label="selected k with distance covariance rule-of-thumb")
    # Add a secondary x-axis with a different scale
    def ratio_transform(x):
        return x / n

    def inverse_ratio_transform(x):
        return x * n
    
    secax = ax.secondary_xaxis('top',
                               functions=(ratio_transform,
                                          inverse_ratio_transform))
    secax.set_xlabel('Ratio (k / n)')
    ax.set_title('distance correlation test for extreme norm(X) Values vs Rest')
    ax.set_xlabel('k')
    ax.set_ylabel('p-value')
    ax.legend()
    plt.grid()
    plt.show()

