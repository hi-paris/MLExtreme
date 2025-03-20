import numpy as np
def hill_estimator(x, k):
    """
    Hill estimator for the tail index based on thee data x, using the k largest order statistics 

    Parameters:
    -----------
    x : array-like
        The input data sample from which to estimate the tail index.

    k : int  >= 2 
        The number of extreme values to consider for the estimation.

    Returns:
    --------
    dict
        A dictionary containing the Hill estimator value ('estimate') and its standard deviation ('sdev').
    """
    sorted_data = np.sort(x)[::-1]  # Sort data in decreasing order
    hill_estimate = 1 / (k-1) * np.sum(np.log(sorted_data[:k] / sorted_data[k - 1]))
    standard_deviation = hill_estimate / np.sqrt(k-1)
    return {"estimate": hill_estimate, "sdev": standard_deviation}



### rank transformation to approximate unit  pareto margins
def rank_transform(x_raw):
    """
    Standardize each column of the input matrix using a Pareto-based rank transformation.

    This function transforms each element in the input matrix by assigning a rank based on its
    position within its column and then applying a Pareto transformation. The transformation
    is defined as:

    v_rank_ij = (n_sample + 1) / (rank(x_raw_ij) + 1) = 1 / ( 1 - (n+1)/n F_emp(x_raw_ij) ) 

    where `rank(x_raw_ij)` is the rank of the element `x_raw_ij` within its column in decreasing order and F_emp is the usual empirical cdf. 

    Parameters:
    -----------
    x_raw : numpy.ndarray
        A 2D array where each column represents a feature and each row represents a sample.
        The function assumes that the input is a numerical matrix.

    Returns:
    --------
    v_rank : numpy.ndarray
        A transformed matrix of the same shape as `x_raw`, where each element has been
        standardized using the Pareto-based rank transformation.

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
        A transformed version of x_test of shape (m, d), where each element is transformed
        using the formula: 1 / (1 -  n/(n+1) *  F_emp(x_test_ij)).
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



## all this is in scipy.stats. 
# def gen_pareto(num_samples, alpha, x_min=1):
#     """
#     Generate samples from a Pareto distribution.

#     Parameters:
#     -----------
#     num_samples : int
#         The number of samples to generate.

#     alpha : float
#         The shape parameter of the Pareto distribution (alpha >0).
#     x_min : float, optional, default to 1
#         The minimum value (scale) for the Pareto distribution (default is 1).
    
#     Returns:
#     --------
#     ndarray, shape (n,)
#         A NumPy array of `n` random samples from the Pareto distribution.
#     """
#     uniform_samples = np.random.uniform(size=num_samples)
#     pareto_samples = x_min * (1 - uniform_samples)**(-1 / alpha)
#     return pareto_samples

# def pareto_quantile(prob, alpha):
#     """
#     Calculate the quantile function (inverse CDF) of a Pareto distribution.

#     Parameters:
#     -----------
#     prob : array-like
#         The probability values at which to evaluate the quantile function.

#     alpha : float
#         The shape parameter of the Pareto distribution.

#     Returns:
#     --------
#     array
#         The quantile values corresponding to the input probabilities.
#     """
#     return (1 - prob)**(-1 / alpha)

# def pareto_cdf(x, alpha, scale=1):
#     """
#     Calculate the cumulative distribution function (CDF) of a Pareto distribution.

#     Parameters:
#     -----------
#     x : array-like
#         The values at which to evaluate the CDF.

#     alpha : float
#         The shape parameter of the Pareto distribution.

#     scale : float, optional
#         The scale parameter of the Pareto distribution. Default is 1.

#     Returns:
#     --------
#     array
#         The CDF values corresponding to the input values.
#     """
#     return (x / scale)**(-alpha)

