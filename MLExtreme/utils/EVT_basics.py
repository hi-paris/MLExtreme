# Author: Anne Sabourin
# Date: 2025
"""
Tools for Extreme Value Analysis in the MLExtreme Package

This submodule provides essential tools for conducting Extreme Value Analysis (EVA) in various contexts and examples within the MLExtreme package. It includes functionalities for rank transformation and a specialized tool for selecting extreme thresholds based on distance covariance tests. These tools are designed to support both multivariate supervised and unsupervised settings, facilitating robust analysis and decision-making in extreme value scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings


def round_signif(numbers, digits):
    """
    Round each number in a list, NumPy array, or a single float to a specified
    number of significant digits.

    Parameters
    ----------
    numbers : float, list, or np.ndarray
        The number, list, or array of numbers to round.

    digits : int
        The number of significant digits to round to.

    Returns
    -------
    float, list, or np.ndarray
        The rounded numbers in the same format as the input.

    Examples
    --------
    >>> import numpy as np
    >>> statistics_array = np.array([0.12345, 6.789, 0.00567])
    >>> statistics_list = [0.12345, 6.789, 0.00567]
    >>> single_float = 0.12345

    Round each element to two significant digits

    >>> rounded_statistics_array = round_signif(statistics_array, 2)
    >>> rounded_statistics_list = round_signif(statistics_list, 2)
    >>> rounded_single_float = round_signif(single_float, 2)

    Print the rounded statistics

    >>> print(f'Statistics (array): {rounded_statistics_array}')
    Statistics (array): [0.12 6.8 0.0057]
    >>> print(f'Statistics (list): {rounded_statistics_list}')
    Statistics (list): [0.12, 6.8, 0.0057]
    >>> print(f'Single float: {rounded_single_float}')
    Single float: 0.12
    """
    if isinstance(numbers, float):
        # Round a single float
        return (
            float(round(numbers, digits - int(np.floor(np.log10(abs(numbers)))) - 1))
            if numbers != 0
            else 0.0
        )
    if isinstance(numbers, np.ndarray):
        # Apply rounding element-wise and return as a NumPy array
        return np.array(
            [
                float(round(num, digits - int(np.floor(np.log10(abs(num)))) - 1))
                if num != 0
                else 0.0
                for num in numbers
            ]
        )
    if isinstance(numbers, list):
        # Apply rounding element-wise and return as a list
        return [
            float(round(num, digits - int(np.floor(np.log10(abs(num)))) - 1))
            if num != 0
            else 0.0
            for num in numbers
        ]
    else:
        raise TypeError("Input must be a float, list, or NumPy array of floats.")


def hill_estimator(x, k):
    """
    Hill estimator for the tail index based on the data `x`, using the
    `k` largest order statistics.

    Parameters
    ----------
    x : array-like
        The input data sample from which to estimate the tail index.
    k : int, >= 2
        The number of extreme values to consider for the estimation.

    Returns
    -------
    dict
        A dictionary containing the Hill estimator value ('estimate') and its
        standard deviation ('sdev').
    """
    sorted_data = np.sort(x)[::-1]  # Sort data in decreasing order
    hill_estimate = 1 / (k - 1) * np.sum(np.log(sorted_data[:k] / sorted_data[k - 1]))
    standard_deviation = hill_estimate / np.sqrt(k - 1)
    return {"estimate": hill_estimate, "sdev": standard_deviation}


def rank_transform(x_raw):
    """
    Standardize each column of the input matrix using a Pareto-based
    rank transformation.

    This function transforms each element in the input matrix by assigning
    a rank based on its position within its column and then applying a
    Pareto transformation. The transformation is defined as:

    .. math::

        v_{\\text{rank},ij} = \\frac{n_{\\text{sample}} + 1}{\\text{rank}(x_{\\text{raw},ij}) + 1} = \\frac{1}{1 - \\frac{n}{n+1} F_{\\text{emp}}(x_{\\text{raw},ij})}

    where :math:`\\text{rank}(x_{\\text{raw},ij})` is the rank of the element
    :math:`x_{\\text{raw},ij}` within its column in decreasing order and
    :math:`F_{\\text{emp}}` is the usual empirical CDF.

    Parameters
    ----------
    x_raw : numpy.ndarray
        A 2D array where each column represents a feature and each row
        represents a sample. The function assumes that the input is a
        numerical matrix.

    Returns
    -------
    numpy.ndarray
        A transformed matrix of the same shape as `x_raw`, where each element
        has been standardized using the Pareto-based rank transformation.

    Example
    -------
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
    Transform each column in `x_test` to approximate a unit Pareto distribution
    based on the empirical cumulative distribution function (ECDF) resulting
    from the corresponding column in `x_train`.

    Parameters
    ----------
    x_train : numpy.ndarray
        A 2D array of shape (n, d) representing the training data.
    x_test : numpy.ndarray
        A 2D array of shape (m, d) representing the test data to be transformed.

    Returns
    -------
    numpy.ndarray
        A transformed version of `x_test` of shape (m, d), where each element is
        transformed using the formula:

        .. math::

            \\frac{1}{1 - \\frac{n}{n+1} F_{\\text{emp}}(x_{\\text{test},ij})}

    Example
    -------
    >>> import numpy as np
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
        ecdf = np.searchsorted(sorted_col, x_test[:, j], side="right") / (n_samples + 1)

        # Apply the transformation
        x_transf[:, j] = 1 / (1 - ecdf)

    return x_transf


def test_indep_radius_rest(
    X, y, ratio_ext, norm_func, random_state=np.random.randint(10**5)
):
    """Test the independence of the angular component and the\
    radius of extreme observations.

    This function performs a distance covariance test to assess the
    independence between the angular component and the radius of
    extreme observations. It uses a specified ratio to determine the
    threshold for extreme observations and performs the test for each
    ratio in `ratio_ext`.

    Parameters
    ----------
    X : array-like
        The input data matrix.
    y : array-like, optional
        The target vector. If None, only the angular component of `X` is
        considered.
    ratio_ext : float or list of float
        The ratio(s) used to determine the threshold for extreme observations.
    norm_func : callable
        A function that computes the norm of `X`.
    random_state : int, optional
        Seed for the random number generator. Default is a random integer.

    Returns
    -------
    tuple
        A tuple containing:
            - pvalues (ndarray): The p-values from the independence tests.
            - ratio_max (float): The maximum ratio for which the p-value is
                                 greater than 0.05.

    Notes
    -----
    - If `y` is None, only the angular component of `X` is considered.
    - For small sample sizes (< 100), a permutation-based test is used.
    - For larger sample sizes, an asymptotic test is used.
    - If no ratios satisfy the condition pvalues > 0.05, a warning is issued and ratio_max is set to 0.

    References
    ----------
    The method is inspired by the paper:
    
    Wan, P., & Davis, R. A. (2019). Threshold selection for multivariate heavy-tailed data. Extremes, 22(1), 131-166.
    """
    from dcor.independence import distance_covariance_test
    from dcor.independence import distance_correlation_t_test

    #
    if isinstance(ratio_ext, float):
        ratio_ext = [ratio_ext]
    norm_X = norm_func(X)
    Theta = X / norm_X.reshape(-1, 1)
    if y is None:
        Z = Theta
    else:
        Z = np.column_stack((Theta, y.reshape(-1, 1)))
    pvalues = []
    count = 0
    for ratio in ratio_ext:
        count += 173
        threshold = np.quantile(norm_X, 1 - ratio)
        id_extreme = norm_X >= threshold
        r_ext = norm_X[id_extreme]
        Z_ext = Z[id_extreme, :]
        if len(r_ext) < 100:
            # perform a distance covariance test with permutation-based
            # computation of the p-value
            test = distance_covariance_test(
                x=Z_ext,
                y=np.log(1 + r_ext).reshape(-1, 1),
                num_resamples=500,
                random_state=random_state + count,
            )
        else:
            # perform a distance covariance test with asymptotic p-value
            test = distance_correlation_t_test(
                x=Z_ext, y=np.log(1 + r_ext).reshape(-1, 1)
            )

        pvalues.append(test.pvalue)

    pvalues = np.array(pvalues)
    # Find indices where the condition is satisfied
    indices = np.where(pvalues > 0.05)[0]

    # Check if any indices satisfy the condition
    if len(indices) > 0:
        i_max = np.max(indices)
    else:
        warnings.warn("No indices satisfy the condition pvalues > 0.05.", UserWarning)

        i_max = 0

    ratio_max = ratio_ext[i_max]
    return pvalues, ratio_max


def plot_indep_radius_rest(pvalues, ratio_ext, ratio_max, n):
    """Plot the p-values from independence tests performed with \
    :func:`test_indep_radius_rest` against the number of extreme observations.

    This function creates a scatter plot of p-values from distance
    correlation tests against the number of extreme observations
    (k). It also highlights the selected k value based on a
    rule-of-thumb.

    Parameters
    ----------
    pvalues : array-like
        The p-values from the independence tests.
    ratio_ext : array-like
        The ratios used to determine the threshold for extreme observations.
    ratio_max : float
        The maximum ratio for which the p-value is greater than 0.05.
    n : int
        The total number of observations.

    Notes
    -----
    - The plot includes a secondary x-axis showing the ratio (k / n).
    - The selected k value is highlighted with a red line.
    - The plot title, labels, and legend are added for clarity.

    """
    kk = (ratio_ext * n).astype(int)
    k_max = int(ratio_max * n)
    fig, ax = plt.subplots()
    #    colors = ['green' if p > 0.05 else 'red' for p in pvalues]
    ax.scatter(kk, pvalues, c="black", label="distance correlation pvalues")
    ax.plot(
        k_max * np.ones(2),
        np.linspace(0, np.max(pvalues), num=2),
        c="red",
        label="selected k with distance covariance rule-of-thumb",
    )

    # Add a secondary x-axis with a different scale
    def ratio_transform(x):
        return x / n

    def inverse_ratio_transform(x):
        return x * n

    secax = ax.secondary_xaxis(
        "top", functions=(ratio_transform, inverse_ratio_transform)
    )
    secax.set_xlabel("Ratio (k / n)")
    ax.set_title("distance correlation test for extreme norm(X) Values vs Rest")
    ax.set_xlabel("k")
    ax.set_ylabel("p-value")
    ax.legend()
    plt.grid()
    plt.show()
