import numpy as np

def hill_estimator(sample_data, num_extremes):
    """
    Estimate the tail index of a distribution using the Hill estimator.

    Parameters:
    -----------
    sample_data : array-like
        The input data sample from which to estimate the tail index.

    num_extremes : int
        The number of extreme values to consider for the estimation.

    Returns:
    --------
    dict
        A dictionary containing the Hill estimator value ('test') and its standard deviation ('sdev').
    """
    sorted_data = np.sort(sample_data)[::-1]  # Sort data in decreasing order
    hill_estimate = 1 / num_extremes * np.sum(np.log(sorted_data[:num_extremes] / sorted_data[num_extremes - 1]))
    standard_deviation = hill_estimate / np.sqrt(num_extremes)
    return {"estimate": hill_estimate, "standard_deviation": standard_deviation}

def generate_pareto_samples(num_samples, alpha):
    """
    Generate samples from a Pareto distribution.

    Parameters:
    -----------
    num_samples : int
        The number of samples to generate.

    alpha : float
        The shape parameter of the Pareto distribution.

    Returns:
    --------
    array
        An array of samples drawn from the Pareto distribution.
    """
    uniform_samples = np.random.uniform(size=num_samples)
    pareto_samples = (1 - uniform_samples)**(-1 / alpha)
    return pareto_samples

def pareto_quantile(prob, alpha):
    """
    Calculate the quantile function (inverse CDF) of a Pareto distribution.

    Parameters:
    -----------
    prob : array-like
        The probability values at which to evaluate the quantile function.

    alpha : float
        The shape parameter of the Pareto distribution.

    Returns:
    --------
    array
        The quantile values corresponding to the input probabilities.
    """
    return (1 - prob)**(-1 / alpha)

def pareto_cdf(x, alpha, scale=1):
    """
    Calculate the cumulative distribution function (CDF) of a Pareto distribution.

    Parameters:
    -----------
    x : array-like
        The values at which to evaluate the CDF.

    alpha : float
        The shape parameter of the Pareto distribution.

    scale : float, optional
        The scale parameter of the Pareto distribution. Default is 1.

    Returns:
    --------
    array
        The CDF values corresponding to the input values.
    """
    return (x / scale)**(-alpha)

