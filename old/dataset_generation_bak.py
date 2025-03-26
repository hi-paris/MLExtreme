from scipy.stats import pareto
import numpy as np
from .EVT_basics import *
# from scipy.stats import gamma
from scipy.special import gamma, betaln, loggamma
import scipy.stats as stat
import matplotlib.pyplot as plt


# Multivariate symmetric logistic Model
def gen_PositiveStable(n, alpha, seed=np.random.randint(2**31)):# From Nathan Huet
    """
    Generate positive stable random variables.
    see "Simulating Multivariate Extreme Value Distributions of Logistic Type", 2003, A. Stephenson for more details

    Inputs:
    -----------------
    - n : sample size
    - alpha : dependence parameter
    - seed : random seed

    Output:
    ------------------
    - Sample of positive stable random variables
    """
    U=stat.uniform.rvs(0,np.pi,size=n,random_state = seed)
    W=stat.expon.rvs(size=n,random_state = seed)
    Term_1=(np.sin((1-alpha)*U)/W)**((1-alpha)/alpha)
    Term_2=np.sin(alpha*U)/((np.sin(U))**(1/alpha))
    return Term_1*Term_2


def gen_multilog(n,Dim,alpha,seed=np.random.randint(2**31)):
    # From Nathan Huet
    
    """
    Generate multivariate logistic random variables.  

    Inputs:
    ---------------------
    - n : sample size
    - Dim : dimension
    - alpha : dependence parameter
    - seed : random seed

    Output:
    ----------------------
    - Sample of multivariate logistic random variables

    Details:
    ----------------------
     Implementation of Algorithm 2.1 in  "Stephenson, A. (2003). Simulating multivariate extreme value distributions of logistic type. Extremes, 6, 49-59.". 
    
    """
    W=stat.expon.rvs(size=(n,Dim),random_state = seed) ### Standard exponential Random Variables

    S=gen_PositiveStable(n,alpha,seed) 
    Result=np.zeros((n,Dim))
    for ii in range(Dim):
        Result[:,ii]=(S/W[:,ii])**alpha
    return Result 
    

################################################################
##  Dirichlet mixture model 
################################################################
## generation of multivariate regularly varying datasets with angular component following  dirichlet mixtures


def gen_dirichlet(n, alpha):
    """
    Generate angular random samples from a Dirichlet distribution.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    alpha : ndarray, shape (n, p)
        The concentration parameters for the Dirichlet distribution. Each row
        of `alpha` corresponds to a set of concentration parameters for a Dirichlet
        distribution.

    Returns
    -------
    ndarray, shape (n, p)
        An array of shape (n, p) where each row is a sample from the Dirichlet
        distribution. Each sample is normalized to sum to 1.
    """
    x = np.random.gamma(alpha, size=(n, len(alpha[0])))
    sm = np.sum(x, axis=1)
    return x / sm[:, None]  # Normalize each row to sum to 1


# def pdf_dirichlet_2D(x, alpha): ## to be removed?
#     """
#     Designed for 2D problems 
#     Compute the Dirichlet probability density function on the 1D simplex for a given alpha vector of size 2.
    
#     Parameters
#     ----------
#     x : ndarray, shape (2,)
#         A point on the 1-D simplex, i.e., (x, 1-x).
#     alpha : ndarray, shape (2,)
#         The concentration parameters for the Dirichlet distribution.
    
#     Returns
#     -------
#     float
#         The probability density at point `x` for the given Dirichlet distribution.
#     """
#     # Normalize the input (x, 1-x) is the simplex
#     x1 = x
#     x2 = 1 - x

#     # Compute the Dirichlet PDF
#     alpha1, alpha2 = alpha
#     return (x1**(alpha1-1) * x2**(alpha2-1)) / (gamma(alpha1) * gamma(alpha2) / gamma(alpha1 + alpha2))

def pdf_dirichlet(x, alpha):
    """
    Compute the Dirichlet probability density function for each row in the data matrix `x`,
    where each row is a point on the (d-1)-dimensional simplex. If `x` is a 1D array,
    it is treated as a single point and converted to a 2D array.

    Parameters
    ----------
    x : ndarray, shape (n, d) or (d,)
        A data matrix where each row is a point on the (d-1)-dimensional simplex,
        i.e., each row sums to 1. If `x` is a 1D array, it is treated as a single point.
    alpha : ndarray, shape (d,)
        The concentration parameters for the Dirichlet distribution.

    Returns
    -------
    ndarray, shape (n,)
        The probability density at each row of `x` for the given Dirichlet distribution.
    """
    # Convert x to a 2D array if it is 1D
    if x.ndim == 1:
        x = x[np.newaxis, :]

    # Ensure each row of x is a valid point on the simplex
    assert np.allclose(np.sum(x, axis=1), 1), "Each row of x must sum to 1"
    assert x.shape[1] == len(alpha), "Each row of x and alpha must have the same length"

    # Compute the log of the Dirichlet PDF for each row
    log_numerator = np.sum((alpha - 1) * np.log(x), axis=1)
    log_denominator = np.sum(loggamma(alpha)) - loggamma(np.sum(alpha))

    log_pdf = log_numerator - log_denominator

    # Exponentiate to get the final PDF values
    return np.exp(log_pdf)


# Example usage
# x_single = np.array([0.2, 0.3, 0.5])
# x_matrix = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
# alpha = np.array([1.5, 2.0, 3.0])

# density_single = pdf_dirichlet(x_single, alpha)
# density_matrix = pdf_dirichlet(x_matrix, alpha)

# print("Density for single point:", density_single)
# print("Densities for matrix:", density_matrix)


def pdf_dirimix(x, Mu, lnu, wei):
    """
    Compute the mixture density on the (d-1)-dimensional simplex for each row in the data matrix `x`.
    If `x` is a 1D array, it is treated as a single point and converted to a 2D array.

    Parameters
    ----------
    x : ndarray, shape (n, d) or (d,)
        A data matrix where each row is a point on the (d-1)-dimensional simplex,
        i.e., each row sums to 1. If `x` is a 1D array, it is treated as a single point.
    Mu : ndarray, shape (d, k)
        The matrix of means (each column is a vector for the Dirichlet concentration parameters).
    lnu : ndarray, shape (k,)
        The vector of log-scales (lnu), one for each mixture component.
    wei : ndarray, shape (k,)
        The vector of weights for each mixture component.

    Returns
    -------
    ndarray, shape (n,)
        The density at each row of `x` on the (d-1)-dimensional simplex.
    """
    # Convert x to a 2D array if it is 1D
    if x.ndim == 1:
        x = x[np.newaxis, :]

    # Number of components in the mixture
    k = len(wei)

    # Evaluate the density for each mixture component
    density = np.zeros(x.shape[0])
    for i in range(k):
        # Compute the effective concentration parameters for each mixture component
        alpha = Mu[:, i] * np.exp(lnu[i])
        # Evaluate the Dirichlet density at each row of x (on the simplex)
        density += wei[i] * pdf_dirichlet(x, alpha)

    return density

# Example usage
# x_single = np.array([0.2, 0.3, 0.5])
# x_matrix = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
# Mu = np.array([[1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
# lnu = np.array([0.5, 1.0])
# wei = np.array([0.6, 0.4])

# density_single = pdf_dirimix(x_single, Mu, lnu, wei)
# density_matrix = pdf_dirimix(x_matrix, Mu, lnu, wei)

# print("Density for single point:", density_single)
# print("Densities for matrix:", density_matrix)


def gen_dirimix(n, Mu, lnu, wei):
    """
    Generate angular samples from a mixture of Dirichlet distributions.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    Mu : ndarray, shape (p, k)
        The matrix of means, where `p` is the number of components,
        and `k` is the  number of mixtures.
    lnu : ndarray, shape (k,)
        The vector of log-scales (lnu), one for each mixture component.
    wei : ndarray, shape (k,)
        The vector of weights for each mixture component.

    Returns
    -------
    ndarray, shape (n, p)
        An array of Dirichlet samples of shape (n, p), each generated from the
        mixture of Dirichlet distributions.
    """
    # Ensure Mu is a 2D matrix if it is a vector
    p, k = Mu.shape
    if len(lnu) != k or len(wei) != k:
        raise ValueError("Length of lnu and wei must be equal to k")

    # Step 1: Generate random mixture component assignments
    u = np.random.rand(n)
    cum_wei = np.cumsum(wei)

    ms = np.array(
        [
            np.where(cum_wei < u[i])[0].max() if np.any(cum_wei < u[i]) else k - 1
            for i in range(n)
        ]
    )

    # Step 2: Compute parameters for Dirichlet distribution for each sample
    matpars = np.array([Mu[:, ms[i]] * np.exp(lnu[ms[i]]) for i in range(n)])

    # Step 3: Generate Dirichlet samples
    return gen_dirichlet(n, matpars)


def plot_pdf_dirimix_2D(Mu, lnu, wei, n_points=500):
    """
    Plot the mixture density on the 1-D simplex in ambient dimension 2.
    
    Parameters
    ----------
    Mu : ndarray, shape (2, k)
        The matrix of means (each column is a vector for the Dirichlet concentration parameters).
    lnu : ndarray, shape (k,)
        The vector of log-scales (lnu), one for each mixture component.
    wei : ndarray, shape (k,)
        The vector of weights for each mixture component.
    n_points : int, optional
        The number of points to use for evaluating the density. Default is 500.
    """
    # Generate points on the 1-D simplex
    x_values1 = np.linspace(0, 1, n_points)
    x = np.column_stack((x_values1, 1 - x_values1))
    # Compute the density for each point
    density_values = pdf_dirimix(x, Mu, lnu, wei) 
    
    # Plot the density
    plt.figure(figsize=(8, 6))
    plt.plot(x_values1, density_values, label="Mixture Density", color='blue')
    plt.fill_between(x_values1, density_values, alpha=0.3, color='blue')
    plt.title("Mixture Density on the 1-D Simplex")
    plt.xlabel(" x[1] first component ")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()


# def pdf_dirichlet_3d(x, alpha):
#     """
#     Compute the Dirichlet probability density function for a 3-component Dirichlet distribution
#     at a point on the 2D simplex (x1, x2, x3) where x3 = 1 - x1 - x2.
    
#     Parameters
#     ----------
#     x : ndarray, shape (2,)
#         A point on the 2D simplex, i.e., (x1, x2).
#     alpha : ndarray, shape (3,)
#         The concentration parameters for the Dirichlet distribution.
    
#     Returns
#     -------
#     float
#         The probability density at point `x` for the given Dirichlet distribution.
#     """
#     x1, x2 = x
#     x3 = 1 - x1 - x2  # x3 is determined by x1 and x2
    
#     # Dirichlet PDF formula for three components
#     alpha1, alpha2, alpha3 = alpha
#     return (x1**(alpha1 - 1) * x2**(alpha2 - 1) * x3**(alpha3 - 1)) / (
#         gamma(alpha1) * gamma(alpha2) * gamma(alpha3) / gamma(alpha1 + alpha2 + alpha3)
#     )

# def pdf_dirimix_3d(x, Mu, lnu, wei):
#     """
#     Compute the mixture density on the 2D simplex (in ambient dimension 3) at point `x`.
    
#     Parameters
#     ----------
#     x : ndarray, shape (2,)
#         A point on the 2D simplex (x1, x2), where x3 is determined as 1 - x1 - x2.
#     Mu : ndarray, shape (3, k)
#         The matrix of means (each column is a vector for the Dirichlet concentration parameters).
#     lnu : ndarray, shape (k,)
#         The vector of log-scales (lnu), one for each mixture component.
#     wei : ndarray, shape (k,)
#         The vector of weights for each mixture component.
    
#     Returns
#     -------
#     float
#         The density at point `x` on the 2D simplex.
#     """
#     k = len(wei)
#     density = 0
#     for i in range(k):
#         # Compute the effective concentration parameters for each mixture component
#         alpha = Mu[:, i] * np.exp(lnu[i])
#         # Evaluate the Dirichlet density at the point x (on the simplex)
#         density += wei[i] * pdf_dirichlet_3d(x, alpha)
    
#     return density

def plot_pdf_dirimix_3D(Mu, lnu, wei, n_points=500):
    """
    Plot the mixture density on the 2D simplex (represented as an equilateral triangle).
    
    Parameters
    ----------
    Mu : ndarray, shape (3, k)
        The matrix of means (each column is a vector for the Dirichlet concentration parameters).
    lnu : ndarray, shape (k,)
        The vector of log-scales (lnu), one for each mixture component.
    wei : ndarray, shape (k,)
        The vector of weights for each mixture component.
    n_points : int, optional
        The number of points to use for evaluating the density. Default is 500.
    """
    # Generate points on the 2D simplex using a meshgrid (x1, x2) where x3 = 1 - x1 - x2
    x1_values = np.linspace(0, 1, n_points)
    x2_values = np.linspace(0, 1, n_points)
    
    X1, X2 = np.meshgrid(x1_values, x2_values)
    X1 = X1.flatten()
    X2 = X2.flatten()
    
    # Remove points where x1 + x2 > 1 (because x3 would be negative)
    valid_points = X1 + X2 <= 1
    X1 = X1[valid_points]
    X2 = X2[valid_points]
    X_full = np.column_stack((X1, X2, 1 - X1 - X2))
    # Compute the density for each point on the simplex
    density_values = pdf_dirimix(X_full, Mu, lnu, wei)
    
    # Plot the density as a heatmap
    plt.figure(figsize=(8, 6))
    plt.tricontourf(X1, X2, density_values, 20, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title("Mixture Density on the 2D Simplex")
    plt.xlabel("x1 (first component)")
    plt.ylabel("x2 (second component)")
    
    # Plot the simplex as a triangle for visual reference
    plt.plot([0, 1, 0], [1, 0, 0], 'k--', lw=2)  # Simplex boundary
    
    plt.grid(True)
    plt.show()




    

def gen_rv_dirimix(n, alpha, Mu, wei, lnu,  scale_weight_noise=1, index_weight_noise = None):
    """
    Generate `n` points where each point `X_i = R_i * Theta_i`:
    - `R_i` is a Pareto-distributed variable with a minimum value of `x_min`
    - `Theta_i` is a point on the simplex generated by a Dirichlet mixture with parameters `(Mu,wei,lnu)` (see gen_dirimix), with an additive angular noise which vanishes for asymptotically large R_i. The angular noise is uniformly distributed on the simplex.
    

    Parameters
    ----------
    n : int
        The number of samples to generate.
    alpha : float
        The shape parameter of the Pareto distribution.
    Mu : ndarray, shape (p, k)
        The `p`-dimensional means for the `k` components of the Dirichlet mixture.
    wei : ndarray, shape (k,)
        The weights for each component of the Dirichlet mixture.
    lnu : ndarray, shape (k,)
        The log-scale parameters for the Dirichlet mixture components.
    scale_weight_noise : float, in [0,1], optional
        The scaling factor for the angular noise (default is 1).
    index_weight_noise : float, >0,  optional
        The exponent ruling how the noise vanishes for large R (default is alpha).

    Returns
    -------
    ndarray, shape (n, p)
        A NumPy array of `n` points where each point is the product of a Pareto variable
        and a Dirichlet-distributed point on the simplex (with a noise).
    """
    # assign  argument index_weight_noise
    if index_weight_noise is None:
        index_weight_noise = alpha
        
    # Generate Pareto samples with the specified x_min
    R =  pareto.rvs(alpha, size = n)## generate_univariate_pareto(n, alpha, x_min)

    # Generate points on the simplex (Dirichlet mixture samples)
    Theta = gen_dirimix(n=n, Mu=Mu, lnu=lnu, wei=wei)

    # generate angular noise
    Noise = gen_dirichlet(n, np.ones((n,np.shape(Mu)[0])))
    # compute noise weights 
    w = scale_weight_noise  * R **(-index_weight_noise)
    # deduce noisy angle  
    newTheta = (1- w[:, np.newaxis]) * Theta +  w[:, np.newaxis] * Noise

    
    # Multiply each Pareto variable with its corresponding Dirichlet point (element-wise)
    X = R[:, np.newaxis] * newTheta

    return X








######################################
## toy examples for classification
######################################

def gen_label(Matrix, angle=0.2):
    """
    Generate labels for a 2D toy example with an explicit decision boundary.

    This function assigns labels to points in a 2D matrix based on their angle with respect to the positive x-axis.
    Points within a specified angular range are labeled as 1, while others are labeled as 0.

    Parameters
    ----------
    Matrix : ndarray, shape (n, 2)
        A 2D array where each row represents a point in 2D space.
    angle : float, optional
        The angular range parameter, specified as a fraction of π/2. Points within this range are labeled as 1.
        Default is 0.2.

    Returns
    -------
    ndarray, shape (n,)
        An array of labels (0 or 1) for each point in the input Matrix.

    Example
    -------
    >>> Matrix = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]])
    >>> gen_label(Matrix, angle=0.2)
    array([0, 0, 0, 0, 1])
    """
    # Calculate the angle of each point with respect to the positive x-axis
    Vect_angle = np.arctan2(Matrix[:, 1], Matrix[:, 0])

    # Assign labels based on the angle
    label = [
        0 if (point_angle < (angle * np.pi / 2) or point_angle > ((1 - angle) * np.pi / 2)) else 1
        for point_angle in Vect_angle
    ]

    return np.array(label)

# # Example usage
# Matrix = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1,1], [-1,-1]])
# labels = gen_label(Matrix, angle=0.2)
# print(labels)


import numpy as np

def gen_label2(Matrix, angle=0.2):
    """
    Generate labels for a 2D toy example with an explicit decision boundary.

    This function assigns labels to points in a 2D matrix based on their angle with respect to the positive x-axis.
    Points within specified angular ranges are labeled as 0, while others are labeled as 1.

    Parameters
    ----------
    Matrix : ndarray, shape (n, 2)
        A 2D array where each row represents a point in 2D space.
    angle : float, optional
        The angular range parameter, specified as a fraction of π/4. Points within this range are labeled as 0.
        Default is 0.2.

    Returns
    -------
    ndarray, shape (n,)
        An array of labels (0 or 1) for each point in the input Matrix.

    Example
    -------
    >>> Matrix = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    >>> gen_label2(Matrix, angle=0.2)
    array([0, 1, 1, 1])
    """
    # Calculate the angle of each point with respect to the positive x-axis
    Vect_angle = np.arctan2(Matrix[:, 1], Matrix[:, 0])

    # Assign labels based on the angle
    label = [
        0 if (
            (0 <= point_angle < angle * np.pi / 4) or
            ((1 - angle) * np.pi / 4 <= point_angle < np.pi / 4) or
            (np.pi / 4 + angle * np.pi / 4 <= point_angle < np.pi / 4 + (1 - angle) * np.pi / 4)
        ) else 1
        for point_angle in Vect_angle
    ]

    return np.array(label)

# Example usage
# Matrix = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
# labels = gen_label2(Matrix, angle=0.2)
# print(labels)





######################################
## toy examples for functional PCA 
######################################
def gen_rv_functional_data(num_samples, grid, alpha, alphanoise, scalenoise=5, om1=1, om2=2, om3=3, om4=4):
    """
    Toy example #1  for functional PCA. Generates random regularly varying functions on a grid,  using Pareto-distributed coefficients and noise components
    with sinusoidal basis functions. The noise components are meant to have  lighter tails than the signal. 


    This function generates `num_samples`  samples  using a combination of:
    - Pareto-distributed random variables (`scipy.stats.pareto.rvs()`)
    - Sine and cosine basis functions with different frequencies (`om1`, `om2`, `om3`, `om4`)
    - Additional noise components controlled by `alphanoise` and `scalenoise`

    Parameters:
    ----------
    num_samples : int
        Number of samples to generate.

    grid : array-like
        A 1D array representing the grid over which the basis functions are evaluated.

    alpha : float
        Shape parameter for the 'signal' Pareto-distributed variables (`a1`, `a2`).

    alphanoise : float
        Shape parameter for the noise Pareto-distributed variables (`a3`, `a4`). in principle alphanoise>alpha. 

    scalenoise : float, optional (default=5)
        Scaling factor for the noise components.

    om1, om2, om3, om4 : float, optional (default=1, 2, 3, 4)
        Frequencies used in the sine and cosine basis functions.

    Returns:
    -------
    result : ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row represents 
        a generated functional data sample.

    Notes:
    ------
    - The random coefficients `a1`, `a2`, `a3`, and `a4` are symmetrically distributed around zero.
    - The resulting dataset is constructed using sinusoidal basis functions with different frequencies.

    Example:
    --------
    ```python
    import numpy as np

    # Define grid
    grid = np.linspace(0, 1, 100)

    # Generate data
    data = gen_rv_functional_data(num_samples=500, grid=grid, alpha=2, alphanoise=3)

    print(data.shape)  # (500, 100)
    ```
    """
    a1 = pareto.rvs(alpha, size= num_samples) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a2 = pareto.rvs(alpha, size= num_samples) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a3 = scalenoise * pareto.rvs(alphanoise, size= num_samples) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a4 = scalenoise * pareto.rvs(alphanoise, size= num_samples) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    
    result = np.dot(a1[:, None], (1 + np.sin(2 * np.pi * om1 * grid))[None, :]) + \
             np.dot(a2[:, None], (1 + np.cos(2 * np.pi * om2 * grid))[None, :]) + \
             np.dot(a3[:, None], (1 + np.sin(2 * np.pi * om3 * grid))[None, :]) + \
             np.dot(a4[:, None], (1 + np.cos(2 * np.pi * om4 * grid))[None, :])
    
    return result

def gen_rv_functional_data_gaussianNoise(num_samples, grid, alpha, sd, scalenoise=5, om1=1, om2=2, om3=3, om4=4, om5=5, om6=6):#ACP
    """
    Toy example #2  for functional PCA. Generates random regularly varying functions on a grid,  using Pareto-distributed coefficients and Gaussian noise components
    with sinusoidal basis functions.

    This function generates `num_samples` data points using a combination of:
    - Pareto-distributed random variables (`scipy.stats.pareto.rvs()`)
    - Gaussian noise components controllehd by `sd` and `scalenoise`
    - Sine and cosine basis functions with different frequencies (`om1` to `om6`)

    Parameters:
    ----------
    num_samples : int
        Number of samples to generate.

    grid : array-like
        A 1D array representing the grid over which the basis functions are evaluated.

    alpha : float (>0)
        Shape parameter for the Pareto-distributed variables (`a1`, `a2`).

    sd : float
        Standard deviation for the Gaussian noise components (`a3`, `a4`, `a5`, `a6`).

    scalenoise : float, optional (default=5)
        Scaling factor for the noise components.

    om1, om2, om3, om4, om5, om6 : float, optional (default=1, 2, 3, 4, 5, 6)
        Frequencies used in the sine and cosine basis functions.

    Returns:
    -------
    result : ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row represents 
        a generated data sample.

    Notes:
    ------
    - Unlike `gen_rv_functional_data`, this function includes **Gaussian noise**
      components instead of Pareto noise for variables `a3`, `a4`, `a5`, and `a6`.
    - The sine and cosine components are scaled by `np.sqrt(2)` for normalization.
    - The function combines six components (instead of four in
     `gen_rv_functional_data`) using different frequencies.
    
    Example:
    --------
    ```python
    import numpy as np

    # Define grid
    grid = np.linspace(0, 1, 100)

    # Generate data
    data = gen_rv_functional_data_gaussianNoise(num_samples=500, grid=grid, alpha=2, sd=0.5)

    print(data.shape)  # (500, 100)
    ```
    """
    a1 = pareto.rvs(alpha, size= num_samples)
    a2 = 0.8 * pareto.rvs(alpha, size= num_samples)
    a3 = scalenoise * np.random.normal(0, sd, size=num_samples)
    a4 = 0.8 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a5 = 0.6 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a6 = 0.4 * scalenoise * np.random.normal(0, sd, size=num_samples)

    result = np.dot(a1[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om1 * grid))[None, :]) + \
             np.dot(a2[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om2 * grid))[None, :]) + \
             np.dot(a3[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om3 * grid))[None, :]) + \
             np.dot(a4[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om4 * grid))[None, :]) + \
             np.dot(a5[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om5 * grid))[None, :]) + \
             np.dot(a6[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om6 * grid))[None, :])
    
    return result

