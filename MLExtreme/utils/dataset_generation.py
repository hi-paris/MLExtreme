# from scipy import stats as stat
# from scipy.stats import pareto
import numpy as np
from .EVT_basics import gen_dirimix, normalize_param_dirimix


def gen_rv_dirimix(alpha, Mu, wei, lnu, scale_weight_noise=1,
                   index_weight_noise=None, size=1):
    """
    Generate `n` points where each point `X_i = R_i * Theta_i`.

    Parameters:
    -----------
    size : int (optional)
        Number of samples to generate.
    alpha : float
        Shape parameter of the Pareto distribution.
    Mu : ndarray, shape (k,d)
        Means for the Dirichlet mixture components.
    wei : ndarray, shape (k,)
        Weights for each mixture component.
    lnu : ndarray, shape (k,)
        Log-scale parameters for the Dirichlet mixture components.
    scale_weight_noise : float, optional
        Scaling factor for the angular noise. Default is 1.
    index_weight_noise : float, optional
        Exponent ruling how the noise vanishes for large R. Default is -alpha.

    Returns:
    --------
    ndarray, shape (n, d)
        Array of points generated from the Pareto and Dirichlet mixture.
    """
    n = size
    if index_weight_noise is None:
        index_weight_noise = alpha

    R = np.random.pareto(alpha, size=n)+1
    Theta = gen_dirimix(Mu=Mu,  wei=wei, lnu=lnu, size=n)
    dim = np.shape(Mu)[1]
    noiseMu = (1 - Mu)/(dim - 1)
    Noise = gen_dirimix(Mu=noiseMu, wei=wei, lnu=lnu, size=n)
    # gen_dirichlet(n, ) np.ones((n, np.shape(Mu)[0])))
    w = scale_weight_noise * R ** (-index_weight_noise)
    newTheta = (1 - w[:, np.newaxis]) * Theta + w[:, np.newaxis] * Noise
    X = R[:, np.newaxis] * newTheta

    return X


def gen_classif_data_diriClasses(mu0=np.array([0.7, 0.3]),
                                 lnu=None, alpha=4,
                                 index_weight_noise=1,
                                 size=10):

    mu1 = 1-mu0
    Mu_temp = np.array([mu0, mu1])
    wei_temp = 0.5 * np.ones(2)
    Mu, wei = normalize_param_dirimix(Mu_temp, wei_temp)
    if lnu is None:
        lnu = np.log(2 / np.min(Mu, axis=1))        
    size0 =  int(size*wei[0])
    size1 = int(size*wei[1])
    Mu0 = Mu[0, :].reshape(1, -1)
    Mu1 = Mu[1, :].reshape(1, -1)
    data0 = gen_rv_dirimix(alpha, Mu0,
                           wei= np.array([1]),
                           lnu= np.array([lnu[0]]), scale_weight_noise=0.8,
                           index_weight_noise=index_weight_noise,
                           size=size0)
    data1 = gen_rv_dirimix(alpha, Mu1,
                           wei= np.array([1]),
                           lnu= np.array([lnu[1]]),
                           scale_weight_noise=0.8,
                           index_weight_noise=index_weight_noise,
                           size=size1)
    y = np.vstack((np.zeros(size0).reshape(-1, 1),
                  np.ones(size1).reshape(-1, 1))).flatten()
    X =  np.vstack((data0, data1))
    permut = np.random.permutation(size0 + size1)
    X = X[permut, :]
    y=y[permut]
    return X,  y


# ## target generation for regression models

def tail_reg_fun_default(angle):
    """
    Default tail regression function for generating target values.

    Parameters:
    angle (np.ndarray): A 1D or 2D array representing angles.

    Returns:
    np.ndarray: The result of the dot product between angle and a beta vector.
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
    """
    Default bulk regression function for generating target values.

    Parameters:
    angle (np.ndarray): A 1D or 2D array representing angles.

    Returns:
    np.ndarray: The result of the dot product between angle and a
      vector beta in a direction different from the tail's vector
    """
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([0.1 * np.ones(p1), 10 * np.ones(p - p1)])
    # beta = np.ones(p)
    return np.dot(angle, beta)


def bulk_decay_fun_default(radius):
    """
    Default bulk decay function for generating target values.

    Parameters:
    radius (np.ndarray): A 1D array representing radii.

    Returns:
    np.ndarray: The result of applying a decay function to the radius.
    """
    return 1 / (1+radius)**(1/2)


def gen_target_CovariateRV(X, tail_reg_fun=tail_reg_fun_default,
                           bulk_reg_fun=bulk_reg_fun_default,
                           bulk_decay_fun=bulk_decay_fun_default):
    """
    Generate target values for covariate random variables.

    Parameters:
     X (np.ndarray): A 2D array representing covariate data.
        tail_reg_fun (function): Tail regression function.
        bulk_reg_fun (function): Bulk regression function.
        bulk_decay_fun (function): Bulk decay function.

    Returns:
     y : the generated target values
    """
    rad = np.linalg.norm(X, axis=1)
    ang = X / rad[:, np.newaxis]
    n = len(rad)
    noise = np.random.normal(loc=0, scale=0.01, size=n)
    tail_mean = tail_reg_fun(ang).flatten()
    bulk_mean = bulk_reg_fun(ang).flatten()
    bulk_weight = bulk_decay_fun(rad).flatten()
    ystar = (1 - bulk_weight)*tail_mean + bulk_weight * bulk_mean
    y = ystar + noise
    return y


# ## toy example for functional PCA
def gen_rv_functional_data(num_samples, grid, alpha, alphanoise, scalenoise=5,
                           om1=1, om2=2, om3=3, om4=4):
    """
    Generate random regularly varying functions on a grid using
    Pareto-distributed coefficients and noise components.

    Parameters:
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

    Returns:
    --------
    ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row
    represents a generated functional data sample.
    """
    a1 = (np.random.pareto(alpha, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a2 = (np.random.pareto(alpha, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a3 = scalenoise * (np.random.pareto(alphanoise, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a4 = scalenoise * (np.random.pareto(alphanoise, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)

    result = np.dot(a1[:, None],
                    (1 + np.sin(2 * np.pi * om1 * grid))[None, :]) + \
            np.dot(a2[:, None],
                   (1 + np.cos(2 * np.pi * om2 * grid))[None, :]) + \
            np.dot(a3[:, None],
                    (1 + np.sin(2 * np.pi * om3 * grid))[None, :]) + \
            np.dot(a4[:, None],
                    (1 + np.cos(2 * np.pi * om4 * grid))[None, :])

    return result


def gen_rv_functional_data_gaussianNoise(num_samples, grid, alpha, sd,
                                         scalenoise=5, om1=1, om2=2,
                                         om3=3, om4=4, om5=5, om6=6):
    """
    Generate random regularly varying functions on a grid using
    Pareto-distributed coefficients and Gaussian noise components.

    Parameters:
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

    Returns:
    --------
    ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row
        represents a generated data sample.
    """
    a1 = (np.random.pareto(alpha, size=num_samples) + 1)
    a2 = 0.8 * (np.random.pareto(alpha, size=num_samples) + 1)
    a3 = scalenoise * np.random.normal(0, sd, size=num_samples)
    a4 = 0.8 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a5 = 0.6 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a6 = 0.4 * scalenoise * np.random.normal(0, sd, size=num_samples)

    result = np.dot(a1[:, None],
                    (np.sqrt(2) * np.sin(2 * np.pi * om1 * grid))[None, :]) + \
             np.dot(a2[:, None],
                    (np.sqrt(2) * np.cos(2 * np.pi * om2 * grid))[None, :]) + \
             np.dot(a3[:, None],
                    (np.sqrt(2) * np.sin(2 * np.pi * om3 * grid))[None, :]) + \
             np.dot(a4[:, None],
                    (np.sqrt(2) * np.cos(2 * np.pi * om4 * grid))[None, :]) + \
             np.dot(a5[:, None],
                    (np.sqrt(2) * np.sin(2 * np.pi * om5 * grid))[None, :]) + \
             np.dot(a6[:, None],
                    (np.sqrt(2) * np.cos(2 * np.pi * om6 * grid))[None, :])

    return result
